"""Cross-speaker evaluation on six verified Common Voice test clips.

Audio is read from D: and mixed with synthetic noise in memory. No model
training or Q75 threshold tuning occurs. Results are saved in a separate CSV.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from compare_multinoise_strategies import (
    DATA_ROOT, INPUT_SNRS, MODEL_DIR, NOISE_TYPES, calculate_cer,
    calculate_stoi, choose_limit, complex_mix, magnitude_clamp,
    metrics, synthesize,
)
from df.enhance import df_features, init_df, load_audio
from evaluate_cer import normalize


DATASET_DIR = (
    DATA_ROOT / "own_voice_evaluation" / "1784653411106-zh-CN-beijing"
    / "zh-CN-beijing"
)
SELECTED_PATHS = (
    "common_voice_zh-CN_22050000.mp3",
    "common_voice_zh-CN_18757799.mp3",
    "common_voice_zh-CN_22052721.mp3",
    "common_voice_zh-CN_21090986.mp3",
    "common_voice_zh-CN_18977976.mp3",
    "common_voice_zh-CN_22052863.mp3",
)
RESULT_DIR = Path(__file__).resolve().parents[1] / "results"


def selected_metadata():
    """Return the chosen test rows in selection order, rejecting duplicates."""
    with (DATASET_DIR / "test.tsv").open(encoding="utf-8", newline="") as handle:
        rows = {row["path"]: row for row in csv.DictReader(handle, delimiter="\t")}
    missing = [name for name in SELECTED_PATHS if name not in rows]
    if missing:
        raise ValueError(f"Selected clips not found in test.tsv: {missing}")
    selected = [rows[name] for name in SELECTED_PATHS]
    if len({row["client_id"] for row in selected}) != len(selected):
        raise ValueError("Selected clips must have distinct client_id values")
    for row in selected:
        if not (DATASET_DIR / "clips" / row["path"]).is_file():
            raise FileNotFoundError(DATASET_DIR / "clips" / row["path"])
        if not normalize(row["sentence"]):
            raise ValueError(f"Empty transcript: {row['path']}")
    return selected


def make_noises(clean, sample_rate, seed):
    generator = torch.Generator().manual_seed(seed)
    white = torch.randn(clean.shape, generator=generator)
    kernel = torch.ones(129, dtype=clean.dtype) / 129
    low_frequency = F.conv1d(
        white.unsqueeze(0), kernel.reshape(1, 1, -1), padding=64
    ).squeeze(0)
    time_axis = torch.arange(clean.shape[-1], dtype=clean.dtype) / sample_rate
    tonal = (
        torch.sin(2 * torch.pi * 300 * time_axis)
        + 0.5 * torch.sin(2 * torch.pi * 1000 * time_axis)
    ).unsqueeze(0)
    return {"white": white, "low_frequency": low_frequency, "tonal": tonal}


def mix_at_snr(clean, raw_noise, target_snr_db):
    clean_power = torch.mean(clean**2)
    noise_power = torch.mean(raw_noise**2)
    ratio = 10 ** (target_snr_db / 10)
    scale = torch.sqrt(clean_power / (ratio * noise_power + 1e-8))
    return clean + raw_noise * scale


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-asr", action="store_true",
        help="Check SNR and STOI without Whisper; CER columns will be empty.",
    )
    args = parser.parse_args()
    selected = selected_metadata()
    model, df_state, _ = init_df(model_base_dir=str(MODEL_DIR), log_file=None)
    model.eval()
    device = next(model.parameters()).device
    sample_rate = df_state.sr()
    asr_model = None
    if not args.skip_asr:
        import whisper
        asr_model = whisper.load_model(
            "small", device=str(device),
            download_root=str(DATA_ROOT / "models" / "Whisper"),
        )

    results = []
    for clip_index, metadata in enumerate(selected):
        audio_path = DATASET_DIR / "clips" / metadata["path"]
        clean, _ = load_audio(str(audio_path), sr=sample_rate)
        clean = clean.mean(dim=0, keepdim=True)
        # Keep all six sources at the same peak level before adding noise.
        peak = clean.abs().max().item()
        if peak == 0:
            raise ValueError(f"Silent reference: {audio_path}")
        clean = clean * (0.3 / peak)
        reference = normalize(metadata["sentence"])
        clean_cer = None
        clean_text = None
        if asr_model is not None:
            clean_cer, clean_text = calculate_cer(asr_model, clean, sample_rate, reference)
        raw_noises = make_noises(clean, sample_rate, seed=2026 + clip_index)
        print(
            f"音频 {clip_index + 1}/6: {metadata['path']}, "
            f"{clean.shape[-1] / sample_rate:.2f} 秒, 干净 CER={clean_cer}",
            flush=True,
        )

        for noise_type in NOISE_TYPES:
            for target_snr in INPUT_SNRS:
                noisy = mix_at_snr(clean, raw_noises[noise_type], target_snr)
                if noisy.abs().max().item() > 1.0:
                    raise ValueError(
                        f"Noisy waveform exceeds [-1,1]: {metadata['path']}, "
                        f"{noise_type}, {target_snr} dB"
                    )
                df_state.reset()
                padded = F.pad(noisy, (0, df_state.fft_size()))
                spec, erb_feature, spec_feature = df_features(
                    padded, df_state, model.nb_df, device=device
                )
                if hasattr(model, "reset_h0"):
                    model.reset_h0(batch_size=noisy.shape[0], device=device)
                with torch.no_grad():
                    enhanced_spec, _, local_snr, _ = model(
                        spec.clone(), erb_feature, spec_feature
                    )
                    q75 = torch.quantile(local_snr[0, :, 0], 0.75).item()
                    chosen_limit = choose_limit(q75)
                    candidates = {
                        "full": enhanced_spec,
                        "fixed_mix_12db": complex_mix(spec, enhanced_spec, 12),
                        "fixed_mix_6db": complex_mix(spec, enhanced_spec, 6),
                        "adaptive_mix": complex_mix(spec, enhanced_spec, chosen_limit),
                        "adaptive_clamp": magnitude_clamp(
                            spec, enhanced_spec, chosen_limit
                        ),
                    }

                input_snr, _ = metrics(noisy, clean)
                input_stoi = calculate_stoi(clean, noisy, sample_rate)
                input_cer = None
                input_text = None
                if asr_model is not None:
                    input_cer, input_text = calculate_cer(
                        asr_model, noisy, sample_rate, reference
                    )

                for strategy, candidate in candidates.items():
                    output = synthesize(df_state, candidate, noisy.shape[-1])
                    if output.shape != clean.shape:
                        raise ValueError(f"Output length mismatch: {metadata['path']}")
                    output_snr, output_mse = metrics(output, clean)
                    output_stoi = calculate_stoi(clean, output, sample_rate)
                    output_cer = None
                    output_text = None
                    if asr_model is not None:
                        output_cer, output_text = calculate_cer(
                            asr_model, output, sample_rate, reference
                        )
                    results.append({
                        "clip_path": metadata["path"],
                        "client_id": metadata["client_id"],
                        "reference_text": metadata["sentence"],
                        "noise_type": noise_type,
                        "target_snr_db": target_snr,
                        "input_snr_db": input_snr,
                        "input_stoi": input_stoi,
                        "input_cer": input_cer,
                        "clean_cer": clean_cer,
                        "q75_db": q75,
                        "chosen_limit_db": "none" if chosen_limit is None else chosen_limit,
                        "strategy": strategy,
                        "output_snr_db": output_snr,
                        "output_stoi": output_stoi,
                        "output_cer": output_cer,
                        "stoi_gain": output_stoi - input_stoi,
                        "snr_gain_db": output_snr - input_snr,
                        "output_mse": output_mse,
                        "clean_asr_text": clean_text,
                        "input_asr_text": input_text,
                        "output_asr_text": output_text,
                        "cer_improvement": None if input_cer is None else input_cer - output_cer,
                    })
            print(f"完成 {clip_index + 1}/6, {noise_type}", flush=True)

    output_path = RESULT_DIR / (
        "commonvoice_6speakers_no_asr.csv" if args.skip_asr
        else "commonvoice_6speakers.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(f"已保存 {len(results)} 行结果: {output_path}")


if __name__ == "__main__":
    main()
