"""Evaluate fixed enhancement strategies on three held-out Chinese recordings.

Only reads audio/model files from D:. Writes a separate CSV in this repository.
The Q75 thresholds are intentionally unchanged from the earlier experiment.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from compare_multinoise_strategies import (
    AUDIO_DIR, DATA_ROOT, INPUT_SNRS, MODEL_DIR, NOISE_TYPES,
    calculate_cer, calculate_stoi, choose_limit, complex_mix,
    magnitude_clamp, metrics, synthesize,
)
from df.enhance import df_features, init_df, load_audio
from evaluate_cer import normalize


RECORDINGS = {
    "7": "再见，今天和你玩的很开心，下次我们什么时候一起出来继续玩呢",
    "8": "你好，很高兴见到你，今天我们一起做一些什么事情呢？对了，我们一起去逛街吧",
    "9": "我做了一个梦，梦见我一会儿骑着摩托车穿越了整片田野，又梦见我一会儿坐着飞机飞越了整个学校",
}
RECORDING_DIR = AUDIO_DIR / "新录音"
RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "multisignal_heldout_789.csv"


def audio_cases():
    """Yield (recording id, reference path, noisy path, noise type, target SNR)."""
    for recording_id in RECORDINGS:
        clean_path = RECORDING_DIR / f"{recording_id}.wav"
        for noise_type in NOISE_TYPES:
            for target_snr in INPUT_SNRS:
                tag = f"m{abs(target_snr)}" if target_snr < 0 else str(target_snr)
                noisy_path = (
                    RECORDING_DIR / recording_id
                    / f"new noisy_{noise_type}_snr_{tag}db.wav"
                )
                yield recording_id, clean_path, noisy_path, noise_type, target_snr


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--skip-asr", action="store_true",
        help="Run SNR/STOI first without Whisper; CER columns remain empty.",
    )
    args = parser.parse_args()

    cases = list(audio_cases())
    missing = [str(path) for case in cases for path in case[1:3] if not path.is_file()]
    if not (MODEL_DIR / "config.ini").is_file():
        missing.append(str(MODEL_DIR / "config.ini"))
    if missing:
        raise FileNotFoundError("Missing required files:\n" + "\n".join(sorted(set(missing))))

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

    rows = []
    for recording_id in RECORDINGS:
        clean_path = RECORDING_DIR / f"{recording_id}.wav"
        clean, _ = load_audio(str(clean_path), sr=sample_rate)
        clean = clean.mean(dim=0, keepdim=True)
        reference = normalize(RECORDINGS[recording_id])
        clean_cer = None
        clean_text = None
        if asr_model is not None:
            clean_cer, clean_text = calculate_cer(asr_model, clean, sample_rate, reference)
        print(f"录音 {recording_id}: {clean.shape[-1] / sample_rate:.2f} 秒; 干净CER={clean_cer}", flush=True)

        for case in (item for item in cases if item[0] == recording_id):
            _, _, noisy_path, noise_type, target_snr = case
            noisy, _ = load_audio(str(noisy_path), sr=sample_rate)
            if noisy.shape != clean.shape:
                raise ValueError(f"Audio/reference shape mismatch: {noisy_path}")

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
                    "adaptive_clamp": magnitude_clamp(spec, enhanced_spec, chosen_limit),
                }

            input_snr, _ = metrics(noisy, clean)
            input_stoi = calculate_stoi(clean, noisy, sample_rate)
            input_cer = None
            input_text = None
            if asr_model is not None:
                input_cer, input_text = calculate_cer(asr_model, noisy, sample_rate, reference)

            for strategy, candidate in candidates.items():
                output = synthesize(df_state, candidate, noisy.shape[-1])
                if output.shape != clean.shape:
                    raise ValueError(f"Unexpected output shape: {strategy}, {noisy_path}")
                output_snr, output_mse = metrics(output, clean)
                output_stoi = calculate_stoi(clean, output, sample_rate)
                output_cer = None
                output_text = None
                if asr_model is not None:
                    output_cer, output_text = calculate_cer(
                        asr_model, output, sample_rate, reference
                    )
                rows.append({
                    "recording_id": recording_id,
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
                    "input_asr_text": input_text,
                    "output_asr_text": output_text,
                    "clean_asr_text": clean_text,
                    "cer_improvement": None if input_cer is None else input_cer - output_cer,
                })
            print(f"完成 {recording_id}/{noise_type}/{target_snr:+} dB; Q75={q75:.2f} dB", flush=True)

    result_path = RESULT_PATH.with_name(
        "multisignal_heldout_789_no_asr.csv" if args.skip_asr else RESULT_PATH.name
    )
    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存 {len(rows)} 行结果: {result_path}")


if __name__ == "__main__":
    main()
