"""Compare enhancement strategies on the same DeepFilterNet3 forward pass.

Reads audio and model weights from D:, and writes a metrics CSV next to this
repository. Optional listening samples are written to D:. Not a latency test.
"""

import csv
import argparse
import random
import sys
from pathlib import Path

# pystoi is installed here because the Conda environment is not writable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "_local_packages"))

import torch
from torch.nn import functional as F
from pystoi import stoi

from df.enhance import df_features, init_df, load_audio, save_audio
from df.utils import as_complex
import whisper
import torchaudio.functional as AF
from evaluate_cer import normalize, edit_distance


DATA_ROOT = Path(r"D:\github\DeepFilterNetAudio")
AUDIO_DIR = DATA_ROOT / "own_voice_evaluation"
MODEL_DIR = DATA_ROOT / "models" / "DeepFilterNet3"
RESULT_PATH = Path(__file__).resolve().parents[1] / "results" / "multinoise_comparison.csv"
NOISE_TYPES = ("white", "low_frequency", "tonal")
INPUT_SNRS = (-5, 0, 5, 10, 15)
LISTENING_CASES = {("white", 15), ("low_frequency", 15), ("tonal", 15)}


def choose_limit(q75):
    """Exploratory thresholds previously chosen using one voice/white noise."""
    if q75 < 8:
        return None
    if q75 < 13:
        return 12
    return 6


def complex_mix(original, enhanced, limit_db):
    """Same post-model mixing rule used by the official enhance() function."""
    if limit_db is None:
        return enhanced
    original_ratio = 10 ** (-abs(limit_db) / 20)
    return original * original_ratio + enhanced * (1 - original_ratio)


def magnitude_clamp(original, enhanced, limit_db):
    """Existing experimental clamp; keep its behavior for a fair comparison."""
    if limit_db is None:
        return enhanced
    floor_ratio = 10 ** (-abs(limit_db) / 20)
    original_magnitude = torch.linalg.vector_norm(original, dim=-1, keepdim=True)
    enhanced_magnitude = torch.linalg.vector_norm(enhanced, dim=-1, keepdim=True)
    scale = torch.clamp(
        original_magnitude * floor_ratio / (enhanced_magnitude + 1e-8), min=1.0
    )
    return enhanced * scale


def synthesize(df_state, spectrum, original_length):
    # Each candidate must start with a fresh overlap-add buffer.
    df_state.reset()
    complex_spectrum = as_complex(spectrum.squeeze(1).cpu()).numpy()
    audio = torch.as_tensor(df_state.synthesis(complex_spectrum))
    delay = df_state.fft_size() - df_state.hop_size()
    return audio[:, delay : original_length + delay]


def metrics(output, clean):
    error_power = torch.mean((output - clean) ** 2).item()
    signal_power = torch.mean(clean**2).item()
    snr = 10 * torch.log10(torch.tensor(signal_power / (error_power + 1e-8))).item()
    return snr, error_power


def calculate_stoi(clean, estimate, sample_rate):
    """Compare two aligned mono waveforms; higher means more intelligible."""
    if clean.shape != estimate.shape:
        raise ValueError(f"STOI input shapes differ: {clean.shape} vs {estimate.shape}")
    clean_1d = clean.squeeze(0).detach().cpu().numpy()
    estimate_1d = estimate.squeeze(0).detach().cpu().numpy()
    return float(stoi(clean_1d, estimate_1d, sample_rate, extended=False))

def calculate_cer(asr_model, waveform, sample_rate, reference):
    # waveform: [1, 采样点数]，目前为 48 kHz
    waveform_16k = AF.resample(waveform.cpu(), sample_rate, 16000)
    audio_1d = waveform_16k.squeeze(0).numpy()

    result = asr_model.transcribe(
        audio_1d,
        language="zh",
        task="transcribe",
        fp16=True,
        verbose=False,
        temperature=0.0,
    )

    prediction = normalize(result["text"])
    errors = edit_distance(reference, prediction)
    return errors / len(reference), result["text"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    dataset = parser.add_mutually_exclusive_group()
    dataset.add_argument(
        "--new-recording", action="store_true",
        help="Use new clean.wav and the 15 new noisy_*.wav files for independent validation.",
    )
    dataset.add_argument(
        "--huidage", action="store_true",
        help="Use the 15 noisy files and clean recording inside the huidage directory.",
    )
    parser.add_argument(
        "--export-listening", action="store_true",
        help="Export three selected cases for blind listening (requires --new-recording or --huidage).",
    )
    args = parser.parse_args()
    if args.export_listening and not (args.new_recording or args.huidage):
        parser.error("--export-listening requires --new-recording or --huidage")
    if args.huidage:
        audio_dir = AUDIO_DIR / "huidage"
        clean_path = audio_dir / "辉大哥，蒋介石真的可以.wav"
        result_path = RESULT_PATH.with_name("multinoise_huidage.csv")
    elif args.new_recording:
        audio_dir = AUDIO_DIR
        clean_path = audio_dir / "new clean.wav"
        result_path = RESULT_PATH.with_name("multinoise_huidage_with_cer.csv")
    else:
        audio_dir = AUDIO_DIR
        clean_path = audio_dir / "clean_reference.wav"
        result_path = RESULT_PATH
    noisy_prefix = "new noisy_" if (args.new_recording or args.huidage) else "noisy_"
    listening_dir = audio_dir / "listening_comparison"
    listening_key = []
    shuffle = random.Random(2026)

    for path in (clean_path, MODEL_DIR / "config.ini"):
        if not path.is_file():
            raise FileNotFoundError(path)

    model, df_state, _ = init_df(model_base_dir=str(MODEL_DIR), log_file=None)
    model.eval()
    device = next(model.parameters()).device
    sample_rate = df_state.sr()
    clean, _ = load_audio(str(clean_path), sr=sample_rate)
    if args.new_recording or args.huidage:
        clean = clean.mean(dim=0, keepdim=True)
    if args.huidage:
        reference = normalize("辉大哥，蒋介石真的可以")
        asr_model = whisper.load_model(
            "small",
            device="cuda",
            download_root=str(DATA_ROOT / "models" / "Whisper"),
        )
        clean_cer, clean_text = calculate_cer(
            asr_model, clean, sample_rate, reference
        )
        print("干净录音 CER:", clean_cer, "识别文字:", clean_text)
    rows = []
    for noise_type in NOISE_TYPES:
        for target_snr in INPUT_SNRS:
            tag = f"m{abs(target_snr)}" if target_snr < 0 else str(target_snr)
            noisy_path = audio_dir / f"{noisy_prefix}{noise_type}_snr_{tag}db.wav"
            if not noisy_path.is_file():
                raise FileNotFoundError(noisy_path)
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
                    "adaptive_clamp": magnitude_clamp(
                        spec, enhanced_spec, chosen_limit
                    ),
                }

            input_snr, _ = metrics(noisy, clean)
            input_stoi = calculate_stoi(clean, noisy, sample_rate)
            if args.huidage:
                input_cer, input_text = calculate_cer(
                    asr_model, noisy, sample_rate, reference
                )
            listening_outputs = {}
            for strategy, candidate in candidates.items():
                output = synthesize(df_state, candidate, noisy.shape[-1])
                if output.shape != clean.shape:
                    raise ValueError(f"Unexpected output shape: {strategy}, {noisy_path}")
                output_snr, output_mse = metrics(output, clean)
                output_stoi = calculate_stoi(clean, output, sample_rate)
                if args.huidage:
                    output_cer, output_text = calculate_cer(
                        asr_model,
                        output,
                        sample_rate,
                        reference,
                    )
                if (args.export_listening and (noise_type, target_snr) in LISTENING_CASES
                        and strategy in ("full", "fixed_mix_12db", "fixed_mix_6db")):
                    listening_outputs[strategy] = output
                rows.append(
                    {
                        "noise_type": noise_type,
                        "target_snr_db": target_snr,
                        "input_snr_db": input_snr,
                        "input_stoi": input_stoi,
                        "input_cer": input_cer if args.huidage else None,
                        "q75_db": q75,
                        "chosen_limit_db": "none" if chosen_limit is None else chosen_limit,
                        "strategy": strategy,
                        "output_snr_db": output_snr,
                        "output_stoi": output_stoi,
                        "output_cer": output_cer if args.huidage else None,
                        "stoi_gain": output_stoi - input_stoi,
                        "snr_gain_db": output_snr - input_snr,
                        "output_mse": output_mse,
                        "input_asr_text": input_text if args.huidage else None,
                        "output_asr_text": output_text if args.huidage else None,
                        "cer_improvement": (input_cer - output_cer) if args.huidage else None,

                    }
                )
            if listening_outputs:
                case_dir = listening_dir / f"{noise_type}_{tag}db"
                case_dir.mkdir(parents=True, exist_ok=True)
                # Save float WAVs without per-file normalization: loudness differences stay audible.
                save_audio(str(case_dir / "input.wav"), noisy, sample_rate, dtype=torch.float32)
                names = list(listening_outputs)
                shuffle.shuffle(names)
                for letter, strategy in zip("ABC", names):
                    save_audio(str(case_dir / f"{letter}.wav"), listening_outputs[strategy],
                               sample_rate, dtype=torch.float32)
                    listening_key.append({"case": case_dir.name, "file": f"{letter}.wav",
                                          "strategy": strategy})
            print(f"完成 {noise_type}, 目标SNR {target_snr:+} dB, Q75 {q75:.2f} dB")

    result_path.parent.mkdir(parents=True, exist_ok=True)
    with result_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"已保存 {len(rows)} 行结果：{result_path}")
    if listening_key:
        key_path = result_path.with_name(
            "listening_key_huidage.csv" if args.huidage else "listening_key_new_recording.csv"
        )
        with key_path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=("case", "file", "strategy"))
            writer.writeheader()
            writer.writerows(listening_key)
        print(f"盲听文件：{listening_dir}；答案表（试听后再看）：{key_path}")
    print("说明：每个数据组选用一条干净语音；Q75阈值仍是先前实验选定的，结果属于探索性评测。")


if __name__ == "__main__":
    main()
