"""Export three cross-speaker diagnostic cases for listening on D:."""

import csv
from pathlib import Path

import torch
from torch.nn import functional as F

from compare_commonvoice_speakers import (
    DATASET_DIR, SELECTED_PATHS, make_noises, mix_at_snr, selected_metadata,
)
from compare_multinoise_strategies import (
    DATA_ROOT, MODEL_DIR, complex_mix, metrics, synthesize,
)
from df.enhance import df_features, init_df, load_audio, save_audio


CASES = (
    ("common_voice_zh-CN_18757799.mp3", "white", -5),
    ("common_voice_zh-CN_18977976.mp3", "low_frequency", -5),
    ("common_voice_zh-CN_22052863.mp3", "white", 5),
)
OUTPUT_DIR = DATA_ROOT / "own_voice_evaluation" / "commonvoice_listening"
RESULT_CSV = Path(__file__).resolve().parents[1] / "results" / "commonvoice_6speakers_cer_normalized.csv"


def main():
    metadata = {row["path"]: row for row in selected_metadata()}
    with RESULT_CSV.open(newline="", encoding="utf-8-sig") as handle:
        results = list(csv.DictReader(handle))
    model, df_state, _ = init_df(model_base_dir=str(MODEL_DIR), log_file=None)
    model.eval()
    device = next(model.parameters()).device
    sample_rate = df_state.sr()

    for name, noise_type, target_snr in CASES:
        if name not in metadata:
            raise ValueError(f"Not a selected test clip: {name}")
        source, _ = load_audio(str(DATASET_DIR / "clips" / name), sr=sample_rate)
        clean = source.mean(dim=0, keepdim=True)
        clean = clean * (0.3 / clean.abs().max().item())
        index = SELECTED_PATHS.index(name)
        noise = make_noises(clean, sample_rate, seed=2026 + index)[noise_type]
        noisy = mix_at_snr(clean, noise, target_snr)

        df_state.reset()
        spec, erb_feature, spec_feature = df_features(
            F.pad(noisy, (0, df_state.fft_size())),
            df_state, model.nb_df, device=device,
        )
        if hasattr(model, "reset_h0"):
            model.reset_h0(batch_size=noisy.shape[0], device=device)
        with torch.no_grad():
            enhanced_spec, _, _, _ = model(
                spec.clone(), erb_feature, spec_feature
            )
            candidates = {
                "full": enhanced_spec,
                "fixed_mix_12db": complex_mix(spec, enhanced_spec, 12),
                "fixed_mix_6db": complex_mix(spec, enhanced_spec, 6),
            }

        case_dir = OUTPUT_DIR / f"{Path(name).stem}_{noise_type}_{target_snr:+}db"
        case_dir.mkdir(parents=True, exist_ok=True)
        save_audio(str(case_dir / "clean.wav"), clean, sample_rate, dtype=torch.float32)
        save_audio(str(case_dir / "noisy.wav"), noisy, sample_rate, dtype=torch.float32)
        for strategy, candidate in candidates.items():
            output = synthesize(df_state, candidate, noisy.shape[-1])
            measured_snr, _ = metrics(output, clean)
            matching = [
                row for row in results
                if row["clip_path"] == name
                and row["noise_type"] == noise_type
                and int(row["target_snr_db"]) == target_snr
                and row["strategy"] == strategy
            ]
            if len(matching) != 1 or abs(measured_snr - float(matching[0]["output_snr_db"])) > 0.01:
                raise ValueError(f"Reproduced result does not match CSV: {case_dir}, {strategy}")
            save_audio(str(case_dir / f"{strategy}.wav"), output, sample_rate, dtype=torch.float32)
        print(f"已导出: {case_dir}")


if __name__ == "__main__":
    main()
