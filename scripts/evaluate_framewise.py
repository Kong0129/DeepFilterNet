"""Frozen four-strategy comparison on 20 additional Common Voice speakers.

Selection uses transcript length, duration and a fixed seed, before ASR/enhancement.
The six previously inspected speakers are excluded. Audio stays in memory.
Results and selection manifest are separate from all earlier experiments.
"""

import argparse
import csv
import json
import math
import random
from pathlib import Path

import torch
import torchaudio
from torch.nn import functional as F

from compare_commonvoice_speakers import (
    DATASET_DIR, RESULT_DIR, make_noises, mix_at_snr, selected_metadata,
)
from compare_multinoise_strategies import (
    DATA_ROOT, MODEL_DIR, INPUT_SNRS, NOISE_TYPES, calculate_cer,
    calculate_stoi, choose_limit, complex_mix, metrics, synthesize,
)
from df.enhance import df_features, init_df, load_audio
from evaluate_cer import edit_distance
from rescore_commonvoice_cer import canonicalize


MANIFEST = RESULT_DIR / "framewise_20speakers_manifest.csv"
CONFIG = {
    "selection_seed": 20261003,
    "speakers": 20,
    "transcript_chars": [8, 60],
    "duration_seconds": [2.0, 12.0],
    "frame_snr_thresholds_db": [8.0, 13.0],
    "original_mix_ratios": [0.0, 10 ** (-12 / 20), 10 ** (-6 / 20)],
    "smoothing_seconds": 0.05,
    "input_snrs_db": list(INPUT_SNRS),
    "noise_types": list(NOISE_TYPES),
    "noise_seed_base": 20261003,
    "clean_peak": 0.3,
    "asr_model": "small",
    "cer_normalization": "OpenCC t2s; Arabic integers to Chinese numerals; no punctuation/spaces",
}
STRATEGIES = ("full", "fixed_mix_12db", "q75_mix", "frame_smooth_50ms")


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def prepare_manifest():
    excluded = {row["client_id"] for row in selected_metadata()}
    if MANIFEST.exists():
        with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
            chosen = list(csv.DictReader(handle))
        if len(chosen) != CONFIG["speakers"]:
            raise ValueError("Manifest must contain exactly 20 clips")
        ids = {row["client_id"] for row in chosen}
        if len(ids) != len(chosen) or ids & excluded:
            raise ValueError("Duplicate or previously inspected speakers in manifest")
        return chosen
    with (DATASET_DIR / "test.tsv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    candidates = sorted(
        (row for row in rows if row["client_id"] not in excluded),
        key=lambda row: row["path"],
    )
    random.Random(CONFIG["selection_seed"]).shuffle(candidates)
    chosen, speaker_ids, sentences = [], set(), set()
    for row in candidates:
        sentence = canonicalize(row["sentence"])
        if row["client_id"] in speaker_ids or sentence in sentences:
            continue
        if not 8 <= len(sentence) <= 60:
            continue
        path = DATASET_DIR / "clips" / row["path"]
        if not path.is_file():
            continue
        info = torchaudio.info(str(path))
        duration = info.num_frames / info.sample_rate
        if not 2.0 <= duration <= 12.0:
            continue
        chosen.append({
            "clip_path": row["path"], "client_id": row["client_id"],
            "reference_text": row["sentence"], "duration_seconds": duration,
            "gender": row.get("gender", ""), "age": row.get("age", ""),
            "noise_seed": CONFIG["noise_seed_base"] + len(chosen),
        })
        speaker_ids.add(row["client_id"])
        sentences.add(sentence)
        if len(chosen) == CONFIG["speakers"]:
            break
    if len(chosen) != CONFIG["speakers"]:
        raise ValueError(f"Only {len(chosen)} eligible new speakers found")
    write_csv(MANIFEST, chosen)
    return chosen


def framewise_mix(spec, enhanced, local_snr, hop_seconds):
    snr = local_snr[:, :, 0]
    target = torch.where(
        snr < 8, 0.0,
        torch.where(snr < 13, 10 ** (-12 / 20), 10 ** (-6 / 20)),
    )
    # The same recurrence as the learning script. CPU avoids thousands of
    # tiny GPU launches; coefficients return to the spectrum device afterward.
    target_cpu = target.detach().cpu()
    smooth = torch.empty_like(target_cpu)
    alpha = math.exp(-hop_seconds / CONFIG["smoothing_seconds"])
    smooth[:, 0] = target_cpu[:, 0]
    for t in range(1, target_cpu.shape[1]):
        smooth[:, t] = alpha * smooth[:, t - 1] + (1 - alpha) * target_cpu[:, t]
    ratio = smooth.to(spec.device)[:, None, :, None, None]
    diagnostic = {
        "frame_full_fraction": (snr < 8).float().mean().item(),
        "frame_12_fraction": ((snr >= 8) & (snr < 13)).float().mean().item(),
        "frame_6_fraction": (snr >= 13).float().mean().item(),
        "frame_mean_original_ratio": smooth.mean().item(),
        "raw_max_ratio_jump": (target_cpu[:, 1:] - target_cpu[:, :-1]).abs().max().item(),
        "smooth_max_ratio_jump": (smooth[:, 1:] - smooth[:, :-1]).abs().max().item(),
    }
    return spec * ratio + enhanced * (1 - ratio), diagnostic


def recognize(asr_model, audio, sample_rate, reference):
    if asr_model is None:
        return {"cer": "", "edit_errors": "", "asr_text": ""}
    _, text = calculate_cer(asr_model, audio, sample_rate, reference)
    errors = edit_distance(reference, canonicalize(text))
    return {"cer": errors / len(reference), "edit_errors": errors, "asr_text": text}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=0, help="Smoke check only; separate output")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    chosen = prepare_manifest()
    print(f"Selected {len(chosen)} new speakers; manifest: {MANIFEST}", flush=True)
    if args.prepare_only:
        return
    name = "framewise_20speakers"
    if args.skip_asr:
        name += "_no_asr"
    if args.limit_cases:
        name += f"_smoke{args.limit_cases}"
    result_path = RESULT_DIR / f"{name}.csv"
    config_path = RESULT_DIR / f"{name}_config.json"
    run_config = dict(CONFIG, skip_asr=args.skip_asr, limit_cases=args.limit_cases)
    if config_path.exists() and json.loads(config_path.read_text(encoding="utf-8")) != run_config:
        raise ValueError("Existing result uses different parameters")
    rows, completed = [], set()
    if result_path.exists():
        if not args.resume:
            raise FileExistsError(f"Use --resume to continue: {result_path}")
        with result_path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        grouped = {}
        for row in rows:
            key = (row["clip_path"], row["noise_type"], int(row["target_snr_db"]))
            grouped.setdefault(key, []).append(row["strategy"])
        for key, strategies in grouped.items():
            if sorted(strategies) != sorted(STRATEGIES):
                raise ValueError(f"Incomplete or duplicate case: {key}")
            completed.add(key)
    config_path.write_text(json.dumps(run_config, ensure_ascii=False, indent=2), encoding="utf-8")
    model, df_state, _ = init_df(model_base_dir=str(MODEL_DIR), log_file=None)
    model.eval()
    device = next(model.parameters()).device
    sample_rate = df_state.sr()
    asr_model = None
    if not args.skip_asr:
        import whisper
        if not (DATA_ROOT / "models" / "Whisper" / "small.pt").is_file():
            raise FileNotFoundError("Local Whisper small.pt is required")
        asr_model = whisper.load_model(
            "small", device=str(device), download_root=str(DATA_ROOT / "models" / "Whisper"),
        )
    case_number = 0
    for metadata in chosen:
        all_keys = [(metadata["clip_path"], noise, snr) for noise in NOISE_TYPES for snr in INPUT_SNRS]
        if all(key in completed for key in all_keys):
            continue
        clean, _ = load_audio(str(DATASET_DIR / "clips" / metadata["clip_path"]), sr=sample_rate)
        clean = clean.mean(dim=0, keepdim=True)
        peak = clean.abs().max().item()
        if peak == 0:
            raise ValueError(f"Silent clip: {metadata['clip_path']}")
        clean = clean * (CONFIG["clean_peak"] / peak)
        reference = canonicalize(metadata["reference_text"])
        clean_asr = recognize(asr_model, clean, sample_rate, reference)
        raw_noises = make_noises(clean, sample_rate, int(metadata["noise_seed"]))
        for noise_type in NOISE_TYPES:
            for target_snr in INPUT_SNRS:
                key = (metadata["clip_path"], noise_type, target_snr)
                case_number += 1
                if args.limit_cases and case_number > args.limit_cases:
                    print(f"Saved {len(rows)} rows: {result_path}", flush=True)
                    return
                if key in completed:
                    continue
                noisy = mix_at_snr(clean, raw_noises[noise_type], target_snr)
                if noisy.abs().max().item() > 1:
                    raise ValueError(f"Input clipping: {key}")
                df_state.reset()
                padded = F.pad(noisy, (0, df_state.fft_size()))
                spec, erb_feature, spec_feature = df_features(padded, df_state, model.nb_df, device=device)
                if hasattr(model, "reset_h0"):
                    model.reset_h0(batch_size=1, device=device)
                with torch.no_grad():
                    enhanced, _, local_snr, _ = model(spec.clone(), erb_feature, spec_feature)
                    q75 = torch.quantile(local_snr[0, :, 0], 0.75).item()
                    chosen_limit = choose_limit(q75)
                    smooth_spec, diagnostics = framewise_mix(
                        spec, enhanced, local_snr, df_state.hop_size() / sample_rate,
                    )
                    candidates = {
                        "full": enhanced,
                        "fixed_mix_12db": complex_mix(spec, enhanced, 12),
                        "q75_mix": complex_mix(spec, enhanced, chosen_limit),
                        "frame_smooth_50ms": smooth_spec,
                    }
                input_snr, _ = metrics(noisy, clean)
                input_stoi = calculate_stoi(clean, noisy, sample_rate)
                input_asr = recognize(asr_model, noisy, sample_rate, reference)
                cache = []
                for strategy, candidate in candidates.items():
                    cached = next((value for spectrum, value in cache if torch.equal(spectrum, candidate)), None)
                    if cached is None:
                        output = synthesize(df_state, candidate, noisy.shape[-1])
                        if output.shape != clean.shape or not torch.isfinite(output).all():
                            raise ValueError(f"Invalid output: {key}/{strategy}")
                        output_snr, output_mse = metrics(output, clean)
                        gain = 10 * torch.log10((output.square().mean() + 1e-12) / (clean.square().mean() + 1e-12)).item()
                        cached = {
                            "output_snr_db": output_snr, "output_mse": output_mse,
                            "output_stoi": calculate_stoi(clean, output, sample_rate),
                            "output_rms_gain_vs_reference_db": gain,
                            **recognize(asr_model, output, sample_rate, reference),
                        }
                        cache.append((candidate, cached))
                    rows.append({
                        "clip_path": metadata["clip_path"], "client_id": metadata["client_id"],
                        "reference_text": metadata["reference_text"], "reference_chars": len(reference),
                        "noise_type": noise_type, "target_snr_db": target_snr, "strategy": strategy,
                        "input_snr_db": input_snr, "input_stoi": input_stoi,
                        "input_cer": input_asr["cer"], "input_edit_errors": input_asr["edit_errors"],
                        "clean_cer": clean_asr["cer"], "clean_edit_errors": clean_asr["edit_errors"],
                        "q75_db": q75, "q75_chosen_limit_db": "none" if chosen_limit is None else chosen_limit,
                        "output_snr_db": cached["output_snr_db"], "output_mse": cached["output_mse"],
                        "output_stoi": cached["output_stoi"], "output_cer": cached["cer"],
                        "output_edit_errors": cached["edit_errors"],
                        "snr_gain_db": cached["output_snr_db"] - input_snr,
                        "stoi_gain": cached["output_stoi"] - input_stoi,
                        "output_rms_gain_vs_reference_db": cached["output_rms_gain_vs_reference_db"],
                        "clean_asr_text": clean_asr["asr_text"], "input_asr_text": input_asr["asr_text"],
                        "output_asr_text": cached["asr_text"], **diagnostics,
                    })
                write_csv(result_path, rows)
                print(f"Completed {len(rows) // 4}/300: {metadata['clip_path']}/{noise_type}/{target_snr:+}dB", flush=True)
    print(f"Saved {len(rows)} rows: {result_path}", flush=True)


if __name__ == "__main__":
    main()
