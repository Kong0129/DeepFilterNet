"""Bounded ERB blending experiment; no training, downloads or old-result edits.

The original 20-speaker set is exploratory. Two fixed mask-guided variants
are specified before scoring: smooth(frame_gate * ERB gain), and
smooth(12-dB original ratio * ERB gain). Neither gain is a band SNR.
All candidates share one model forward pass. CSVs are computational records.
"""

import argparse
import csv
import json
import math
import random
from collections import defaultdict

import numpy as np
import torch
import torchaudio
from torch.nn import functional as F

from evaluate_framewise import (
    MANIFEST, DATASET_DIR, RESULT_DIR, prepare_manifest, recognize, write_csv,
    make_noises, mix_at_snr, selected_metadata, canonicalize,
    DATA_ROOT, MODEL_DIR, INPUT_SNRS, NOISE_TYPES, calculate_stoi,
    choose_limit, complex_mix, metrics, synthesize, framewise_mix,
)
from df.enhance import df_features, init_df, load_audio

STRATEGIES = (
    "full", "fixed_mix_12db", "q75_mix", "frame_smooth_50ms",
    "erb_frame_50ms", "erb_fixed12_50ms",
)
PROTOCOL = {
    "gamma": 1.0, "smoothing_seconds": 0.05,
    "frame_thresholds": [8, 13], "noise_types": list(NOISE_TYPES),
    "input_snrs": list(INPUT_SNRS), "clean_peak": 0.3,
    "asr": "Whisper small; zh; temperature=0; existing canonicalization",
    "selection_seed": 20261004, "heldout_speakers": 12,
    "candidates": list(STRATEGIES),
}


def smooth_time(values, hop_seconds):
    """Causal first-order smoothing of [B,1,T,bands], CPU recurrence."""
    source = values.detach().cpu()
    result = torch.empty_like(source)
    alpha = math.exp(-hop_seconds / PROTOCOL["smoothing_seconds"])
    result[:, :, 0] = source[:, :, 0]
    for t in range(1, source.shape[2]):
        result[:, :, t] = alpha * result[:, :, t - 1] + (1 - alpha) * source[:, :, t]
    return result.to(values.device)


def erb_mix(spec, enhanced, erb_mask, local_snr, inverse_fb, hop_seconds, fixed=False):
    snr = local_snr[:, :, 0]
    frame_gate = torch.where(
        snr < 8, 0.0,
        torch.where(snr < 13, 10 ** (-12 / 20), 10 ** (-6 / 20)),
    )[:, None, :, None]
    if fixed:
        frame_gate = torch.full_like(frame_gate, 10 ** (-12 / 20))
    band_ratio = smooth_time(frame_gate * erb_mask.clamp(0, 1), hop_seconds)
    frequency_ratio = (band_ratio @ inverse_fb).unsqueeze(-1)
    assert frequency_ratio.shape == spec.shape[:-1] + (1,)
    assert frequency_ratio.min() >= 0 and frequency_ratio.max() <= 0.502
    return spec * frequency_ratio + enhanced * (1 - frequency_ratio), frequency_ratio.mean().item()


def heldout_manifest():
    path = RESULT_DIR / "erb_heldout12_manifest.csv"
    old = prepare_manifest()
    excluded = {row["client_id"] for row in old + selected_metadata()}
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as handle:
            chosen = list(csv.DictReader(handle))
        assert len(chosen) == 12 and len({r["client_id"] for r in chosen}) == 12
        assert not {r["client_id"] for r in chosen} & excluded
        return chosen
    with (DATASET_DIR / "test.tsv").open(encoding="utf-8", newline="") as handle:
        candidates = sorted(csv.DictReader(handle, delimiter="\t"), key=lambda r: r["path"])
    random.Random(PROTOCOL["selection_seed"]).shuffle(candidates)
    chosen, sentences = [], {canonicalize(r["reference_text"]) for r in old}
    for row in candidates:
        reference = canonicalize(row["sentence"])
        if row["client_id"] in excluded or reference in sentences or not 8 <= len(reference) <= 60:
            continue
        audio_path = DATASET_DIR / "clips" / row["path"]
        if not audio_path.is_file():
            continue
        info = torchaudio.info(str(audio_path))
        duration = info.num_frames / info.sample_rate
        if not 2 <= duration <= 12:
            continue
        chosen.append({
            "clip_path": row["path"], "client_id": row["client_id"],
            "reference_text": row["sentence"], "duration_seconds": duration,
            "gender": row.get("gender", ""), "noise_seed": 20261004 + len(chosen),
        })
        excluded.add(row["client_id"])
        sentences.add(reference)
        if len(chosen) == 12:
            break
    if len(chosen) != 12:
        raise ValueError(f"Only {len(chosen)} eligible held-out speakers")
    write_csv(path, chosen)
    return chosen


def report(rows, path):
    groups = defaultdict(list)
    cases = defaultdict(dict)
    for row in rows:
        groups[row["strategy"]].append(row)
        cases[(row["client_id"], row["noise_type"], str(row["target_snr_db"]))][row["strategy"]] = row
    assert all(set(case) == set(STRATEGIES) for case in cases.values())
    fields = ("output_snr_db", "output_stoi", "output_cer")
    summary = {name: {
        field: float(np.mean([float(r[field]) for r in group]))
        for field in fields if group[0][field] != ""
    } for name, group in groups.items()}
    if rows[0]["output_cer"] != "":
        for name, group in groups.items():
            summary[name]["cer_median"] = float(np.median([float(r["output_cer"]) for r in group]))
            summary[name]["cer_gt_one"] = sum(float(r["output_cer"]) > 1 for r in group)
            summary[name]["cer_micro"] = sum(int(r["output_edit_errors"]) for r in group) / sum(int(r["reference_chars"]) for r in group)
    paired = {}
    for candidate in STRATEGIES[-2:]:
        paired[candidate] = {}
        for baseline in STRATEGIES[:4]:
            paired[candidate][baseline] = {}
            for field in fields:
                if field not in summary[candidate]:
                    continue
                by_speaker = defaultdict(list)
                counts = {"better": 0, "tie": 0, "worse": 0}
                for case in cases.values():
                    delta = float(case[candidate][field]) - float(case[baseline][field])
                    by_speaker[case[candidate]["client_id"]].append(delta)
                    signed = -delta if field == "output_cer" else delta
                    counts["tie" if abs(signed) < 1e-7 else "better" if signed > 0 else "worse"] += 1
                speaker_means = np.array([np.mean(d) for d in by_speaker.values()])
                rng = np.random.default_rng(20261004)
                bootstrap = rng.choice(speaker_means, (5000, len(speaker_means))).mean(axis=1)
                paired[candidate][baseline][field] = {
                    "delta": float(speaker_means.mean()),
                    "speaker_bootstrap_95pct_ci": np.quantile(bootstrap, [.025, .975]).tolist(),
                    **counts,
                }
    by_noise = {noise: {name: {
        field: float(np.mean([float(r[field]) for r in group if r["noise_type"] == noise]))
        for field in fields if field in summary[name]
    } for name, group in groups.items()} for noise in NOISE_TYPES
                if any(r["noise_type"] == noise for r in rows)}
    payload = {"conditions": len(cases), "speakers": len({r["client_id"] for r in rows}),
               "summary": summary, "by_noise": by_noise, "paired": paired}
    path.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"conditions": len(cases), "summary": summary}, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--heldout", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    chosen = heldout_manifest() if args.heldout else prepare_manifest()
    if args.prepare_only:
        print(f"Prepared {len(chosen)} speakers; no inference", flush=True)
        return
    tag = "erb_heldout12" if args.heldout else "erb_exploration20"
    tag += "_asr" if args.asr else "_quality"
    if args.limit:
        tag += f"_smoke{args.limit}"
    path = RESULT_DIR / f"{tag}.csv"
    config_path = RESULT_DIR / f"{tag}_config.json"
    config = dict(
        PROTOCOL, heldout=args.heldout, asr=args.asr, limit=args.limit,
        selection_seed=20261004 if args.heldout else 20261003,
        speakers=len(chosen),
        manifest="erb_heldout12_manifest.csv" if args.heldout else MANIFEST.name,
    )
    if config_path.exists():
        assert json.loads(config_path.read_text(encoding="utf-8")) == config
    else:
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
    rows, completed = [], set()
    if path.exists():
        if not args.resume:
            raise FileExistsError(path)
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["clip_path"], row["noise_type"], int(row["target_snr_db"]))].append(row["strategy"])
        assert all(sorted(names) == sorted(STRATEGIES) for names in grouped.values())
        completed = set(grouped)
    model, state, _ = init_df(model_base_dir=str(MODEL_DIR), log_file=None)
    model.eval()
    device = next(model.parameters()).device
    sr = state.sr()
    fb = model.mask.erb_inv_fb
    assert fb.shape == (32, 481) and torch.allclose(fb.sum(dim=0), torch.ones(481, device=device))
    # All-zero gains must produce full enhancement; all-one gains with a
    # fixed gate must reproduce the existing scalar 12-dB blend.
    toy_spec = torch.randn(1, 1, 8, 481, 2, device=device)
    toy_enhanced = torch.randn_like(toy_spec)
    toy_snr = torch.full((1, 8, 1), 15.0, device=device)
    toy_mask = torch.zeros(1, 1, 8, 32, device=device)
    zero, _ = erb_mix(toy_spec, toy_enhanced, toy_mask, toy_snr, fb, .01, fixed=True)
    one, _ = erb_mix(toy_spec, toy_enhanced, toy_mask + 1, toy_snr, fb, .01, fixed=True)
    assert torch.equal(zero, toy_enhanced)
    assert torch.allclose(one, complex_mix(toy_spec, toy_enhanced, 12), atol=1e-6)
    print("Boundary checks passed: zero gain -> full; unit gain -> fixed12", flush=True)
    asr = None
    if args.asr:
        import whisper
        model_path = DATA_ROOT / "models" / "Whisper" / "small.pt"
        if not model_path.is_file():
            raise FileNotFoundError(model_path)
        asr = whisper.load_model(str(model_path), device=str(device))
    for metadata in chosen:
        clean, _ = load_audio(str(DATASET_DIR / "clips" / metadata["clip_path"]), sr=sr)
        clean = clean.mean(dim=0, keepdim=True)
        clean *= .3 / clean.abs().max()
        noises = make_noises(clean, sr, int(metadata["noise_seed"]))
        reference = canonicalize(metadata["reference_text"])
        for noise_type in NOISE_TYPES:
            for target_snr in INPUT_SNRS:
                key = (metadata["clip_path"], noise_type, target_snr)
                if key in completed:
                    continue
                if args.limit and len(completed) >= args.limit:
                    report(rows, path)
                    return
                noisy = mix_at_snr(clean, noises[noise_type], target_snr)
                assert noisy.abs().max() <= 1 and torch.isfinite(noisy).all()
                state.reset()
                spec, ef, sf = df_features(F.pad(noisy, (0, state.fft_size())), state, model.nb_df, device=device)
                if hasattr(model, "reset_h0"):
                    model.reset_h0(batch_size=1, device=device)
                with torch.no_grad():
                    enhanced, mask, lsnr, _ = model(spec.clone(), ef, sf)
                    q75 = torch.quantile(lsnr[0, :, 0], .75).item()
                    hop_seconds = state.hop_size() / sr
                    frame, _ = framewise_mix(spec, enhanced, lsnr, hop_seconds)
                    erb_frame, ratio1 = erb_mix(spec, enhanced, mask, lsnr, fb, hop_seconds)
                    erb_fixed, ratio2 = erb_mix(spec, enhanced, mask, lsnr, fb, hop_seconds, fixed=True)
                    candidates = dict(zip(STRATEGIES, (
                        enhanced, complex_mix(spec, enhanced, 12),
                        complex_mix(spec, enhanced, choose_limit(q75)), frame, erb_frame, erb_fixed,
                    )))
                input_snr, _ = metrics(noisy, clean)
                assert abs(input_snr - target_snr) < .05
                cache = []
                for name, spectrum in candidates.items():
                    cached = next((v for s, v in cache if torch.equal(s, spectrum)), None)
                    if cached is None:
                        output = synthesize(state, spectrum, noisy.shape[-1])
                        assert output.shape == clean.shape and torch.isfinite(output).all()
                        snr, mse = metrics(output, clean)
                        recognized = recognize(asr, output, sr, reference)
                        cached = {"output_snr_db": snr, "output_mse": mse,
                                  "output_stoi": calculate_stoi(clean, output, sr),
                                  "output_cer": recognized["cer"], "output_edit_errors": recognized["edit_errors"],
                                  "output_text": recognized["asr_text"]}
                        cache.append((spectrum, cached))
                    rows.append({"clip_path": metadata["clip_path"], "client_id": metadata["client_id"],
                                 "reference_text": metadata["reference_text"], "reference_chars": len(reference),
                                 "noise_type": noise_type, "target_snr_db": target_snr, "strategy": name,
                                 "input_snr_db": input_snr, "q75_db": q75,
                                 "mean_erb_frame_ratio": ratio1, "mean_erb_fixed_ratio": ratio2, **cached})
                completed.add(key)
                write_csv(path, rows)
                print(f"Completed {len(completed)}/{len(chosen)*15}: {key}", flush=True)
    report(rows, path)


if __name__ == "__main__":
    main()
