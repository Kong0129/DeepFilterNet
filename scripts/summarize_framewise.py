"""Audit and summarize the frozen 20-speaker framewise experiment."""

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from evaluate_framewise import CONFIG, MANIFEST, RESULT_DIR, STRATEGIES


def mean(rows, field):
    return float(np.mean([float(row[field]) for row in rows]))


def summarize(rows):
    return {
        "conditions": len(rows),
        "snr_db": mean(rows, "output_snr_db"),
        "stoi": mean(rows, "output_stoi"),
        "cer_macro": mean(rows, "output_cer"),
        "cer_median": float(np.median([float(row["output_cer"]) for row in rows])),
        "cer_gt_one_conditions": sum(float(row["output_cer"]) > 1 for row in rows),
        "cer_micro": sum(int(row["output_edit_errors"]) for row in rows)
        / sum(int(row["reference_chars"]) for row in rows),
        "rms_gain_vs_reference_db": mean(rows, "output_rms_gain_vs_reference_db"),
    }


def paired(rows, baseline):
    grouped = defaultdict(dict)
    for row in rows:
        grouped[(row["clip_path"], row["noise_type"], row["target_snr_db"])][row["strategy"]] = row
    result = {}
    fields = (("snr", "output_snr_db", True), ("stoi", "output_stoi", True), ("cer", "output_cer", False))
    for name, field, higher_better in fields:
        delta_by_speaker = defaultdict(list)
        counts = {"better": 0, "tie": 0, "worse": 0}
        deltas = []
        for case in grouped.values():
            candidate = case["frame_smooth_50ms"]
            delta = float(candidate[field]) - float(case[baseline][field])
            deltas.append(delta)
            delta_by_speaker[candidate["client_id"]].append(delta)
            signed = delta if higher_better else -delta
            counts["tie" if abs(signed) <= 1e-7 else "better" if signed > 0 else "worse"] += 1
        # Resample speakers, preserving the within-speaker repeated conditions.
        speaker_means = np.array([np.mean(v) for v in delta_by_speaker.values()])
        generator = np.random.default_rng(20261003)
        boot = generator.choice(speaker_means, size=(5000, len(speaker_means)), replace=True).mean(axis=1)
        result[name] = {
            "candidate_minus_baseline": float(np.mean(deltas)),
            "speaker_bootstrap_95pct_ci": np.quantile(boot, [0.025, 0.975]).tolist(),
            **counts,
        }
    return result


def main():
    source = RESULT_DIR / "framewise_20speakers.csv"
    with source.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with MANIFEST.open(encoding="utf-8-sig", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    assert len(rows) == 1200, f"Expected 1200 rows, found {len(rows)}"
    assert len({r["client_id"] for r in rows}) == 20
    assert {r["clip_path"] for r in rows} == {r["clip_path"] for r in manifest}
    cases = defaultdict(list)
    strategies = defaultdict(list)
    for row in rows:
        cases[(row["clip_path"], row["noise_type"], row["target_snr_db"])].append(row)
        strategies[row["strategy"]].append(row)
        for field in ("output_snr_db", "output_stoi", "output_cer", "output_mse"):
            assert np.isfinite(float(row[field])), (field, row["clip_path"])
        assert abs(float(row["input_snr_db"]) - float(row["target_snr_db"])) < 0.05
        assert abs(sum(float(row[f"frame_{tag}_fraction"]) for tag in ("full", "12", "6")) - 1) < 1e-6
    assert len(cases) == 300
    for key, group in cases.items():
        assert sorted(r["strategy"] for r in group) == sorted(STRATEGIES), key
        assert len({r["input_asr_text"] for r in group}) == 1, key
        limits = {r["q75_chosen_limit_db"] for r in group}
        assert len(limits) == 1
        if next(iter(limits)) in ("none", "12"):
            baseline = "full" if next(iter(limits)) == "none" else "fixed_mix_12db"
            a = next(r for r in group if r["strategy"] == baseline)
            b = next(r for r in group if r["strategy"] == "q75_mix")
            assert a["output_asr_text"] == b["output_asr_text"]
            assert a["output_snr_db"] == b["output_snr_db"]
    summary = {s: summarize(strategies[s]) for s in STRATEGIES}
    comparisons = {s: paired(rows, s) for s in STRATEGIES if s != "frame_smooth_50ms"}
    worst = max(strategies["frame_smooth_50ms"], key=lambda row: float(row["output_cer"]))
    worst_key = (worst["clip_path"], worst["noise_type"], worst["target_snr_db"])
    sensitivity = {
        "diagnostic_case": {
            "clip_path": worst["clip_path"], "noise_type": worst["noise_type"],
            "target_snr_db": worst["target_snr_db"], "frame_cer": float(worst["output_cer"]),
            "reference_text": worst["reference_text"], "output_asr_text": worst["output_asr_text"],
        },
        "without_same_one_condition_all_strategies": {
            s: summarize([
                row for row in strategies[s]
                if (row["clip_path"], row["noise_type"], row["target_snr_db"]) != worst_key
            ]) for s in STRATEGIES
        },
        "without_same_one_speaker_all_strategies": {
            s: summarize([row for row in strategies[s] if row["client_id"] != worst["client_id"]])
            for s in STRATEGIES
        },
    }
    grouped_summary = {}
    for field in ("noise_type", "target_snr_db"):
        grouped_summary[field] = {
            value: {s: summarize([r for r in strategies[s] if r[field] == value]) for s in STRATEGIES}
            for value in sorted(
                {r[field] for r in rows},
                key=float if field == "target_snr_db" else str,
            )
        }
    clean_rows = [next(r for r in rows if r["clip_path"] == item["clip_path"]) for item in manifest]
    input_rows = [group[0] for group in cases.values()]
    data = {
        "clips": 20, "conditions": 300, "rows": 1200, "config": CONFIG,
        "gender_metadata_counts": {
            label or "unspecified": sum(item["gender"] == label for item in manifest)
            for label in sorted({item["gender"] for item in manifest})
        },
        "source_audio_total_seconds": sum(float(item["duration_seconds"]) for item in manifest),
        "clean_cer_macro": mean(clean_rows, "clean_cer"),
        "input_cer_macro": mean(input_rows, "input_cer"),
        "summary": summary, "paired_comparisons": comparisons,
        "grouped_summary": grouped_summary, "posthoc_sensitivity": sensitivity,
    }
    (RESULT_DIR / "framewise_20speakers_summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    lines = [
        "# 冻结逐帧平滑方案：20 位新增说话人评测", "",
        "从 Common Voice 中文北京子集 test.tsv 按固定随机种子选择 20 位此前未使用的说话人，",
        "排除原先六位说话人。每人一条 2–12 秒、8–60 字语音；选取不参考模型分数或听感。",
        "三类合成噪声、五档输入 SNR，共 300 个条件、四种方案、1200 行结果。", "",
        f"名单性别标注分布：{data['gender_metadata_counts']}；每人一条语音。",
        f"原始语音总时长约 {data['source_audio_total_seconds']:.1f} 秒。", "",
        "参数冻结：逐帧阈值 8/13 dB，原始频谱比例 0/0.251/0.501，一阶平滑时间 50 ms。",
        "所有方案共享同一次模型前向结果。CER 使用本地 Whisper small，并统一繁简体、数字、标点和空格。",
        "没有对输出做响度归一化；RMS 增益是全段信号能量比，包含残留噪声，不等于人声响度。", "",
        "| 方案 | 平均输出 SNR (dB) | 平均 STOI | 平均片段 CER | 汇总字符 CER | RMS 增益/参考 (dB) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for strategy, values in summary.items():
        lines.append(f"| {strategy} | {values['snr_db']:.3f} | {values['stoi']:.4f} | {values['cer_macro']:.4f} | {values['cer_micro']:.4f} | {values['rms_gain_vs_reference_db']:.3f} |")
    lines += ["", f"干净参考的平均 CER：{data['clean_cer_macro']:.4f}；带噪输入的平均 CER：{data['input_cer_macro']:.4f}。", "",
        "## CER 异常案例与敏感性分析", "",
        f"逐帧方案在 `{worst['clip_path']}` / {worst['noise_type']} / {worst['target_snr_db']} dB 的 CER 为 {float(worst['output_cer']):.4f}。",
        "Whisper 在这条输出上生成了大量重复文字。CER 包含插入错误，因此可以大于 1；原始分数保持不变。",
        "下表为事后诊断：在四种方案中同时移除同一个条件或同一位说话人，观察均值变化，不能替代完整结果。", "",
        "| 方案 | CER 中位数 | CER 大于 1 的条件数 | 同时排除该条件后的平均 CER | 同时排除该说话人后的平均 CER |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for strategy, values in summary.items():
        one = sensitivity["without_same_one_condition_all_strategies"][strategy]["cer_macro"]
        speaker = sensitivity["without_same_one_speaker_all_strategies"][strategy]["cer_macro"]
        lines.append(f"| {strategy} | {values['cer_median']:.4f} | {values['cer_gt_one_conditions']} | {one:.4f} | {speaker:.4f} |")
    lines += ["",
        "敏感性分析显示逐帧的平均 CER 明显受单个重复识别案例影响；排除后仍未胜过固定 12 dB 或 Q75。",
        "不能把这个识别异常直接解释成人声严重失真，也不能为了更好看的平均分删除原始记录。", "",
        "## 逐条件配对比较", "", "差值均为逐帧平滑减去基线；SNR/STOI 越高越好，CER 越低越好。",
        "95% 区间以说话人为单位重采样 5000 次，避免把同一说话人的 15 个噪声条件当作独立样本。", "",
        "| 基线 | 指标 | 平均差值 | 说话人 bootstrap 95% 区间 | 更好/相同/更差条件数 |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for baseline, metrics in comparisons.items():
        for metric, values in metrics.items():
            low, high = values["speaker_bootstrap_95pct_ci"]
            lines.append(f"| {baseline} | {metric} | {values['candidate_minus_baseline']:.4f} | [{low:.4f}, {high:.4f}] | {values['better']}/{values['tie']}/{values['worse']} |")
    for field, groups in grouped_summary.items():
        lines += ["", f"## 按 {field} 分组", "", "| 分组 | 方案 | SNR (dB) | STOI | 平均 CER |", "| --- | --- | ---: | ---: | ---: |"]
        for value, group in groups.items():
            for strategy, metrics in group.items():
                lines.append(f"| {value} | {strategy} | {metrics['snr_db']:.3f} | {metrics['stoi']:.4f} | {metrics['cer_macro']:.4f} |")
    lines += ["", "## 解释边界", "",
        "20 位新增说话人比原六位更有代表性，但每人仍只有一条短语音，且噪声为合成噪声。",
        "本次评价只支持对冻结规则的这组比较；若看过这些结果后再调参数，此集合就成为开发数据。",
        "CER 反映指定 ASR 系统的识别表现；不能直接证明人声吞字的内部原因，也不能替代人工听感。",
        "原六位和自录 9 号的试听是开发/诊断证据，不并入本次新增说话人均值。", "",
        "复现：`python scripts/evaluate_framewise.py --resume`，随后 `python scripts/summarize_framewise.py`。", "",
    ]
    report = RESULT_DIR / "framewise_20speakers_review.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary": summary, "paired": comparisons}, ensure_ascii=False, indent=2))
    print(f"Report: {report}")


if __name__ == "__main__":
    main()
