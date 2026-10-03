"""Audit both stages and write a bounded, reproducible research report."""

import csv
import json
from collections import defaultdict

import numpy as np

from research_erb_blending import RESULT_DIR, STRATEGIES, report, selected_metadata


def read_rows(name):
    with (RESULT_DIR / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def audit(rows, speakers, asr):
    assert len(rows) == speakers * 15 * 6
    assert len({r["client_id"] for r in rows}) == speakers
    cases = defaultdict(list)
    for row in rows:
        cases[(row["clip_path"], row["noise_type"], row["target_snr_db"])].append(row)
        assert abs(float(row["input_snr_db"]) - float(row["target_snr_db"])) < .05
        for field in ("output_snr_db", "output_mse", "output_stoi"):
            assert np.isfinite(float(row[field]))
        assert 0 <= float(row["mean_erb_frame_ratio"]) <= .502
        assert 0 <= float(row["mean_erb_fixed_ratio"]) <= .252
        if asr:
            assert np.isfinite(float(row["output_cer"]))
            assert abs(float(row["output_cer"]) - int(row["output_edit_errors"]) / int(row["reference_chars"])) < 1e-10
        else:
            assert row["output_cer"] == ""
    for group in cases.values():
        assert sorted(r["strategy"] for r in group) == sorted(STRATEGIES)
        assert len({r["q75_db"] for r in group}) == 1
        assert len({r["reference_text"] for r in group}) == 1
    assert len(cases) == speakers * 15


def table(summary, asr):
    lines = ["| 方案 | 平均输出 SNR / dB | 平均 STOI |" + (" 平均 CER | CER 中位数 |" if asr else ""),
             "| --- | ---: | ---: |" + (" ---: | ---: |" if asr else "")]
    for strategy in STRATEGIES:
        values = summary[strategy]
        text = f"| {strategy} | {values['output_snr_db']:.4f} | {values['output_stoi']:.6f} |"
        if asr:
            text += f" {values['output_cer']:.4%} | {values['cer_median']:.4%} |"
        lines.append(text)
    return lines


def main():
    exploratory = read_rows("erb_exploration20_quality.csv")
    heldout = read_rows("erb_heldout12_asr.csv")
    audit(exploratory, 20, False)
    audit(heldout, 12, True)
    old_ids = {row["client_id"] for row in selected_metadata()}
    exploration_ids = {r["client_id"] for r in exploratory}
    heldout_ids = {r["client_id"] for r in heldout}
    assert not exploration_ids & heldout_ids and not old_ids & heldout_ids
    manifest = read_rows("erb_heldout12_manifest.csv")
    assert {r["client_id"] for r in manifest} == heldout_ids
    previous = read_rows("framewise_20speakers.csv")
    previous_by_key = {(r["clip_path"], r["noise_type"], r["target_snr_db"], r["strategy"]): r for r in previous}
    parity = {}
    for field, tolerance in (("output_snr_db", .01), ("output_stoi", 1e-4)):
        parity[field] = max(
            abs(float(r[field]) - float(previous_by_key[(r["clip_path"], r["noise_type"], r["target_snr_db"], r["strategy"])][field]))
            for r in exploratory if r["strategy"] in STRATEGIES[:4]
        )
        assert parity[field] < tolerance
    report(exploratory, RESULT_DIR / "erb_exploration20_quality.csv")
    report(heldout, RESULT_DIR / "erb_heldout12_asr.csv")
    exploration = json.loads((RESULT_DIR / "erb_exploration20_quality.json").read_text(encoding="utf-8"))
    validation = json.loads((RESULT_DIR / "erb_heldout12_asr.json").read_text(encoding="utf-8"))
    lines = [
        "# ERB 掩码引导的子带混合研究（2026-10-03）", "",
        "## 本轮结论", "",
        "保留固定比例 ERB 方案作为小规模候选，不继续扩展逐帧 SNR × ERB 方案。固定比例 ERB 在两批数据上相对 full 均有小幅 SNR/STOI 均值改善，新 12 人验证的描述性差值区间也支持这一方向。它提供的是折中：相对普通固定 12 dB 保住更多 SNR，但 CER 更差，不能声称全面优于现有方案。",
        "",
        "新验证中 full 的平均 CER 明显受一个 CER=20 的重复识别条件影响。所有策略同时排除该条件后，full 的 CER 为约 32.47%，固定 ERB 约 28.00%，普通固定 12 dB 约 25.80%。完整结果仍为主结果，敏感性分析不代替独立验证。尚未证明说话起始的沙沙声或主观闷声已经解决。",
        "",
        "## 目的与方案", "",
        "检验子带约束能否改善统一整帧混合的噪声回混问题。本轮不训练、不微调、不下载模型，也不修改已有评测脚本和旧结果。",
        "",
        "- `erb_frame_50ms`：逐帧 8/13 dB 决策得到原声比例，乘 32 维 ERB 增益，然后对每个频带做 50 ms 因果一阶平滑。",
        "- `erb_fixed12_50ms`：固定原声比例 10^(-12/20)，乘 ERB 增益，再做同样的频带平滑；不依赖局部 SNR 阈值。",
        "- 两者通过模型的 ERB 反向映射展开为 481 个频率格，同一实数比例同时乘实部与虚部。ERB 增益不是频带真实 SNR，也不是语音存在概率。",
        "- 只测试上述两个预先指定的版本，gamma 固定为 1，不根据验证结果挑阈值、频段或平滑时间。",
        "",
        "## 数据与公平性", "",
        "探索：原 20 位 Common Voice 说话人标识，300 个带噪条件、1800 行策略结果，仅计算 SNR/MSE/STOI。",
        "独立验证：排除原六位及上述 20 位后，按固定种子、文本长度和时长选 12 个新 client_id，每位一条录音。三类合成噪声 × 五档 SNR = 180 个条件、1080 行结果，包含 Whisper small CER。",
        "",
        "音频仅在内存中加噪，48 kHz、单声道、干净峰值 0.3。同一条件各方案共用一次模型前向；每次分析/合成重置状态，恢复相同长度。CER 使用相同中文解码设置和繁简体/数字规范化。没有听感测试，也没有真实流式延迟测试。",
        "环境：Python 3.9.18，torch/torchaudio 2.5.1+cu121，deepfilternet/deepfilterlib 0.5.6。官方权重和录音均从 D: 读取，无新增模型下载或录音上传。",
        f"重跑的 1200 行探索基线与原记录核对：最大 SNR 差 {parity['output_snr_db']:.6f} dB，最大 STOI 差 {parity['output_stoi']:.8f}。浮点重跑并非逐位相同，但差异远小于本轮比较的效应。",
        "",
        "## 探索结果", "", *table(exploration["summary"], False), "",
        "## 独立验证结果", "", *table(validation["summary"], True), "",
        "## 配对差值与不确定性", "",
        "下列差值为候选减去基线，SNR/STOI 正值更好，CER 负值更好。按说话人整组重采样 5000 次，不将同一人的 15 个条件当成独立样本。区间为未经多重比较校正的描述性 95% bootstrap 区间。", "",
        "| 候选 | 基线 | 指标 | 差值 | 95% 区间 |",
        "| --- | --- | --- | ---: | --- |",
    ]
    for candidate, baselines in validation["paired"].items():
        for baseline, measures in baselines.items():
            for metric, values in measures.items():
                lo, hi = values["speaker_bootstrap_95pct_ci"]
                lines.append(f"| {candidate} | {baseline} | {metric} | {values['delta']:+.6f} | [{lo:+.6f}, {hi:+.6f}] |")
    lines += ["", "## ASR 异常诊断", "",
              "CER 可能超过 100%，原因是插入错误数可以超过参考字数；不删除这些原始结果，也不把 Whisper 错误直接解释为人耳听不清。以下给出各策略最差条件，方便核查。", ""]
    for strategy in STRATEGIES:
        group = [r for r in heldout if r["strategy"] == strategy]
        worst = max(group, key=lambda r: float(r["output_cer"]))
        lines.append(f"- {strategy}：CER > 1 的条件 {sum(float(r['output_cer']) > 1 for r in group)} 个；最高 CER {float(worst['output_cer']):.4f}，{worst['clip_path']} / {worst['noise_type']} / {worst['target_snr_db']} dB。")
    worst = max(heldout, key=lambda r: float(r["output_cer"]))
    if float(worst["output_cer"]) > 1:
        worst_key = (worst["clip_path"], worst["noise_type"], worst["target_snr_db"])
        lines += ["", "### 事后敏感性分析", "",
                  f"所有策略同时排除最大 CER 对应的同一个条件：{worst_key}；再同时排除其说话人全部条件。仅用于解释均值对识别异常的敏感性，不能代替上面的完整验证结果。", "",
                  "| 策略 | 排除同一个最差条件后的 CER | 排除其说话人后的 CER |",
                  "| --- | ---: | ---: |"]
        for strategy in STRATEGIES:
            group = [r for r in heldout if r["strategy"] == strategy]
            without_case = [float(r["output_cer"]) for r in group if (r["clip_path"], r["noise_type"], r["target_snr_db"]) != worst_key]
            without_speaker = [float(r["output_cer"]) for r in group if r["client_id"] != worst["client_id"]]
            lines.append(f"| {strategy} | {np.mean(without_case):.4%} | {np.mean(without_speaker):.4%} |")
    lines += ["", "## 解释边界", "",
              "SNR 的误差项包括残留噪声和语音失真，不能仅凭 SNR 证明沙沙声减少。STOI 是参考可懂度指标，CER 依赖单个识别器。没有测量音乐噪声、主观自然度或语言准确性的人类评分。",
              "验证样本只有 12 位说话人、三类合成噪声，不能外推到所有真实环境。验证结果一旦用于后续调参，该集合将变成开发数据，不能再声称它是新的独立测试。",
              "",
              "## 文件与复现", "",
              "入口 `scripts/research_erb_blending.py`；审计汇总 `scripts/summarize_erb_blending.py`。探索/验证 CSV、JSON、配置和验证名单均保存为 `results/erb_*`。已有结果默认拒绝覆盖，可用 `--resume` 继续。",
              "",
              "```powershell",
              "& C:/anaconda/envs/pytorch/python.exe scripts/research_erb_blending.py --resume",
              "& C:/anaconda/envs/pytorch/python.exe scripts/research_erb_blending.py --heldout --asr --resume",
              "& C:/anaconda/envs/pytorch/python.exe scripts/summarize_erb_blending.py",
              "```", ""]
    (RESULT_DIR / "erb_blending_review.md").write_text("\n".join(lines), encoding="utf-8")
    print("Audit passed: 20 exploration speakers, 12 disjoint held-out speakers, 2880 rows", flush=True)


if __name__ == "__main__":
    main()
