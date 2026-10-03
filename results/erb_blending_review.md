# ERB 掩码引导的子带混合研究（2026-10-03）

## 本轮结论

保留固定比例 ERB 方案作为小规模候选，不继续扩展逐帧 SNR × ERB 方案。固定比例 ERB 在两批数据上相对 full 均有小幅 SNR/STOI 均值改善，新 12 人验证的描述性差值区间也支持这一方向。它提供的是折中：相对普通固定 12 dB 保住更多 SNR，但 CER 更差，不能声称全面优于现有方案。

新验证中 full 的平均 CER 明显受一个 CER=20 的重复识别条件影响。所有策略同时排除该条件后，full 的 CER 为约 32.47%，固定 ERB 约 28.00%，普通固定 12 dB 约 25.80%。完整结果仍为主结果，敏感性分析不代替独立验证。尚未证明说话起始的沙沙声或主观闷声已经解决。

## 目的与方案

检验子带约束能否改善统一整帧混合的噪声回混问题。本轮不训练、不微调、不下载模型，也不修改已有评测脚本和旧结果。

- `erb_frame_50ms`：逐帧 8/13 dB 决策得到原声比例，乘 32 维 ERB 增益，然后对每个频带做 50 ms 因果一阶平滑。
- `erb_fixed12_50ms`：固定原声比例 10^(-12/20)，乘 ERB 增益，再做同样的频带平滑；不依赖局部 SNR 阈值。
- 两者通过模型的 ERB 反向映射展开为 481 个频率格，同一实数比例同时乘实部与虚部。ERB 增益不是频带真实 SNR，也不是语音存在概率。
- 只测试上述两个预先指定的版本，gamma 固定为 1，不根据验证结果挑阈值、频段或平滑时间。

## 数据与公平性

探索：原 20 位 Common Voice 说话人标识，300 个带噪条件、1800 行策略结果，仅计算 SNR/MSE/STOI。
独立验证：排除原六位及上述 20 位后，按固定种子、文本长度和时长选 12 个新 client_id，每位一条录音。三类合成噪声 × 五档 SNR = 180 个条件、1080 行结果，包含 Whisper small CER。

音频仅在内存中加噪，48 kHz、单声道、干净峰值 0.3。同一条件各方案共用一次模型前向；每次分析/合成重置状态，恢复相同长度。CER 使用相同中文解码设置和繁简体/数字规范化。没有听感测试，也没有真实流式延迟测试。
环境：Python 3.9.18，torch/torchaudio 2.5.1+cu121，deepfilternet/deepfilterlib 0.5.6。官方权重和录音均从 D: 读取，无新增模型下载或录音上传。
重跑的 1200 行探索基线与原记录核对：最大 SNR 差 0.001168 dB，最大 STOI 差 0.00002797。浮点重跑并非逐位相同，但差异远小于本轮比较的效应。

## 探索结果

| 方案 | 平均输出 SNR / dB | 平均 STOI |
| --- | ---: | ---: |
| full | 11.5544 | 0.881115 |
| fixed_mix_12db | 10.8169 | 0.886427 |
| q75_mix | 11.6034 | 0.887090 |
| frame_smooth_50ms | 11.7105 | 0.880537 |
| erb_frame_50ms | 11.6515 | 0.881500 |
| erb_fixed12_50ms | 11.7169 | 0.887248 |

## 独立验证结果

| 方案 | 平均输出 SNR / dB | 平均 STOI | 平均 CER | CER 中位数 |
| --- | ---: | ---: | ---: | ---: |
| full | 11.9713 | 0.891985 | 43.3997% | 22.2222% |
| fixed_mix_12db | 11.0846 | 0.903197 | 26.0599% | 20.0000% |
| q75_mix | 11.9955 | 0.899050 | 40.9217% | 20.0000% |
| frame_smooth_50ms | 12.0377 | 0.889913 | 35.0534% | 22.2222% |
| erb_frame_50ms | 12.0461 | 0.891861 | 43.1106% | 22.2222% |
| erb_fixed12_50ms | 12.1199 | 0.901507 | 28.5472% | 20.7143% |

## 配对差值与不确定性

下列差值为候选减去基线，SNR/STOI 正值更好，CER 负值更好。按说话人整组重采样 5000 次，不将同一人的 15 个条件当成独立样本。区间为未经多重比较校正的描述性 95% bootstrap 区间。

| 候选 | 基线 | 指标 | 差值 | 95% 区间 |
| --- | --- | --- | ---: | --- |
| erb_frame_50ms | full | output_snr_db | +0.074815 | [+0.049619, +0.099419] |
| erb_frame_50ms | full | output_stoi | -0.000124 | [-0.001281, +0.000878] |
| erb_frame_50ms | full | output_cer | -0.002890 | [-0.008983, +0.002470] |
| erb_frame_50ms | fixed_mix_12db | output_snr_db | +0.961522 | [+0.410733, +1.491569] |
| erb_frame_50ms | fixed_mix_12db | output_stoi | -0.011336 | [-0.027082, +0.003419] |
| erb_frame_50ms | fixed_mix_12db | output_cer | +0.170507 | [+0.028722, +0.416503] |
| erb_frame_50ms | q75_mix | output_snr_db | +0.050572 | [-0.325719, +0.390892] |
| erb_frame_50ms | q75_mix | output_stoi | -0.007189 | [-0.016339, +0.000461] |
| erb_frame_50ms | q75_mix | output_cer | +0.021889 | [+0.000679, +0.056729] |
| erb_frame_50ms | frame_smooth_50ms | output_snr_db | +0.008401 | [-0.122353, +0.143399] |
| erb_frame_50ms | frame_smooth_50ms | output_stoi | +0.001948 | [-0.000114, +0.004261] |
| erb_frame_50ms | frame_smooth_50ms | output_cer | +0.080572 | [-0.073357, +0.316797] |
| erb_fixed12_50ms | full | output_snr_db | +0.148640 | [+0.047481, +0.254775] |
| erb_fixed12_50ms | full | output_stoi | +0.009523 | [+0.000809, +0.019102] |
| erb_fixed12_50ms | full | output_cer | -0.148524 | [-0.375270, -0.020228] |
| erb_fixed12_50ms | fixed_mix_12db | output_snr_db | +1.035347 | [+0.588824, +1.470422] |
| erb_fixed12_50ms | fixed_mix_12db | output_stoi | -0.001690 | [-0.008503, +0.004749] |
| erb_fixed12_50ms | fixed_mix_12db | output_cer | +0.024874 | [+0.005863, +0.046324] |
| erb_fixed12_50ms | q75_mix | output_snr_db | +0.124396 | [-0.176080, +0.389708] |
| erb_fixed12_50ms | q75_mix | output_stoi | +0.002457 | [-0.000743, +0.006209] |
| erb_fixed12_50ms | q75_mix | output_cer | -0.123745 | [-0.349431, -0.000557] |
| erb_fixed12_50ms | frame_smooth_50ms | output_snr_db | +0.082226 | [-0.023599, +0.199573] |
| erb_fixed12_50ms | frame_smooth_50ms | output_stoi | +0.011594 | [+0.002408, +0.022660] |
| erb_fixed12_50ms | frame_smooth_50ms | output_cer | -0.065062 | [-0.132379, -0.017929] |

## ASR 异常诊断

CER 可能超过 100%，原因是插入错误数可以超过参考字数；不删除这些原始结果，也不把 Whisper 错误直接解释为人耳听不清。以下给出各策略最差条件，方便核查。

- full：CER > 1 的条件 4 个；最高 CER 20.0000，common_voice_zh-CN_18909682.mp3 / low_frequency / -5 dB。
- fixed_mix_12db：CER > 1 的条件 0 个；最高 CER 1.0000，common_voice_zh-CN_22035201.mp3 / low_frequency / -5 dB。
- q75_mix：CER > 1 的条件 3 个；最高 CER 20.0000，common_voice_zh-CN_18909682.mp3 / low_frequency / -5 dB。
- frame_smooth_50ms：CER > 1 的条件 4 个；最高 CER 5.3571，common_voice_zh-CN_43283740.mp3 / low_frequency / -5 dB。
- erb_frame_50ms：CER > 1 的条件 4 个；最高 CER 20.0000，common_voice_zh-CN_18909682.mp3 / low_frequency / -5 dB。
- erb_fixed12_50ms：CER > 1 的条件 3 个；最高 CER 1.2727，common_voice_zh-CN_18909682.mp3 / low_frequency / -5 dB。

### 事后敏感性分析

所有策略同时排除最大 CER 对应的同一个条件：('common_voice_zh-CN_18909682.mp3', 'low_frequency', '-5')；再同时排除其说话人全部条件。仅用于解释均值对识别异常的敏感性，不能代替上面的完整验证结果。

| 策略 | 排除同一个最差条件后的 CER | 排除其说话人后的 CER |
| --- | ---: | ---: |
| full | 32.4689% | 30.8713% |
| fixed_mix_12db | 25.7992% | 24.9028% |
| q75_mix | 29.9772% | 28.1681% |
| frame_smooth_50ms | 34.6398% | 33.1712% |
| erb_frame_50ms | 32.1783% | 30.4457% |
| erb_fixed12_50ms | 27.9957% | 26.6796% |

## 解释边界

SNR 的误差项包括残留噪声和语音失真，不能仅凭 SNR 证明沙沙声减少。STOI 是参考可懂度指标，CER 依赖单个识别器。没有测量音乐噪声、主观自然度或语言准确性的人类评分。
验证样本只有 12 位说话人、三类合成噪声，不能外推到所有真实环境。验证结果一旦用于后续调参，该集合将变成开发数据，不能再声称它是新的独立测试。

## 文件与复现

入口 `scripts/research_erb_blending.py`；审计汇总 `scripts/summarize_erb_blending.py`。探索/验证 CSV、JSON、配置和验证名单均保存为 `results/erb_*`。已有结果默认拒绝覆盖，可用 `--resume` 继续。

```powershell
& C:/anaconda/envs/pytorch/python.exe scripts/research_erb_blending.py --resume
& C:/anaconda/envs/pytorch/python.exe scripts/research_erb_blending.py --heldout --asr --resume
& C:/anaconda/envs/pytorch/python.exe scripts/summarize_erb_blending.py
```
