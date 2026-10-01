# DeepFilterNet3 复现与策略评测：个人项目记录

本项目以 [DeepFilterNet](https://github.com/Rikorose/DeepFilterNet) 官方预训练 DeepFilterNet3 为基础。目标是理解语音增强链路，并研究降噪强度与语音保真度的权衡；**没有从头训练或微调模型，也没有完成流式部署**。

## 已完成的工作

1. 跑通预训练模型的 WAV 推理，检查采样率、波形、STFT/ERB/DF 特征及模型输出的形状。理解共享编码器、ERB 掩码分支、DF 复数滤波分支和局部 SNR 输出的数据流。
2. 实现基于局部 SNR 第 75 百分位数（Q75）的探索性增强策略：Q75 `<8 dB` 使用完整增强；`8–13 dB` 使用 12 dB 混合限制；`>=13 dB` 使用 6 dB 混合限制。阈值来自早期单人声音频实验，并非在公开测试集上优化得到。
3. 在相同模型前向结果上比较完整增强、固定 12/6 dB 混合、自适应混合、自适应幅度下限五种后处理。指标包括参考音频 SNR/MSE、STOI、Whisper CER 和人工听感。SNR 是有干净参考的离线评价值；模型输出的局部 SNR/Q75 是无参考预测，两者不能混称。
4. 曾在五条自录音频上比较一遍式与两遍式处理：输出波形最大差异为 0；当次计时中一遍式耗时约降低 42%–52%。这是特定机器与样本上的离线计时，**不能**作为流式端到端延迟或普遍加速结论。
5. 使用自录语音及 Common Voice 中文子集做独立说话人检查。公开数据部分选取 `test.tsv` 中六个不同 `client_id` 的片段，每条加入三类合成噪声 × 五档输入 SNR，得到 90 个带噪条件、450 条策略结果。模型参数和 Q75 阈值均保持不变。

## 核心文件与结果

| 文件 | 作用 |
| --- | --- |
| `scripts/compare_multinoise_strategies.py` | 自录音频的多噪声、五策略对照；定义共用的策略和指标函数 |
| `scripts/compare_multisignal.py` | 多条自录音频的额外验证 |
| `scripts/compare_commonvoice_speakers.py` | 六个公开测试说话人的离线对照；音频在内存中动态加噪 |
| `scripts/rescore_commonvoice_cer.py` | 对已保存的识别文本统一繁简体与数字写法后重算 CER，不重复模型推理 |
| `scripts/export_commonvoice_listening.py` | 导出三个事后选定的诊断片段供听感对照 |
| `results/commonvoice_6speakers.csv` | 原始 450 行结果 |
| `results/commonvoice_6speakers_cer_normalized.csv` | 统一文本规范后的 CER 结果 |
| `results/commonvoice_case_review.md` | 三个片段的指标与听感记录、选择偏差说明 |

在六说话人集合上，完整增强的平均输出 SNR 约 12.28 dB、归一化 CER 约 0.33；固定 12 dB 混合约为 11.22 dB、0.29；固定 6 dB 混合约为 9.09 dB、0.28。STOI 均约 0.92。这表明**在这组短语音和这个 ASR 模型上**，更高 SNR 未必对应更低 CER。自适应策略没有在该集合上证明优于固定策略，不能宣称“全面提升”。三个听感例子是看过结果后选出的诊断个案，不能当作无偏性能估计。

## 复现实验的边界

- 运行环境为 Windows、Python 3.9、PyTorch/CUDA；官方模型和音频位于 `D:\github\DeepFilterNetAudio`，该路径写在脚本常量中，换机器需先修改。模型权重、录音和 Common Voice 音频不在此 Git 仓库。
- 评测依赖 `deepfilternet`、`torch`、`torchaudio`、`pystoi`、`openai-whisper`；CER 重算另需 `opencc-python-reimplemented`。本机部分包放在未跟踪的 `scripts/_local_packages`，克隆仓库后需自行安装依赖。
- 音频转写的 CER 受语音内容、Whisper 识别错误及文本规范化影响；简短片段的 CER 波动尤其大。STOI、SNR、CER 和听感应一起看。
- 目前六个公开片段只支持探索性结论。若以后要发表或声称策略泛化提升，须在独立开发集定规则，再冻结策略并在未查看过的测试说话人上评估。
- 当前工作没有训练 DeepFilterNet3、没有 ONNX/量化部署、没有真实流式延迟测试；简历和面试介绍应准确区分这些边界。

## 建议的简历表述

“基于 DeepFilterNet3 预训练模型实现语音增强推理与可复现评测，梳理 ERB 掩码、复数深度滤波和局部 SNR 数据流；设计 Q75 驱动的降噪强度选择，并在自录及六位公开语音说话人的多噪声条件下比较 SNR、STOI、CER 与听感，发现降噪强度和语音可懂度存在权衡。”
