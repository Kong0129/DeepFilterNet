import math

import torch
from torch.nn import functional as F

from df.enhance import df_features
from df.utils import as_complex


from df.enhance import init_df, load_audio, save_audio
from df.enhance import enhance, init_df, load_audio, save_audio
import sys
from pathlib import Path

# 根据当前文件的位置，找到上面的 scripts 目录。
scripts_dir = Path(__file__).resolve().parents[2]

# 告诉 Python：导入模块时，也到这个目录里寻找。
sys.path.insert(0, str(scripts_dir))

from compare_multinoise_strategies import calculate_stoi


def calculate_snr(estimate, target):
    # Keep this helper local: importing add_noise would generate audio files.
    signal_power = torch.mean(target ** 2)
    error_power = torch.mean((estimate - target) ** 2)
    return (10 * torch.log10(signal_power / error_power)).item()

def enhance_erb_fixed12(model, df_state, audio):
    # 当前练习只处理单声道：[1, samples]。
    if audio.ndim != 2 or audio.shape[0] != 1:
        raise ValueError("请输入形状为 [1, samples] 的单声道音频")

    original_length = audio.shape[-1]
    device = next(model.parameters()).device

    # 每条独立音频开始前，重置处理状态。
    df_state.reset()
    if hasattr(model, "reset_h0"):
        model.reset_h0(batch_size=1, device=device)

    # 1. 波形补零，再提取模型需要的三个输入。
    padded_audio = F.pad(
        audio,
        (0, df_state.fft_size()),
    )

    spec, erb_feature, spec_feature = df_features(
        padded_audio,
        df_state,
        model.nb_df,
        device=device,
    )

    model.eval()

    with torch.no_grad():
        # 2. 同一次前向得到增强频谱和 ERB 增益。
        enhanced_spec, erb_mask, _, _ = model(
            spec.clone(),
            erb_feature,
            spec_feature,
        )

        # 3. 得到频带级原声比例：[B, 1, T, 32]。
        original_ratio = 10 ** (-12 / 20)
        target_ratio = original_ratio * erb_mask.clamp(0, 1)

        # 4. 每个频带独立沿时间平滑。
        source = target_ratio.detach().cpu()
        smooth_ratio = torch.empty_like(source)

        hop_seconds = df_state.hop_size() / df_state.sr()
        alpha = math.exp(-hop_seconds / 0.05)

        smooth_ratio[:, :, 0] = source[:, :, 0]

        for t in range(1, source.shape[2]):
            smooth_ratio[:, :, t] = (
                alpha * smooth_ratio[:, :, t - 1]
                + (1 - alpha) * source[:, :, t]
            )

        smooth_ratio = smooth_ratio.to(device)

        # 5. 32 个频带 → 481 个频率格，再增加广播维度。
        frequency_ratio = (
            smooth_ratio @ model.mask.erb_inv_fb
        ).unsqueeze(-1)

        # 6. 混合原始带噪频谱与增强频谱。
        selected_spec = (
            spec * frequency_ratio
            + enhanced_spec * (1 - frequency_ratio)
        )

    # 7. 转回复数表示，合成波形。
    complex_spec = as_complex(
        selected_spec.squeeze(1).cpu()
    ).numpy()

    df_state.reset()
    output = torch.as_tensor(
        df_state.synthesis(complex_spec)
    )

    # 去掉分析/合成延迟，恢复原音频长度。
    delay = df_state.fft_size() - df_state.hop_size()
    output = output[:, delay : delay + original_length]

    return output




root = Path(r"D:\github\DeepFilterNetAudio")

model_dir = root / "models" / "DeepFilterNet3"

# 可以换成你已有的一条带噪 WAV 文件。
input_path = (
    root / "own_voice_evaluation" / "noisy_snr_5db.wav"
)

output_path = (
    root / "own_voice_evaluation" / "erb_fixed12_practice.wav"
)

model, df_state, _ = init_df(
    model_base_dir=str(model_dir),
    log_file=None,
)

audio, _ = load_audio(
    str(input_path),
    sr=df_state.sr(),
)
reference_path = (
    root
    / "own_voice_evaluation"
    / "clean_reference.wav"
)

reference, _ = load_audio(
    str(reference_path),
    sr=df_state.sr(),
)

reference = reference.mean(dim=0, keepdim=True)

# 双声道输入也先转为单声道。
audio = audio.mean(dim=0, keepdim=True)
# 完整增强：不掺回原声。
df_state.reset()

if hasattr(model, "reset_h0"):
    model.reset_h0(
        batch_size=1,
        device=next(model.parameters()).device,
    )

full_output = enhance(
    model,
    df_state,
    audio,
    pad=True,
    atten_lim_db=None,
)

full_output_path = (
    root / "own_voice_evaluation" / "full_practice.wav"
)

save_audio(
    str(full_output_path),
    full_output,
    df_state.sr(),
)

print("full 输出形状:", full_output.shape)
print("full 保存位置:", full_output_path)
output = enhance_erb_fixed12(
    model,
    df_state,
    audio,
)

print("输入波形形状:", audio.shape)
print("输出波形形状:", output.shape)

save_audio(
    str(output_path),
    output,
    df_state.sr(),
)

print("保存位置:", output_path)


print("带噪输入 SNR:", calculate_snr(audio, reference))
print("full SNR:", calculate_snr(full_output, reference))
print("ERB SNR:", calculate_snr(output, reference))

print("带噪输入 STOI:", calculate_stoi(reference, audio, df_state.sr()))
print("full STOI:", calculate_stoi(reference, full_output, df_state.sr()))
print("ERB STOI:", calculate_stoi(reference, output, df_state.sr()))
