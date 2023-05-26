import gradio as gr
import numpy as np

def generate_audio():
    # 生成一个随机的 5 秒钟的音频片段
    sr = 44100  # 音频采样率
    duration = 5  # 音频时长（秒）
    t = np.linspace(0, duration, int(sr * duration), False)
    freq = 440.0
    audio = 0.5 * np.sin(2 * np.pi * freq * t)
    return audio, sr

audio, sr = generate_audio()

audio_element = gr.Audio(audio, type="numpy", sample_rate=sr, autoplay=True)

iface = gr.Interface(
    fn=lambda: audio_element,
    inputs=None,
    outputs="html",
    title="Automatic Audio Playback Demo",
    theme="default"
)

iface.launch()
