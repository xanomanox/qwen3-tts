import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:1",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

ref_audio = input("Enter the path to the reference audio file: ").strip()
ref_text_path  = input("Enter the path to the reference text used in the reference audio: ").strip()
with open(ref_text_path, "r") as f:
    ref_text = f.read().strip()

wavs, sr = model.generate_voice_clone(
    text="You can't do that or you'll blow the reactor and fry the ship ... and us!",
    instruct="Speak very slowly and angrily, with a loud and forceful tone, and just a little bit of panic.",
    language="English",
    ref_audio=ref_audio,
    ref_text=ref_text,
)
sf.write("instr_clone.wav", wavs[0], sr)