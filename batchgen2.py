
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# given a voice design reference audio, 
source_script_path = input("Enter the path to the source script text file: ").strip()

ref_text = "I absolutely adore your plans." 

# build a reusable clone prompt from the voice design reference
clone_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:1",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio="voice_design_reference.wav",
    ref_text=ref_text,
)

with open(source_script_path, "r") as f:
    sentences = [line.strip() for line in f if line.strip()]

# reuse it for multiple single calls
for i, sentence in enumerate(sentences):
    wavs, sr = clone_model.generate_voice_clone(
        text=sentence,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
    )
    sf.write(f"sj_{i}.wav", wavs[0], sr)


# or batch generate in one call
# wavs, sr = clone_model.generate_voice_clone(
#     text=sentences,
#     language="English",
#     voice_clone_prompt=voice_clone_prompt,
# )
# for i, w in enumerate(wavs):
#     sf.write(f"sj_{i}.wav", w, sr)