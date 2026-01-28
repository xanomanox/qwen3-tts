import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    device_map="cuda:1",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

source_script = input("Enter the path to the source script text file: ").strip()
with open(source_script, "r") as f:
    sentences = [line.strip() for line in f if line.strip()]
instruct_source = input("Enter the path to the instruction text file: ").strip()
with open(instruct_source, "r") as f:
    instruction = f.read().strip()

basename = "oa_script"
# um_generations = 50

for idx, sentence in enumerate(sentences):
    wavs, sr = model.generate_custom_voice(
        text=sentence,
        language="English", # Pass `Auto` (or omit) for auto language adaptive; if the target language is known, set it explicitly.
        speaker="Ono_Anna",
        instruct=instruction, # Omit if not needed.
    )
    sf.write(f"{basename}_{idx + 1}.wav", wavs[0], sr)


# batch inference
# wavs, sr = model.generate_custom_voice(
#     text=[
#         "It's time to relax and unwind after a long day.",
#     ],
#     language=["English"],
#     speaker=["Ono_Anna"],
#     instruct=["Speak slowly in a calm and soothing tone."],
# )
# sf.write("output_custom_voice_1.wav", wavs[0], sr)
# sf.write("output_custom_voice_2.wav", wavs[1], sr)