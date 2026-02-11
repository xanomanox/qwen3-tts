
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

source_script_path = input("Enter the path to the source script text file: ").strip()
model_choice = input("Choose model (custom_voice/voice_clone): ").strip().lower()

if model_choice == "custom_voice":
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:1",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    instruct_source = input("Enter the path to the instruction text file: ").strip()
    with open(instruct_source, "r") as f:
        instruction = f.read().strip()
    speaker = input("Enter the speaker name (e.g., Ono_Anna, Sohee, Vivian, Serena): ").strip()
    voice_clone_prompt = None
else:
    ref_text = "I absolutely adore your plans."
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:1",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    voice_clone_prompt = model.create_voice_clone_prompt(
        ref_audio="voice_design_reference.wav",
        ref_text=ref_text,
    )
    instruction = None
    speaker = None

with open(source_script_path, "r") as f:
    sentences = [line.strip() for line in f if line.strip()]

processor = model.processor
max_tokens = int(input("Enter the maximum number of tokens per batch (e.g., 2000): ").strip())


def _count_tokens(text: str) -> int:
    enc = processor(text=text, return_tensors="pt", padding=False)
    return int(enc["input_ids"].shape[-1])


def _make_batches(sentences_list):
    batches = []
    current = []
    current_tokens = 0

    for idx, sentence in enumerate(sentences_list):
        sentence_tokens = _count_tokens(sentence)
        if sentence_tokens > max_tokens:
            print(
                f"Skipping sentence {idx}: {sentence_tokens} tokens exceeds max_tokens={max_tokens}."
            )
            continue

        if current and (current_tokens + sentence_tokens > max_tokens):
            batches.append(" ".join(current))
            current = [sentence]
            current_tokens = sentence_tokens
        else:
            current.append(sentence)
            current_tokens += sentence_tokens

    if current:
        batches.append(" ".join(current))
        print(f"current {current_tokens}:", current)
    return batches


batches = _make_batches(sentences)
basename = input("Enter the base name for output wav files: ").strip()

for i, batch_text in enumerate(batches):
    batch_tokens = _count_tokens(batch_text)
    if model_choice == "custom_voice":
        wavs, sr = model.generate_custom_voice(
            text=batch_text,
            language="English",
            speaker=speaker,
            instruct=instruction,
        )
    else:
        wavs, sr = model.generate_voice_clone(
            text=batch_text,
            language="English",
            voice_clone_prompt=voice_clone_prompt,
        )
    sf.write(f"{basename}_batch_{i}.wav", wavs[0], sr)
    print(f"Generated batch {i} with {batch_tokens} tokens.")


# or batch generate in one call. this takes lots of memory and will blow out a 8GB.
# wavs, sr = clone_model.generate_voice_clone(
#     text=sentences,
#     language="English",
#     voice_clone_prompt=voice_clone_prompt,
# )
# for i, w in enumerate(wavs):
#     sf.write(f"{basename}_{i}.wav", w, sr)