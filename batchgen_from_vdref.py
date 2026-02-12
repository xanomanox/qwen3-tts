
import json
import os
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

CACHE_PATH = os.path.join(os.path.dirname(__file__), ".batchgen_from_vdref_cache.json")


def _load_cache():
    if not os.path.exists(CACHE_PATH):
        return {}
    try:
        with open(CACHE_PATH, "r") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_cache(cache):
    try:
        with open(CACHE_PATH, "w") as f:
            json.dump(cache, f, indent=2)
    except Exception:
        pass


def _prompt_with_default(key, prompt, cache, validator=None, transform=None):
    default = cache.get(key)
    prompt_text = f"{prompt}"
    if default:
        prompt_text += f" [default: {default}]"
    prompt_text += ": "

    while True:
        value = input(prompt_text).strip()
        if not value and default is not None:
            value = str(default)
        if transform is not None:
            try:
                value = transform(value)
            except Exception:
                print("Invalid input type. Please try again.")
                continue
        if validator is not None and not validator(value):
            print("Invalid input value. Please try again.")
            continue
        cache[key] = value
        return value


cache = _load_cache()

source_script_path = _prompt_with_default(
    "source_script_path",
    "Enter the path to the source script text file",
    cache,
    validator=lambda v: isinstance(v, str) and os.path.isfile(v),
)
model_choice = _prompt_with_default(
    "model_choice",
    "Choose model (custom_voice/voice_clone)",
    cache,
    validator=lambda v: str(v).lower() in {"custom_voice", "voice_clone"},
    transform=lambda v: str(v).lower(),
)

if model_choice == "custom_voice":
    instruct_options_path = _prompt_with_default(
        "instruct_options_path",
        "Enter the path to the instruction options file",
        cache,
        validator=lambda v: isinstance(v, str) and os.path.isfile(v),
    )
    speaker_choice = _prompt_with_default(
        "speaker_choice",
        "Choose speaker (1=Ono_Anna, 2=Sohee, 3=Vivian, 4=Serena)",
        cache,
        validator=lambda v: isinstance(v, int) and 1 <= v <= 4,
        transform=lambda v: int(v),
    )
else:
    instruct_options_path = None
    speaker_choice = None

max_tokens = _prompt_with_default(
    "max_tokens",
    "Enter the maximum number of tokens per batch (min=64, max=768)",
    cache,
    validator=lambda v: isinstance(v, int) and 64 <= v <= 768,
    transform=lambda v: int(v),
)
basename = _prompt_with_default(
    "basename",
    "Enter the base name for output wav files",
    cache,
)
generation_count = _prompt_with_default(
    "generation_count",
    "how many generations to create for each instruction?",
    cache,
    validator=lambda v: isinstance(v, int) and 1 <= v <= 64,
    transform=lambda v: int(v),
)

_save_cache(cache)

if model_choice == "custom_voice":
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:1",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    with open(instruct_options_path, "r") as f:
        raw_instructions = f.read()
    if not raw_instructions or not raw_instructions.strip():
        raise ValueError("Instruction options file is empty.")
    if "====" not in raw_instructions and len(raw_instructions.strip()) > 1024:
        raise ValueError(
            "Instruction exceeds 1024 characters without a separator in instruction options file."
        )
    instructions = []
    current_lines = []
    for line in raw_instructions.splitlines():
        if line.strip() == "====":
            chunk = "\n".join(current_lines).strip()
            if chunk:
                instructions.append(chunk)
            current_lines = []
        else:
            current_lines.append(line)
    final_chunk = "\n".join(current_lines).strip()
    if final_chunk:
        instructions.append(final_chunk)
    if not instructions:
        raise ValueError("No instructions found in instruction options file.")
    speaker_map = {
        1: "Ono_Anna",
        2: "Sohee",
        3: "Vivian",
        4: "Serena",
    }
    speaker = speaker_map[speaker_choice]
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
    instructions = [None]
    speaker = None

with open(source_script_path, "r") as f:
    sentences = [line.strip() for line in f if line.strip()]

processor = model.processor


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

for instr_idx, instruction in enumerate(instructions, start=1):
    for gen_idx in range(1, generation_count + 1):
        for batch_idx, batch_text in enumerate(batches, start=1):
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
            out_path = f"{basename}_instr{instr_idx}_gen{gen_idx}_batch{batch_idx}.wav"
            sf.write(out_path, wavs[0], sr)
            print(
                f"Generated instr {instr_idx} gen {gen_idx} batch {batch_idx} with {batch_tokens} tokens."
            )


# or batch generate in one call. this takes lots of memory and will blow out a 8GB.
# wavs, sr = clone_model.generate_voice_clone(
#     text=sentences,
#     language="English",
#     voice_clone_prompt=voice_clone_prompt,
# )
# for i, w in enumerate(wavs):
#     sf.write(f"{basename}_{i}.wav", w, sr)