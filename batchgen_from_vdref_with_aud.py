
import json
import os
import shutil
import subprocess
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


_MISSING = object()


def _parse_bool(raw: str) -> bool:
    v = raw.strip().lower()
    if v in {"1", "true", "t", "y", "yes", "on"}:
        return True
    if v in {"0", "false", "f", "n", "no", "off"}:
        return False
    raise ValueError("Expected a boolean (true/false, y/n, 1/0).")


def _prompt_with_default(
    key,
    prompt,
    cache,
    *,
    default=_MISSING,
    parser=None,
    validator=None,
):
    """Prompt for a value.

    Default resolution:
    - if key exists in cache => use cached value as the presented default
    - else if default is provided => use that as the presented default
    - else => no default (input required)

    The chosen/parsed value is stored back into cache.
    """
    presented_default = cache.get(key, default)

    prompt_text = f"{prompt}"
    if presented_default is not _MISSING:
        prompt_text += f" [default: {presented_default}]"
    prompt_text += ": "

    while True:
        raw = input(prompt_text)
        raw_stripped = raw.strip()

        if raw_stripped == "":
            if presented_default is _MISSING:
                print("A value is required. Please try again.")
                continue
            value = presented_default
        else:
            if parser is None:
                value = raw_stripped
            else:
                try:
                    value = parser(raw_stripped)
                except Exception:
                    print("Invalid input type. Please try again.")
                    continue

        if validator is not None and not validator(value):
            print("Invalid input value. Please try again.")
            continue

        # Store back for reuse as defaults in later prompts.
        cache[key] = value
        return value


def _sox_play_once(path: str) -> None:
    play_path = "/usr/bin/play"
    if not os.path.exists(play_path):
        play_path = shutil.which("play")
    if not play_path:
        print("sox 'play' command not found; skipping playback for", path)
        return

    try:
        proc = subprocess.Popen(
            [play_path, "-q", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        proc.wait()
    except Exception as e:
        print("Playback failed:", e)


def _get_resolved_sampling_defaults(model):
    """Return resolved defaults for sampling parameters as actually used by this model.

    Prefers Qwen3TTSModel's internal default-merging logic (generate_config / model config)
    so the defaults shown to the user match runtime behavior.
    """
    fallback = {
        "do_sample": True,
        "temperature": 1.0,
        "top_k": 50,
        "top_p": 0.9,
        "subtalker_dosample": True,
        "subtalker_temperature": 1.0,
        "subtalker_top_k": 50,
        "subtalker_top_p": 0.9,
    }
    try:
        prepare = getattr(model, "_prepare_generate_kwargs", None)
        if prepare is None:
            return fallback
        resolved = prepare()  # type: ignore[call-arg]
        for k in list(fallback.keys()):
            if k in resolved:
                fallback[k] = resolved[k]
        return fallback
    except Exception:
        return fallback


cache = _load_cache()
model_choice = _prompt_with_default(
    "model_choice",
    "Choose model (custom_voice/voice_clone)",
    cache,
    default="custom_voice",
    parser=lambda v: str(v).lower(),
    validator=lambda v: v in {"custom_voice", "voice_clone"},
)

if model_choice == "custom_voice":
    model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
        device_map="cuda:1",
        dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    # These will be collected in the audition loop (instruction file, speaker selection, etc.)
    instructions = []
    speaker = None
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


def _collect_sampling_params(cache_dict, defaults):
    do_sample = _prompt_with_default(
        "do_sample",
        "do_sample (true/false)",
        cache_dict,
        default=defaults.get("do_sample", True),
        parser=_parse_bool,
    )
    temperature = _prompt_with_default(
        "temperature",
        "temperature (float)",
        cache_dict,
        default=defaults.get("temperature", 1.0),
        parser=float,
        validator=lambda v: isinstance(v, (int, float)) and v >= 0,
    )
    top_k = _prompt_with_default(
        "top_k",
        "top_k (int)",
        cache_dict,
        default=defaults.get("top_k", 50),
        parser=int,
        validator=lambda v: isinstance(v, int) and v >= 0,
    )
    top_p = _prompt_with_default(
        "top_p",
        "top_p (0.0 - 1.0)",
        cache_dict,
        default=defaults.get("top_p", 0.9),
        parser=float,
        validator=lambda v: isinstance(v, (int, float)) and 0 <= float(v) <= 1,
    )

    subtalker_dosample = _prompt_with_default(
        "subtalker_dosample",
        "subtalker_dosample (true/false)",
        cache_dict,
        default=defaults.get("subtalker_dosample", True),
        parser=_parse_bool,
    )
    subtalker_temperature = _prompt_with_default(
        "subtalker_temperature",
        "subtalker_temperature (float)",
        cache_dict,
        default=defaults.get("subtalker_temperature", 1.0),
        parser=float,
        validator=lambda v: isinstance(v, (int, float)) and v >= 0,
    )
    subtalker_top_k = _prompt_with_default(
        "subtalker_top_k",
        "subtalker_top_k (int)",
        cache_dict,
        default=defaults.get("subtalker_top_k", 50),
        parser=int,
        validator=lambda v: isinstance(v, int) and v >= 0,
    )
    subtalker_top_p = _prompt_with_default(
        "subtalker_top_p",
        "subtalker_top_p (0.0 - 1.0)",
        cache_dict,
        default=defaults.get("subtalker_top_p", 0.9),
        parser=float,
        validator=lambda v: isinstance(v, (int, float)) and 0 <= float(v) <= 1,
    )

    return {
        "do_sample": do_sample,
        "temperature": float(temperature),
        "top_k": int(top_k),
        "top_p": float(top_p),
        "subtalker_dosample": subtalker_dosample,
        "subtalker_temperature": float(subtalker_temperature),
        "subtalker_top_k": int(subtalker_top_k),
        "subtalker_top_p": float(subtalker_top_p),
    }

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


def _load_sentences(script_path: str):
    with open(script_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def _parse_instruction_file(path: str):
    with open(path, "r") as f:
        raw_instructions = f.read()

    if not raw_instructions or not raw_instructions.strip():
        raise ValueError("Instruction options file is empty.")
    if "====" not in raw_instructions and len(raw_instructions.strip()) > 1024:
        raise ValueError(
            "Instruction exceeds 1024 characters without a separator in instruction options file."
        )

    parsed = []
    current_lines = []
    for line in raw_instructions.splitlines():
        if line.strip() == "====":
            chunk = "\n".join(current_lines).strip()
            if chunk:
                parsed.append(chunk)
            current_lines = []
        else:
            current_lines.append(line)

    final_chunk = "\n".join(current_lines).strip()
    if final_chunk:
        parsed.append(final_chunk)
    if not parsed:
        raise ValueError("No instructions found in instruction options file.")
    return parsed


def _collect_audition_and_sampling(cache_dict, model_obj):
    resolved_defaults = _get_resolved_sampling_defaults(model_obj)

    # Audition text is mode-agnostic. Use the ref_text string as a sensible first-run default.
    audition_text = _prompt_with_default(
        "audition_text",
        "Enter audition text",
        cache_dict,
        default="I absolutely adore your plans.",
        validator=lambda v: isinstance(v, str) and len(v.strip()) > 0,
        parser=str,
    )

    sampling_kwargs = _collect_sampling_params(cache_dict, resolved_defaults)
    return audition_text, sampling_kwargs


def _generate_audition_wav(text, sampling_kwargs, *, instruction=None):
    if model_choice == "custom_voice":
        wavs, sr = model.generate_custom_voice(
            text=text,
            language="English",
            speaker=speaker,
            instruct=instruction,
            **sampling_kwargs,
        )
        return wavs, sr

    wavs, sr = model.generate_voice_clone(
        text=text,
        language="English",
        voice_clone_prompt=voice_clone_prompt,
        **sampling_kwargs,
    )
    return wavs, sr


def _audition_loop(cache_dict):
    # Persist these across 'i' cycles.
    audition_text = None
    sampling_kwargs = None
    instruction_idx = 0

    while True:
        # Allow modifying any/all previously supplied parameters (via 'r').
        # Re-prompt core inputs first.
        global source_script_path, instruct_options_path, speaker_choice, max_tokens, basename, generation_count
        global instructions, speaker

        source_script_path = _prompt_with_default(
            "source_script_path",
            "Enter the path to the source script text file",
            cache_dict,
            validator=lambda v: isinstance(v, str) and os.path.isfile(v),
        )
        if model_choice == "custom_voice":
            instruct_options_path = _prompt_with_default(
                "instruct_options_path",
                "Enter the path to the instruction options file",
                cache_dict,
                validator=lambda v: isinstance(v, str) and os.path.isfile(v),
            )
            speaker_choice = _prompt_with_default(
                "speaker_choice",
                "Choose speaker (1=Ono_Anna, 2=Sohee, 3=Vivian, 4=Serena)",
                cache_dict,
                default=1,
                parser=int,
                validator=lambda v: isinstance(v, int) and 1 <= v <= 4,
            )
        max_tokens = _prompt_with_default(
            "max_tokens",
            "Enter the maximum number of tokens per batch (min=64, max=768)",
            cache_dict,
            default=256,
            parser=int,
            validator=lambda v: isinstance(v, int) and 64 <= v <= 768,
        )
        basename = _prompt_with_default(
            "basename",
            "Enter the base name for output wav files",
            cache_dict,
            default="output",
        )
        generation_count = _prompt_with_default(
            "generation_count",
            "How many generations to create?",
            cache_dict,
            default=1,
            parser=int,
            validator=lambda v: isinstance(v, int) and 1 <= v <= 64,
        )

        # Refresh derived settings (instructions / speaker / batches).
        if model_choice == "custom_voice":
            instructions = _parse_instruction_file(instruct_options_path)
            speaker_map = {
                1: "Ono_Anna",
                2: "Sohee",
                3: "Vivian",
                4: "Serena",
            }
            speaker = speaker_map[speaker_choice]
            try:
                instruction_idx = int(cache_dict.get("instruction_index", 0))
            except Exception:
                instruction_idx = 0
            if instructions:
                instruction_idx %= len(instructions)
            cache_dict["instruction_index"] = instruction_idx

        sentences_local = _load_sentences(source_script_path)
        batches_local = _make_batches(sentences_local)

        audition_text, sampling_kwargs = _collect_audition_and_sampling(cache_dict, model)

        # Save all parameters only after the full set has been collected.
        _save_cache(cache_dict)

        # Use refreshed batches in subsequent batch generation.
        global batches
        batches = batches_local

        while True:
            current_instruction = None
            instr_for_filename = 0
            if model_choice == "custom_voice":
                if not instructions:
                    raise ValueError("No instructions available for custom_voice.")
                current_instruction = instructions[instruction_idx]
                instr_for_filename = instruction_idx + 1
                print(f"Current instruction: {instr_for_filename}/{len(instructions)}")

            wavs, sr = _generate_audition_wav(
                audition_text,
                sampling_kwargs,
                instruction=current_instruction,
            )
            suffix = f"_instr{instr_for_filename}" if model_choice == "custom_voice" else ""
            audition_path = f"{basename}_audition{suffix}.wav"
            sf.write(audition_path, wavs[0], sr)

            print(f"Playing audition clip: {audition_path}")
            _sox_play_once(audition_path)

            choice = input(
                "Press Enter to accept, 'r' regenerate, 'm' modify parameters, 'i' next instruction, or 'q' to quit: "
            ).strip().lower()
            if choice == "":
                if model_choice == "custom_voice":
                    return sampling_kwargs, instruction_idx
                return sampling_kwargs, None
            if choice == "r":
                # Regenerate using the same settings.
                continue
            if choice == "m":
                # Modify any/all previously supplied parameters.
                break
            if choice == "q":
                raise SystemExit("User quit.")
            if choice == "i":
                if model_choice != "custom_voice":
                    print("Instruction cycling is only available for custom_voice.")
                    continue
                instruction_idx = (instruction_idx + 1) % len(instructions)
                cache_dict["instruction_index"] = instruction_idx
                _save_cache(cache_dict)
                continue

            print("Unrecognized input. Use Enter / r / m / i / q only.")


sampling_kwargs, selected_instruction_idx = _audition_loop(cache)

selected_instruction = None
selected_instruction_for_filename = 0
if model_choice == "custom_voice":
    if selected_instruction_idx is None:
        raise ValueError("No instruction index selected for custom_voice.")
    selected_instruction = instructions[int(selected_instruction_idx)]
    selected_instruction_for_filename = int(selected_instruction_idx) + 1

for gen_idx in range(1, generation_count + 1):
    for batch_idx, batch_text in enumerate(batches, start=1):
        batch_tokens = _count_tokens(batch_text)
        if model_choice == "custom_voice":
            wavs, sr = model.generate_custom_voice(
                text=batch_text,
                language="English",
                speaker=speaker,
                instruct=selected_instruction,
                **sampling_kwargs,
            )
        else:
            wavs, sr = model.generate_voice_clone(
                text=batch_text,
                language="English",
                voice_clone_prompt=voice_clone_prompt,
                **sampling_kwargs,
            )

        instr_suffix = (
            f"_instr{selected_instruction_for_filename}" if model_choice == "custom_voice" else ""
        )
        out_path = f"{basename}{instr_suffix}_gen{gen_idx}_batch{batch_idx}.wav"
        sf.write(out_path, wavs[0], sr)
        print(f"Generated gen {gen_idx} batch {batch_idx} with {batch_tokens} tokens.")


# or batch generate in one call. this takes lots of memory and will blow out a 8GB.
# wavs, sr = clone_model.generate_voice_clone(
#     text=sentences,
#     language="English",
#     voice_clone_prompt=voice_clone_prompt,
# )
# for i, w in enumerate(wavs):
#     sf.write(f"{basename}_{i}.wav", w, sr)