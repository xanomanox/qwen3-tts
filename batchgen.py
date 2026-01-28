import os
import subprocess

import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# create a reference audio in the target style using the VoiceDesign model
design_model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:1",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

num_generations = 100
ref_text = "I absolutely adore your plans." 
basename = "audition"


def _start_playback(path):
    try:
        proc = subprocess.Popen(
            ["/usr/bin/play", "-q", path],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return ("sox", proc)
    except FileNotFoundError:
        print("sox 'play' command not found; skipping playback for", path)
        return None


def _stop_playback(handle):
    if not handle:
        return
    _, obj = handle
    if obj.poll() is None:
        obj.terminate()


def _is_playing(handle):
    if not handle:
        return False
    _, obj = handle
    return obj.poll() is None

sr = None  # sampling rate of the generated audio

for idx in range(num_generations):
    wavs, sr = design_model.generate_voice_design(
        text=ref_text,
        language="English",
        instruct="Speak in pleasant, warm tones with a friendly and engaging demeanor.",
    )
    audio_path = f"{basename}_{idx + 1}.wav"
    sf.write(audio_path, wavs[0], sr)

    # audition the wavfile for the user to accept or reject using sox "play" as the backend.
    # controls: a=accept, r=reject (try next), s=stop playback, q=quit.
    play_handle = None
    accepted = False

    while True:
        if play_handle is None or not _is_playing(play_handle):
            play_handle = _start_playback(audio_path)

        user_choice = input("[a]ccept / [r]eject / [s]top / [q]uit: ").strip().lower()

        if user_choice == "a":
            _stop_playback(play_handle)
            accepted = True
            break
        if user_choice == "r":
            _stop_playback(play_handle)
            break
        if user_choice == "s":
            _stop_playback(play_handle)
            play_handle = None
            continue
        if user_choice == "q":
            _stop_playback(play_handle)
            raise SystemExit("User quit auditioning.")

        print("Unrecognized input. Use a/r/s/q.")

    if accepted:
        print(f"Accepted clip: {audio_path}")
        os.replace(audio_path, "voice_design_reference.wav")  # overwrite if exists
        break
# release the design model from GPU memory
del design_model
torch.cuda.empty_cache()

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

sentences = [
    "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
    "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
]

# reuse it for multiple single calls
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_1.wav", wavs[0], sr)

wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_2.wav", wavs[0], sr)

# or batch generate in one call
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English", "English"],
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(f"clone_batch_{i}.wav", w, sr)