## Tweaking parameters

**FROM GITHUB COPILOT**

In most Qwen3‑TTS / “custom voice generation” setups, **top_k, top_p, and temperature** are sampling controls applied to a **text token stream** (LLM side) and/or to **acoustic/codebook tokens** (TTS/vocoder side). A “subtalker” set of variants (often named like `sub_*`, `subtalker_*`, or similar) typically means:

- **Main variants** control sampling for the *primary* generation stream (the one that most directly determines intelligible content / the dominant acoustic trajectory).
- **Subtalker variants** control sampling for a *secondary* stream that contributes **style/prosody/voice identity nuances** (and sometimes background “breathiness,” micro‑timing, expressive variation), while being designed not to destabilize core intelligibility as much.

The exact wiring differs by repo, but the *observable audio consequences* usually follow these patterns.

---

## Mental model: “main” vs “subtalker”
Think of the model producing something like:

1) **What to say / high-level acoustic plan** (main stream)  
2) **How to say it / fine-grained expressive or identity details** (subtalker stream)

So:

- If you push **main** sampling toward randomness (high temperature, low top_p, etc.), you risk **pronunciation errors, skipped words, unstable rhythm**, or even gibberish/unstable acoustic codes.
- If you push **subtalker** sampling toward randomness, you more often get **style drift**, **more/less expressiveness**, **inconsistent timbre**, **quirky prosody**, but typically with **less impact on literal intelligibility**—until you push it too far.

---

## Temperature vs subtalker_temperature

### Main `temperature`
Controls global randomness of token selection in the main stream.

**Audio effects when increased:**
- More variation in pacing/intonation and sometimes phrasing
- Higher risk of artifacts: unstable pitch, odd stress patterns, occasional misreads
- For custom voices: can “wash out” identity because the core stream itself becomes noisy

**When decreased (toward 0):**
- More deterministic delivery
- Often clearer and more stable, but can become flat/robotic
- Identity tends to be consistent but may sound less lively

### `subtalker_temperature`
Controls randomness in the *secondary/style* stream.

**When increased:**
- More expressive micro-variations: subtle pitch wiggles, emphasis changes, breathiness variance
- Can increase perceived “human-ness” or spontaneity
- But can cause **voice identity instability**: the voice may wander in timbre/accent, or sound like it “channel switches” between nearby speakers/styles

**When decreased:**
- The “style layer” becomes consistent
- Typically yields a more uniform, studio-like voiceprint
- Can reduce emotional range and make the custom voice less characterful

**Practical difference:**  
Raising **main temperature** risks *content/prosody coherence*. Raising **subtalker temperature** mainly risks *style/timbre coherence*.

---

## top_p vs subtalker_top_p (nucleus sampling)

### Main `top_p`
Restricts sampling to a probability mass (e.g., 0.9 keeps the smallest set of tokens whose probabilities sum to 0.9).

**Lower top_p (more restrictive):**
- Cleaner, safer, more consistent output
- Less expressive variation; may sound “samey”
- Helps prevent rare-token glitches that can manifest as weird phonemes or acoustic pops (if applied to acoustic tokens)

**Higher top_p (less restrictive):**
- More diversity/expressiveness
- Higher chance of odd pronunciations or unstable acoustic detail, especially on long generations
- Can introduce “creative” but wrong prosody

### `subtalker_top_p`
Same principle, but applied to the subtalker stream.

**Lower subtalker_top_p:**
- Locks style to high-probability choices → more consistent timbre and prosody
- Good for keeping a custom voice “on-model”

**Higher subtalker_top_p:**
- Allows rarer stylistic tokens → more idiosyncratic inflection, more variability
- Risk: style drift, occasional “coloring” that sounds like a different mic/room/voice

**Practical difference:**  
Increasing **main top_p** can destabilize intelligibility sooner; increasing **subtalker top_p** tends to destabilize *identity* sooner.

---

## top_k vs subtalker_top_k

### Main `top_k`
Keeps only the top K most likely tokens.

**Small top_k:**
- Very stable, but can cause repetitive prosody (“stuck in a groove”)
- In some TTS tokenizers/codebooks, too-small top_k can produce *metallic* or *over-quantized* sounding textures because diversity is suppressed

**Large top_k:**
- More variety
- Higher risk of occasional glitches (clicks, roughness, sudden pitch jumps) if low-probability acoustic tokens slip in

### `subtalker_top_k`
Controls diversity of the subtalker stream.

**Small subtalker_top_k:**
- Strong identity consistency, less “acting”
- Sometimes makes the custom voice sound overly controlled / less expressive

**Large subtalker_top_k:**
- More expressive coloration and variation
- Risk of inconsistent “persona” across sentences

---

## What you’ll *hear* when tuning subtalker params
Common audible signatures that point to subtalker sampling being too “hot” (too random):

- **Timbre drift**: same text, same seed, but voice sounds like a slightly different speaker each sentence
- **Accent drift**: vowels shift subtly, consonant sharpness changes
- **Prosody jitter**: micro-pauses and emphasis feel inconsistent
- **Room/mic illusion changes**: voice sounds alternately “closer/farther,” brighter/duller

Whereas main sampling too “hot” tends to cause:

- **Mispronunciations / slurring**
- **Odd word emphasis at syntactic boundaries**
- **Rhythm instability** (speed-ups/slow-downs that don’t match phrasing)
- In worst cases, **token collapse** → noisy/garbled audio

---

## Rules of thumb for custom voice generation
- If your goal is **faithful voice identity**: keep **subtalker_temperature low-ish** and **subtalker_top_p/top_k moderate**.
- If your goal is **expressive acting** while keeping intelligibility: keep **main temperature/top_p conservative**, and increase **subtalker** parameters gradually.
- If you hear **content errors**, reduce **main** randomness first.
- If you hear **identity drift** but words are correct, reduce **subtalker** randomness first.

---

## One clarification to make this 100% accurate for your repo
In `xanomanox/qwen3-tts`, are these “subtalker” knobs applied to:
1) the LLM text generation,  
2) the acoustic/codebook token generation, or  
3) both (two-stage)?

If you point me to the parameter names as they appear in your config/CLI (or paste the inference config section), I can map the explanation directly to the repo’s actual codepath and defaults.


## Anomalies

``` 
Enter audition text [default: Hey look! A cave! I wonder what's inside?]:
do_sample (true/false) [default: True]: False
temperature (float) [default: 1.0]:
top_k (int) [default: 50]:
top_p (0.0 - 1.0) [default: 0.9]:
subtalker_dosample (true/false) [default: True]:
subtalker_temperature (float) [default: 1.0]:
subtalker_top_k (int) [default: 50]:
subtalker_top_p (0.0 - 1.0) [default: 0.9]:
Current instruction: 1/4
The following generation flags are not valid and may be ignored: ['top_p']. Set `TRANSFORMERS_VERBOSITY=info` for more details.
```

## Failure modes

1. set main do_sample to False. generate and audition. sounds generic. set subtalker  do_sample to False. GPU 3D graph maxes out and the generation function never completes.

