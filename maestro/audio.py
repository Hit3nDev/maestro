"""
Sound engine: pure NumPy additive synthesis, exported as WAV bytes for Streamlit.
No external audio libraries needed.
"""
import io
import wave
import numpy as np
from .theory import midi_to_freq

SR = 22050  # sample rate


def _envelope(n, a=0.01, d=0.05, s=0.7, r=0.08):
    na, nd, nr = int(SR * a), int(SR * d), int(SR * r)
    if n <= 0:
        return np.zeros(0, dtype=np.float32)
    if na + nd + nr >= n:
        env = np.ones(n)
        f = max(1, n // 10)
        env[:f] = np.linspace(0, 1, f)
        env[-f:] = np.linspace(1, 0, f)
        return env
    ns = n - na - nd - nr
    return np.concatenate([np.linspace(0, 1, na), np.linspace(1, s, nd),
                           np.full(ns, s), np.linspace(s, 0, nr)])


def synth_tone(freq, dur, waveform="piano", volume=0.4):
    n = int(SR * dur)
    t = np.arange(n) / SR
    if waveform == "sine":
        y = np.sin(2 * np.pi * freq * t)
    elif waveform == "square":
        y = np.sign(np.sin(2 * np.pi * freq * t)) * 0.5
    elif waveform == "triangle":
        y = (2 / np.pi) * np.arcsin(np.sin(2 * np.pi * freq * t))
    else:  # "piano": decaying harmonics
        y = np.zeros(n)
        for k, amp in [(1, 1.0), (2, 0.5), (3, 0.25), (4, 0.12), (5, 0.06)]:
            y += amp * np.sin(2 * np.pi * freq * k * t) * np.exp(-1.5 * k * t)
        y /= 1.9
    return (y * _envelope(n) * volume).astype(np.float32)


def render_melody(midi_notes, beats, bpm, waveform="piano", volume=0.4):
    """midi_notes: list of MIDI ints (None = rest). beats: duration of each in beats."""
    beat = 60.0 / bpm
    parts = []
    for m, b in zip(midi_notes, beats):
        dur = b * beat
        if m is None:
            parts.append(np.zeros(int(SR * dur), dtype=np.float32))
        else:
            parts.append(synth_tone(midi_to_freq(m), dur, waveform, volume))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def render_chords(chords, beats, bpm, waveform="piano", volume=0.35):
    """chords: list of lists of MIDI ints. beats: duration of each chord in beats."""
    beat = 60.0 / bpm
    parts = []
    for ch, b in zip(chords, beats):
        dur = b * beat
        if not ch:
            parts.append(np.zeros(int(SR * dur), dtype=np.float32))
            continue
        tones = [synth_tone(midi_to_freq(m), dur, waveform, volume) for m in ch]
        parts.append(np.sum(tones, axis=0) / (len(ch) ** 0.5))
    return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)


def to_wav_bytes(samples):
    samples = np.asarray(samples, dtype=np.float32)
    if samples.size:
        peak = float(np.max(np.abs(samples)))
        if peak > 1.0:
            samples = samples / peak
    pcm = (samples * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.tobytes())
    return buf.getvalue()