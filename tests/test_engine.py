"""Uçtan uca test: Konuşma kayıtlarını tek bir 'ortam kaydı' gibi birleştirip motora akıtır.

Kullanım: python tests/test_engine.py <wav_klasörü> [model_anahtarı] [kaynak_dil] [hedef_dil] [gürültü_std]
Mikrofon gerektirmez. Ağ olmadan çalışmalıdır (ör. `unshare -rn` altında).
"""
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "app"))
from engine import SAMPLE_RATE, Engine, models_dir  # noqa: E402


def build_ambient(wav_dir: Path, noise_std: float = 0.0003):
    rng = np.random.default_rng(0)
    parts, bounds, t = [], [], 0.0
    for f in sorted(wav_dir.glob("*.wav")):
        x, sr = sf.read(f, dtype="float32")
        assert sr == SAMPLE_RATE, sr
        gap = rng.uniform(1.0, 2.0)
        parts.append(np.zeros(int(gap * sr), np.float32))
        t += gap
        parts.append(x)
        bounds.append((round(t, 1), round(t + len(x) / sr, 1), f.name))
        t += len(x) / sr
    parts.append(np.zeros(int(1.5 * SAMPLE_RATE), np.float32))
    audio = np.concatenate(parts)
    audio += rng.normal(0, noise_std, len(audio)).astype(np.float32)  # oda gürültüsü
    return audio, bounds


def main():
    wav_dir = Path(sys.argv[1])
    key = sys.argv[2] if len(sys.argv) > 2 else "large-v3-turbo"
    src = sys.argv[3] if len(sys.argv) > 3 else "ro"
    tgt = sys.argv[4] if len(sys.argv) > 4 else "tr"
    noise = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0003
    realtime = len(sys.argv) > 6 and sys.argv[6] == "rt"   # mikrofon gibi gerçek hızda akıt
    audio, bounds = build_ambient(wav_dir, noise)
    print(f"Toplam kayıt: {len(audio) / SAMPLE_RATE:.1f} sn, {len(bounds)} konuşma")
    for b in bounds:
        print("  gerçek konuşma:", b)

    eng = Engine()
    eng.set_languages(src, tgt)
    t0 = time.time()
    eng.load(models_dir() / f"whisper-{key}")
    print(f"Model yükleme: {time.time() - t0:.1f} sn")
    t0 = time.time()
    eng.start_from_array(audio, SAMPLE_RATE, realtime=realtime)
    n = 0
    while True:
        kind, val = eng.events.get()
        if kind == "result":
            n += 1
            print(f"\n[{val.t_start:.1f}-{val.t_end:.1f}s] gecikme={val.latency:.1f}s\n  {val.src.upper()}: {val.source}"
                  f"\n  {val.tgt.upper()}: {val.target}")
        elif kind == "error":
            print("HATA:", val)
        elif kind == "done":
            break
    print(f"\n{n} sonuç, işlem süresi {time.time() - t0:.1f} sn, elenen: {eng.stats}")


if __name__ == "__main__":
    main()
