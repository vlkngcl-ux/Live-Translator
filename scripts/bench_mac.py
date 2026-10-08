"""Apple Silicon'da Whisper motorlarını karşılaştırır: hız (parça başına sn) ve doğruluk (WER).

Motorlar:
  faster-whisper (mevcut, yalnızca CPU)  |  whisper.cpp / pywhispercpp (Metal GPU)  |  mlx-whisper (Apple GPU)
Kayıtlar: Google FLEURS Türkçe test kümesinden ilk N kayıt (CC-BY 4.0), doğru metinleriyle.

Kullanım: python scripts/bench_mac.py [N]
Yalnızca ölçüm içindir; uygulama paketine girmez.
"""
import csv
import io
import json
import os
import platform
import re
import subprocess
import sys
import tarfile
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
LANG = "tr"
RESULTS = []


def load_clips():
    base = "https://huggingface.co/datasets/google/fleurs/resolve/main/data/tr_tr"
    ref = {}
    tsv = urllib.request.urlopen(base + "/test.tsv").read().decode("utf-8")
    for row in csv.reader(io.StringIO(tsv), delimiter="\t", quoting=csv.QUOTE_NONE):
        ref[row[1]] = row[2]
    clips = []
    with urllib.request.urlopen(base + "/audio/test.tar.gz") as r, tarfile.open(fileobj=r, mode="r|gz") as t:
        for m in t:
            if m.isfile() and m.name.endswith(".wav"):
                x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
                assert sr == 16000
                clips.append((x, ref[os.path.basename(m.name)]))
                if len(clips) >= N:
                    break
    return clips


def norm(s):
    return re.sub(r"[^\w\s']", " ", s.lower().replace("’", "'")).split()


def wer(ref, hyp):
    d = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, d[0] = d[0], i
        for j, h in enumerate(hyp, 1):
            prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (r != h))
    return d[len(hyp)] / max(1, len(ref))


def run(name, load_fn, transcribe_fn, clips):
    print(f"\n### {name}", flush=True)
    try:
        t = time.time()
        model = load_fn()
        load_s = time.time() - t
        transcribe_fn(model, clips[0][0])            # ısınma (ilk çağrı derleme/önbellek içerir)
        times, refs, hyps = [], [], []
        for x, ref in clips:
            t = time.time()
            hyp = transcribe_fn(model, x)
            times.append(time.time() - t)
            refs += norm(ref)
            hyps += norm(hyp)
            print(f"  {len(x) / 16000:4.1f} sn ses -> {times[-1]:5.2f} sn | {hyp[:90]}", flush=True)
        row = {"motor": name, "yukleme_sn": round(load_s, 1),
               "ort_sn": round(float(np.mean(times)), 2), "en_kotu_sn": round(float(np.max(times)), 2),
               "wer": round(100 * wer(refs, hyps), 1)}
        print("  SONUÇ", json.dumps(row, ensure_ascii=False), flush=True)
        RESULTS.append(row)
    except Exception:
        print("  HATA:\n" + traceback.format_exc(), flush=True)
        RESULTS.append({"motor": name, "hata": traceback.format_exc().splitlines()[-1]})


# ---------------- faster-whisper (mevcut) ----------------
def fw(size):
    path = ROOT / "models" / f"whisper-{size}"

    def load():
        from faster_whisper import WhisperModel
        if not (path / "model.bin").exists():
            raise FileNotFoundError(path)
        return WhisperModel(str(path), device="cpu", compute_type="int8", cpu_threads=os.cpu_count())

    def tr(m, x):
        segs, _ = m.transcribe(x, language=LANG, beam_size=1, vad_filter=True,
                               condition_on_previous_text=False)
        return " ".join(s.text.strip() for s in segs)
    return load, tr


# ---------------- whisper.cpp (Metal) ----------------
def wcpp(fname, audio_ctx=0):
    def load():
        from huggingface_hub import hf_hub_download
        from pywhispercpp.model import Model
        p = hf_hub_download("ggerganov/whisper.cpp", fname)
        m = Model(p, n_threads=min(8, os.cpu_count() or 4), print_progress=False,
                  print_realtime=False, redirect_whispercpp_logs_to=None)
        try:
            print("  system_info:", Model.system_info(), flush=True)
        except Exception as e:
            print("  system_info alınamadı:", e)
        return m

    def tr(m, x):
        kw = dict(language=LANG, no_context=True, single_segment=False)
        if audio_ctx:
            # Ses kısa olduğunda kodlayıcı bağlamını küçült (30 sn yerine sesin gerçek uzunluğu).
            kw["audio_ctx"] = int(min(1500, len(x) / 16000 * 50 + 64))
        segs = m.transcribe(x, **kw)
        return " ".join(s.text.strip() for s in segs)
    return load, tr


# ---------------- MLX ----------------
def mlx(repo):
    def load():
        import mlx.core as mx
        print("  mlx cihazı:", mx.default_device(), "| metal:", mx.metal.is_available(), flush=True)
        return repo

    def tr(repo_, x):
        import mlx_whisper
        out = mlx_whisper.transcribe(x, path_or_hf_repo=repo_, language=LANG,
                                     condition_on_previous_text=False, verbose=None)
        return out["text"].strip()
    return load, tr


def main():
    info = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string", "hw.ncpu", "hw.memsize"],
                          capture_output=True, text=True).stdout.split("\n")
    print("Makine:", platform.platform(), "|", " | ".join(i for i in info if i), flush=True)
    clips = load_clips()
    print(f"{len(clips)} Türkçe kayıt, toplam {sum(len(x) for x, _ in clips) / 16000:.0f} sn", flush=True)

    run("faster-whisper turbo int8 (CPU, mevcut)", *fw("large-v3-turbo"), clips)
    run("faster-whisper small int8 (CPU, mevcut)", *fw("small"), clips)
    run("whisper.cpp turbo q5_0", *wcpp("ggml-large-v3-turbo-q5_0.bin"), clips)
    run("whisper.cpp turbo q5_0 + audio_ctx", *wcpp("ggml-large-v3-turbo-q5_0.bin", audio_ctx=1), clips)
    run("whisper.cpp turbo q8_0", *wcpp("ggml-large-v3-turbo-q8_0.bin"), clips)
    run("mlx turbo", *mlx("mlx-community/whisper-large-v3-turbo"), clips)

    print("\n================ ÖZET ================")
    for r in RESULTS:
        print(json.dumps(r, ensure_ascii=False))


if __name__ == "__main__":
    main()
