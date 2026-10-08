"""Modelleri indirip CTranslate2 int8 biçimine çevirir (yalnızca derleme sırasında, internetle).

Çıktı:  models/whisper-<boyut>/   models/nllb/   models/selftest_{ro,en,tr}.npy
Kullanım: python scripts/prepare_models.py [whisper boyutları, virgüllü]  (varsayılan: large-v3-turbo,small)

Gerekenler (yalnızca bu betik için): ctranslate2 transformers torch sentencepiece soundfile huggingface_hub
"""
import io
import shutil
import subprocess
import sys
import tarfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODELS = ROOT / "models"


def convert(model_id: str, out: Path, copy_files):
    if (out / "model.bin").exists():
        print(f"[var] {out.name}")
        return
    print(f"[çevriliyor] {model_id} -> {out}")
    # Komut adı yerine aynı Python ile çağır: PATH'teki başka bir Python'a bağlanmasın.
    cmd = [sys.executable, "-m", "ctranslate2.converters.transformers",
           "--model", model_id, "--quantization", "int8",
           "--output_dir", str(out), "--copy_files", *copy_files]
    subprocess.run(cmd, check=True)


def selftest_audio(fleurs_lang: str, dst: Path):
    """CI'daki --selftest için gerçek bir konuşma örneği (Google FLEURS, CC-BY 4.0)."""
    if dst.exists():
        return
    import numpy as np
    import soundfile as sf

    url = f"https://huggingface.co/datasets/google/fleurs/resolve/main/data/{fleurs_lang}/audio/dev.tar.gz"
    with urllib.request.urlopen(url) as r, tarfile.open(fileobj=r, mode="r|gz") as t:
        for m in t:
            if m.isfile() and m.name.endswith(".wav"):
                x, sr = sf.read(io.BytesIO(t.extractfile(m).read()), dtype="float32")
                assert sr == 16000, sr
                np.save(dst, x.astype(np.float32))
                print(f"[örnek ses] {m.name} ({len(x) / sr:.1f} sn)")
                return


def main():
    sizes = (sys.argv[1] if len(sys.argv) > 1 else "large-v3-turbo,small").split(",")
    MODELS.mkdir(exist_ok=True)
    for s in sizes:
        s = s.strip()
        convert(f"openai/whisper-{s}", MODELS / f"whisper-{s}",
                ["tokenizer.json", "preprocessor_config.json"])
    convert("facebook/nllb-200-distilled-600M", MODELS / "nllb", ["sentencepiece.bpe.model"])
    # Mac (Apple GPU) için MLX biçiminde turbo. Yalnızca Mac paketine girer (build/app.spec).
    mlx_dir = MODELS / "mlx-whisper-large-v3-turbo"
    if not (mlx_dir / "weights.safetensors").exists():
        from huggingface_hub import snapshot_download
        print("[indiriliyor] mlx-community/whisper-large-v3-turbo")
        snapshot_download("mlx-community/whisper-large-v3-turbo", local_dir=str(mlx_dir),
                          allow_patterns=["config.json", "weights.safetensors"])
    for code, fleurs_lang in (("ro", "ro_ro"), ("en", "en_us"), ("tr", "tr_tr")):
        selftest_audio(fleurs_lang, MODELS / f"selftest_{code}.npy")
    shutil.rmtree(MODELS / ".cache", ignore_errors=True)
    for p in sorted(MODELS.iterdir()):
        size = sum(f.stat().st_size for f in p.rglob("*")) if p.is_dir() else p.stat().st_size
        print(f"  {p.name:28s} {size / 1e6:8.0f} MB")


if __name__ == "__main__":
    main()
