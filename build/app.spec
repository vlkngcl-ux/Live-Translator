# -*- mode: python ; coding: utf-8 -*-
# PyInstaller tanımı — Windows, macOS (ve test için Linux) için aynı dosya.
# Çalıştırma: pyinstaller build/app.spec --noconfirm
import sys
from pathlib import Path

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules

ROOT = Path(SPECPATH).parent

# Modeller (CI test sesi .npy pakete girmez)
datas = []
for f in (ROOT / "models").rglob("*"):
    if f.is_file() and f.suffix != ".npy":
        datas.append((str(f), str(f.parent.relative_to(ROOT))))
datas += collect_data_files("faster_whisper")          # Silero VAD (.onnx)

binaries = collect_dynamic_libs("ctranslate2")          # ctranslate2 / OpenMP kütüphaneleri
hiddenimports = ["engine", "export"] + collect_submodules("av")

a = Analysis(
    [str(ROOT / "app" / "main.py")],
    pathex=[str(ROOT / "app")],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["torch", "transformers", "matplotlib", "scipy", "pandas", "IPython",
              "PIL", "pytest", "notebook", "tensorflow",
              "hf_xet"],  # yalnızca internetten indirme için; offline uygulamada gereksiz
    noarchive=False,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="CanliCevirmen",
    console=False,          # siyah komut penceresi açılmasın
    upx=False,
    codesign_identity=None, # macOS: ad-hoc imza (Apple hesabı olmadan)
)
coll = COLLECT(exe, a.binaries, a.datas, name="CanliCevirmen", upx=False)

if sys.platform == "darwin":
    app = BUNDLE(
        coll,
        name="CanliCevirmen.app",
        bundle_identifier="com.canlicevirmen.app",
        info_plist={
            "CFBundleDisplayName": "Canlı Çevirmen",
            "CFBundleName": "Canlı Çevirmen",
            "CFBundleShortVersionString": "1.0.0",
            "LSMinimumSystemVersion": "11.0",
            "NSHighResolutionCapable": True,
            # Bu anahtar olmadan macOS mikrofon izni istemez ve ses sessizce boş gelir.
            "NSMicrophoneUsageDescription":
                "Romence konuşmayı dinleyip Türkçeye çevirmek için mikrofon gerekir. "
                "Ses cihazınızda işlenir, hiçbir yere gönderilmez.",
        },
    )
