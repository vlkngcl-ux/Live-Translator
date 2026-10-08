"""Canlı Çevirmen — konuşmayı dinler, seçilen dile çevirip metin olarak gösterir. Offline.

Diller: Romence, İngilizce, Türkçe (her yönde).
"""
import os
import queue
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Pencere modunda paketlenen uygulamada stdout/stderr None olur; yazmaya çalışan
# kütüphaneler çökmesin diye boş bir hedefe yönlendir.
if sys.stdout is None or sys.stderr is None:
    _null = open(os.devnull, "w", encoding="utf-8")
    sys.stdout = sys.stdout or _null
    sys.stderr = sys.stderr or _null

import export  # noqa: E402
from engine import LANGUAGES, Engine, available_whisper_models, lang_name  # noqa: E402

APP_NAME = "Canlı Çevirmen"
LANG_CODES = list(LANGUAGES)                     # ["ro", "en", "tr"]
LANG_NAMES = [lang_name(c) for c in LANG_CODES]  # ["Romence", "İngilizce", "Türkçe"]


class App:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.engine = Engine()
        self.results = []
        self.running = False
        self.busy = False          # model yükleniyor / kuyruk boşaltılıyor
        self.font_size = 15
        self.devices = []

        root.title(APP_NAME)
        root.geometry("920x640")
        root.minsize(680, 440)
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.models = available_whisper_models()
        self._build_ui()
        self._load_devices()
        if not self.models:
            messagebox.showerror(APP_NAME, "Model dosyaları bulunamadı. Uygulamayı yeniden kurun.")
            self.start_btn.state(["disabled"])
        root.after(100, self._poll)

    # ---------------- arayüz ----------------
    def _build_ui(self):
        pad = {"padx": 6, "pady": 4}
        top = ttk.Frame(self.root)
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Mikrofon:").pack(side="left")
        self.device_var = tk.StringVar()
        self.device_cb = ttk.Combobox(top, textvariable=self.device_var, state="readonly", width=34)
        self.device_cb.pack(side="left", padx=(2, 10))

        ttk.Label(top, text="Mod:").pack(side="left")
        self.model_var = tk.StringVar(value=self.models[0][0] if self.models else "")
        self.model_cb = ttk.Combobox(top, textvariable=self.model_var, state="readonly", width=16,
                                     values=[m[0] for m in self.models])
        self.model_cb.pack(side="left", padx=(2, 10))

        self.start_btn = ttk.Button(top, text="▶ Başlat", command=self.toggle)
        self.start_btn.pack(side="left")

        self.level = tk.Canvas(top, width=90, height=14, highlightthickness=1,
                               highlightbackground="#999")
        self.level.pack(side="left", padx=10)
        self.level_bar = self.level.create_rectangle(0, 0, 0, 14, fill="#3a9d5d", width=0)

        # Dil seçimi
        langs = ttk.Frame(self.root)
        langs.pack(fill="x", **pad)
        ttk.Label(langs, text="Konuşulan dil:").pack(side="left")
        self.src_var = tk.StringVar(value=lang_name("ro"))
        self.src_cb = ttk.Combobox(langs, textvariable=self.src_var, state="readonly",
                                   width=12, values=LANG_NAMES)
        self.src_cb.pack(side="left", padx=(2, 6))
        self.swap_btn = ttk.Button(langs, text="⇄", width=3, command=self._swap)
        self.swap_btn.pack(side="left")
        ttk.Label(langs, text="Çeviri dili:").pack(side="left", padx=(6, 0))
        self.tgt_var = tk.StringVar(value=lang_name("tr"))
        self.tgt_cb = ttk.Combobox(langs, textvariable=self.tgt_var, state="readonly",
                                   width=12, values=LANG_NAMES)
        self.tgt_cb.pack(side="left", padx=(2, 6))
        self._prev = {"src": self.src_var.get(), "tgt": self.tgt_var.get()}
        self.src_cb.bind("<<ComboboxSelected>>", lambda e: self._lang_changed("src"))
        self.tgt_cb.bind("<<ComboboxSelected>>", lambda e: self._lang_changed("tgt"))

        # Alt çubuklar metin alanından ÖNCE yerleştirilir; böylece pencere küçülse de görünür kalırlar.
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", side="bottom")
        self.status = tk.StringVar(value="Hazır. Dilleri seçip Başlat'a basın.")
        ttk.Label(bar, textvariable=self.status, anchor="w").pack(
            side="left", fill="x", expand=True, padx=8, pady=3)
        self.backlog = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.backlog, anchor="e").pack(side="right", padx=8)
        # Son cümlenin gecikmesi: konuşma bittikten çevirinin ekrana gelmesine kadar geçen süre
        self.latency_var = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.latency_var, anchor="e").pack(side="right", padx=8)

        opts = ttk.Frame(self.root)
        opts.pack(fill="x", side="bottom", **pad)
        self.show_src = tk.BooleanVar(value=False)
        self.show_time = tk.BooleanVar(value=True)
        self.show_interim = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Konuşulan metni de göster/kaydet", variable=self.show_src,
                        command=self._redraw).pack(side="left")
        ttk.Checkbutton(opts, text="Saat", variable=self.show_time,
                        command=self._redraw).pack(side="left", padx=(10, 0))
        ttk.Checkbutton(opts, text="Ön çeviri", variable=self.show_interim,
                        command=self._redraw).pack(side="left", padx=10)
        ttk.Button(opts, text="A−", width=3, command=lambda: self._font(-1)).pack(side="left")
        ttk.Button(opts, text="A+", width=3, command=lambda: self._font(+1)).pack(side="left", padx=(2, 0))

        ttk.Button(opts, text="Temizle", command=self.clear).pack(side="right")
        ttk.Button(opts, text="DOCX kaydet", command=lambda: self.save("docx")).pack(side="right", padx=4)
        ttk.Button(opts, text="TXT kaydet", command=lambda: self.save("txt")).pack(side="right")

        mid = ttk.Frame(self.root)
        mid.pack(fill="both", expand=True, **pad)
        self.text = tk.Text(mid, wrap="word", font=("TkDefaultFont", self.font_size),
                            padx=10, pady=8, undo=False, height=8)
        sb = ttk.Scrollbar(mid, command=self.text.yview)
        self.text.configure(yscrollcommand=sb.set, state="disabled")
        self.text.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")
        # Ön çeviri bölgesi her zaman metnin sonunda, "istart" işaretinden sonra durur.
        # Kesin sonuçlar bu işaretin ÖNÜNE eklenir; böylece ön çeviri hep en altta kalır.
        self.text.mark_set("istart", "end-1c")
        self.text.mark_gravity("istart", "right")
        self.interim = None          # gösterilen ön çeviri (Result) veya None
        self._apply_tag_fonts()

    def _apply_tag_fonts(self):
        self.text.tag_configure("time", foreground="#888888",
                                font=("TkDefaultFont", max(9, self.font_size - 5)))
        self.text.tag_configure("src", foreground="#7a7a7a",
                                font=("TkDefaultFont", max(10, self.font_size - 3), "italic"))
        self.text.tag_configure("interim", foreground="#8a8a8a",
                                font=("TkDefaultFont", self.font_size, "italic"))

    # ---------------- dil seçimi ----------------
    def _code(self, name: str) -> str:
        return LANG_CODES[LANG_NAMES.index(name)]

    def _lang_changed(self, which: str):
        """İki tarafta aynı dil seçilirse diğer tarafı önceki seçimle değiştir (yer değiştirme)."""
        src, tgt = self.src_var.get(), self.tgt_var.get()
        if src == tgt:
            if which == "src":
                self.tgt_var.set(self._prev["src"])
            else:
                self.src_var.set(self._prev["tgt"])
        self._prev = {"src": self.src_var.get(), "tgt": self.tgt_var.get()}

    def _swap(self):
        s, t = self.src_var.get(), self.tgt_var.get()
        self.src_var.set(t)
        self.tgt_var.set(s)
        self._prev = {"src": t, "tgt": s}

    def _load_devices(self):
        try:
            devs, default = Engine.input_devices()
        except Exception as e:
            self.status.set(f"Ses aygıtları okunamadı: {e}")
            return
        self.devices = devs
        names = [name for _, name in devs]
        self.device_cb["values"] = names
        if names:
            idx = next((k for k, (i, _) in enumerate(devs) if i == default), 0)
            self.device_cb.current(idx)
        else:
            self.status.set("Mikrofon bulunamadı.")

    # ---------------- başlat / durdur ----------------
    def _controls(self):
        return (self.model_cb, self.device_cb, self.src_cb, self.tgt_cb)

    def toggle(self):
        if self.busy:
            return
        if self.running:
            self.engine.stop()
            self.running = False
            self.busy = True
            self.start_btn.configure(text="Bitiriliyor…")
            self.start_btn.state(["disabled"])
            self.status.set("Kalan konuşma işleniyor…")
        else:
            self._start()

    def _start(self):
        if not self.devices:
            messagebox.showerror(APP_NAME, "Kullanılabilir mikrofon yok.")
            return
        try:
            self.engine.set_languages(self._code(self.src_var.get()), self._code(self.tgt_var.get()))
        except ValueError as e:
            messagebox.showerror(APP_NAME, str(e))
            return
        model_path = dict(self.models)[self.model_var.get()]
        device = self.devices[self.device_cb.current()][0]
        self.busy = True
        self.start_btn.state(["disabled"])
        self.swap_btn.state(["disabled"])
        for w in self._controls():
            w.state(["disabled"])

        def work():
            try:
                self.engine.load(model_path, interim=self.show_interim.get())
                self.engine.start(device)
                self.engine.events.put(("started", None))
            except Exception as e:
                self.engine.events.put(("start_failed", str(e)))

        threading.Thread(target=work, daemon=True).start()

    def _set_idle(self):
        self.running = False
        self.busy = False
        self.start_btn.configure(text="▶ Başlat")
        self.start_btn.state(["!disabled"])
        self.swap_btn.state(["!disabled"])
        for w in self._controls():
            w.state(["!disabled", "readonly"])
        self.backlog.set("")
        self._draw_level(0)
        if self.interim is not None:     # oturum bitti; asılı kalan ön çeviri olmasın
            self.interim = None
            self._render_interim()
        st = self.engine.stats
        if st.get("vad_skipped") or st.get("halluc_dropped"):
            self.status.set(f"Durduruldu. (Konuşma içermeyen {st['vad_skipped']} ses parçası ve "
                            f"{st['halluc_dropped']} olası uydurma metin elendi.)")

    # ---------------- olay döngüsü ----------------
    def _poll(self):
        try:
            while True:
                kind, val = self.engine.events.get_nowait()
                if kind == "status":
                    if val == "Dinleniyor…":
                        val = (f"Dinleniyor… ({lang_name(self.engine.src)} → "
                               f"{lang_name(self.engine.tgt)})")
                    self.status.set(val)
                elif kind == "result":
                    self.results.append(val)
                    if self.interim is not None and self.interim.rid <= val.rid:
                        self.interim = None              # ön çeviri yerini kesin çeviriye bırakır
                    self._append(val)
                    self._render_interim()
                    if val.latency:
                        self.latency_var.set(f"Gecikme: {val.latency:.1f} sn".replace(".", ","))
                elif kind == "interim":
                    self.interim = val
                    self._render_interim()
                elif kind == "discard":
                    if self.interim is not None and self.interim.rid <= val:
                        self.interim = None
                        self._render_interim()
                elif kind == "error":
                    self.status.set(val)
                elif kind == "level":
                    self._draw_level(val)
                elif kind == "backlog":
                    self.backlog.set(f"Sırada bekleyen: {val}" if val else "")
                elif kind == "started":
                    self.running = True
                    self.busy = False
                    self.start_btn.configure(text="■ Durdur")
                    self.start_btn.state(["!disabled"])
                elif kind == "start_failed":
                    self._set_idle()
                    hint = ""
                    if sys.platform == "darwin":
                        hint = ("\n\nmacOS: Sistem Ayarları → Gizlilik ve Güvenlik → Mikrofon "
                                "bölümünden bu uygulamaya izin verin.")
                    messagebox.showerror(APP_NAME, f"Başlatılamadı: {val}{hint}")
                    self.status.set("Başlatılamadı.")
                elif kind == "done":
                    self._set_idle()
        except queue.Empty:
            pass
        self.root.after(100, self._poll)

    def _draw_level(self, rms):
        self.level.coords(self.level_bar, 0, 0, min(90, int(rms * 900)), 14)

    # ---------------- metin alanı ----------------
    def _append(self, r):
        """Kesin sonucu ön çeviri bölgesinin ('istart') ÖNÜNE ekler."""
        self.text.configure(state="normal")
        if self.show_time.get():
            self.text.insert("istart", f"[{export.clock(r)}]\n", "time")
        self.text.insert("istart", r.target + "\n")
        if self.show_src.get():
            self.text.insert("istart", r.source + "\n", "src")
        self.text.insert("istart", "\n")
        self.text.configure(state="disabled")
        self.text.see("end")

    def _render_interim(self):
        """En alttaki ön çeviri bölgesini yeniden çizer (yoksa boşaltır)."""
        self.text.configure(state="normal")
        self.text.delete("istart", "end-1c")
        r = self.interim if self.show_interim.get() else None
        if r is not None:
            pos = self.text.index("istart")
            label = f"[{export.clock(r)}] " if self.show_time.get() else ""
            self.text.insert("istart", f"{label}… {r.target}\n", "interim")
            if self.show_src.get():
                self.text.insert("istart", r.source + "\n", "src")
            self.text.mark_set("istart", pos)           # işaret ön çevirinin BAŞINDA kalsın
        self.text.configure(state="disabled")
        self.text.see("end")

    def _redraw(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.mark_set("istart", "end-1c")
        self.text.configure(state="disabled")
        for r in self.results:
            self._append(r)
        self._render_interim()

    def _font(self, d):
        self.font_size = max(10, min(32, self.font_size + d))
        self.text.configure(font=("TkDefaultFont", self.font_size))
        self._apply_tag_fonts()

    def clear(self):
        if self.results and not messagebox.askyesno(APP_NAME, "Tüm metin silinsin mi?"):
            return
        self.results.clear()
        self.interim = None
        self._redraw()

    def save(self, kind):
        if not self.results:
            messagebox.showinfo(APP_NAME, "Kaydedilecek metin yok.")
            return
        ext = ".docx" if kind == "docx" else ".txt"
        types = [("Word belgesi", "*.docx")] if kind == "docx" else [("Metin dosyası", "*.txt")]
        path = filedialog.asksaveasfilename(defaultextension=ext, filetypes=types,
                                            initialfile="ceviri" + ext)
        if not path:
            return
        fn = export.save_docx if kind == "docx" else export.save_txt
        try:
            fn(path, list(self.results), include_source=self.show_src.get(),
               include_time=self.show_time.get())
            self.status.set(f"Kaydedildi: {path}")
        except Exception as e:
            messagebox.showerror(APP_NAME, f"Kaydedilemedi: {e}")

    def on_close(self):
        if self.results and not messagebox.askyesno(
                APP_NAME, "Çıkılsın mı? Kaydedilmemiş metin kaybolur."):
            return
        try:
            self.engine.stop()
        finally:
            self.root.destroy()


# Selftest'te her konuşma dili için denenecek çeviri yönleri
SELFTEST_PAIRS = {"ro": ["tr", "en"], "en": ["ro", "tr"], "tr": ["en"]}


def selftest(out_path: str, sample: str = None) -> int:
    """Paketlenmiş uygulamanın her işletim sisteminde çalıştığını doğrulamak için (CI kullanır).

    `sample`: 16 kHz float32 .npy dosyası (Romence kabul edilir) veya içinde
    selftest_<dil>.npy dosyaları olan bir klasör. Her örnek, ilgili dilde tanınır ve
    SELFTEST_PAIRS'teki her hedefe çevrilir. Sonuç JSON olarak yazılır; pencere açmaz.
    """
    import json
    import time
    import traceback

    import numpy as np

    from engine import Recognizer, Translator, models_dir

    report = {"ok": False, "models": {}}
    try:
        samples = {}
        if sample and os.path.isdir(sample):
            for code in LANG_CODES:
                p = os.path.join(sample, f"selftest_{code}.npy")
                if os.path.exists(p):
                    samples[code] = np.load(p).astype(np.float32)
        elif sample:
            samples["ro"] = np.load(sample).astype(np.float32)
        else:
            samples["ro"] = np.zeros(16000 * 3, np.float32)

        threads = os.cpu_count() or 2
        tr = Translator(models_dir() / "nllb", threads)
        report["translate_text"] = tr.translate(
            "Bună ziua, ce mai faceți? Astăzi vremea este frumoasă.", "ro", "tr")
        all_ok = bool(report["translate_text"])
        for label, path in available_whisper_models():
            rec = Recognizer(path, threads)
            rows = []
            for src, audio in samples.items():
                t = time.time()
                text = rec.transcribe(audio, language=src)
                for tgt in SELFTEST_PAIRS[src]:
                    out = tr.translate(text, src, tgt)
                    rows.append({"pair": f"{src}→{tgt}", "source": text, "target": out,
                                 "seconds": round(time.time() - t, 1)})
                    all_ok = all_ok and bool(text) and bool(out)
            report["models"][label] = rows
        report["sounddevice"] = _check_sounddevice()
        report["ok"] = all_ok and bool(report["models"])
    except Exception:
        report["error"] = traceback.format_exc()
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return 0 if report["ok"] else 1


def _check_sounddevice():
    try:
        devs, _ = Engine.input_devices()
        return f"ok, {len(devs)} giriş aygıtı"
    except Exception as e:  # CI makinelerinde ses kartı olmayabilir; kütüphanenin yüklenmesi yeter
        return f"yüklendi, aygıt sorgusu: {e}"


def main():
    if "--selftest" in sys.argv:
        args = sys.argv[sys.argv.index("--selftest") + 1:]
        sys.exit(selftest(args[0], args[1] if len(args) > 1 else None))
    root = tk.Tk()
    if sys.platform.startswith("linux"):
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
