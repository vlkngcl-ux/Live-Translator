"""Canlı Çevirmen — Romence konuşmayı dinler, Türkçe metin olarak gösterir. Offline."""
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
from engine import Engine, available_whisper_models  # noqa: E402

APP_NAME = "Canlı Çevirmen (RO → TR)"


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
        root.geometry("900x620")
        root.minsize(640, 420)
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

        # Alt çubuklar metin alanından ÖNCE yerleştirilir; böylece pencere küçülse de görünür kalırlar.
        bar = ttk.Frame(self.root)
        bar.pack(fill="x", side="bottom")
        self.status = tk.StringVar(value="Hazır. Başlat'a basın.")
        ttk.Label(bar, textvariable=self.status, anchor="w").pack(
            side="left", fill="x", expand=True, padx=8, pady=3)
        self.backlog = tk.StringVar(value="")
        ttk.Label(bar, textvariable=self.backlog, anchor="e").pack(side="right", padx=8)

        opts = ttk.Frame(self.root)
        opts.pack(fill="x", side="bottom", **pad)
        self.show_ro = tk.BooleanVar(value=False)
        self.show_time = tk.BooleanVar(value=True)
        ttk.Checkbutton(opts, text="Romence aslını da göster/kaydet", variable=self.show_ro,
                        command=self._redraw).pack(side="left")
        ttk.Checkbutton(opts, text="Zaman damgası", variable=self.show_time,
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
        self._apply_tag_fonts()

    def _apply_tag_fonts(self):
        self.text.tag_configure("time", foreground="#888888",
                                font=("TkDefaultFont", max(9, self.font_size - 5)))
        self.text.tag_configure("ro", foreground="#7a7a7a",
                                font=("TkDefaultFont", max(10, self.font_size - 3), "italic"))

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
        model_path = dict(self.models)[self.model_var.get()]
        device = self.devices[self.device_cb.current()][0]
        self.busy = True
        for w in (self.start_btn, self.model_cb, self.device_cb):
            w.state(["disabled"])

        def work():
            try:
                self.engine.load(model_path)
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
        for w in (self.model_cb, self.device_cb):
            w.state(["!disabled", "readonly"])
        self.backlog.set("")
        self._draw_level(0)

    # ---------------- olay döngüsü ----------------
    def _poll(self):
        try:
            while True:
                kind, val = self.engine.events.get_nowait()
                if kind == "status":
                    self.status.set(val)
                elif kind == "result":
                    self.results.append(val)
                    self._append(val)
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
        self.text.configure(state="normal")
        if self.show_time.get():
            self.text.insert("end", f"[{export._ts(r.t_start)}]\n", "time")
        self.text.insert("end", r.turkish + "\n")
        if self.show_ro.get():
            self.text.insert("end", r.romanian + "\n", "ro")
        self.text.insert("end", "\n")
        self.text.configure(state="disabled")
        self.text.see("end")

    def _redraw(self):
        self.text.configure(state="normal")
        self.text.delete("1.0", "end")
        self.text.configure(state="disabled")
        for r in self.results:
            self._append(r)

    def _font(self, d):
        self.font_size = max(10, min(32, self.font_size + d))
        self.text.configure(font=("TkDefaultFont", self.font_size))
        self._apply_tag_fonts()

    def clear(self):
        if self.results and not messagebox.askyesno(APP_NAME, "Tüm metin silinsin mi?"):
            return
        self.results.clear()
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
            fn(path, list(self.results), include_romanian=self.show_ro.get(),
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


def selftest(out_path: str, npy_path: str = None) -> int:
    """Paketlenmiş uygulamanın her işletim sisteminde çalıştığını doğrulamak için (CI kullanır).

    Tüm modelleri yükler, örnek Romence sesi (16 kHz float32 .npy) tanır ve çevirir;
    sonucu JSON olarak yazar. Pencere açmaz.
    """
    import json
    import time
    import traceback

    import numpy as np

    from engine import Recognizer, Translator, models_dir

    report = {"ok": False, "models": {}}
    try:
        threads = os.cpu_count() or 2
        tr = Translator(models_dir() / "nllb", threads)
        report["translate_text"] = tr.translate("Bună ziua, ce mai faceți? Astăzi vremea este frumoasă.")
        audio = np.load(npy_path).astype(np.float32) if npy_path else np.zeros(16000 * 3, np.float32)
        for label, path in available_whisper_models():
            t = time.time()
            ro = Recognizer(path, threads).transcribe(audio)
            report["models"][label] = {"romanian": ro, "turkish": tr.translate(ro),
                                       "seconds": round(time.time() - t, 1)}
        report["sounddevice"] = _check_sounddevice()
        report["ok"] = bool(report["translate_text"]) and bool(report["models"])
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
