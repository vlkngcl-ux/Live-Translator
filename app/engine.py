"""Ses yakalama, konuşma bölme, konuşma tanıma (Whisper) ve çeviri (NLLB).

Diller: Romence, İngilizce, Türkçe (her yönde).

Tamamen offline çalışır: modeller uygulama klasöründen yüklenir, ağa çıkılmaz.
"""
import os

# Hugging Face kütüphanelerinin internete çıkmasını kesin olarak engelle.
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import collections
import queue
import re
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

# Konuşma bölme ayarları
FRAME_SEC = 0.03            # enerji ölçüm penceresi
MIN_CHUNK_SEC = 2.0         # bu süreden kısa parçalar, sessizlik olsa bile bekletilir (v1.1.0: 4.0)
MAX_CHUNK_SEC = 15.0        # bu süreye ulaşan parça zorla gönderilir (v1.1.0: 20.0; Whisper sınırı 30 sn)
SILENCE_END_SEC = 0.6       # konuşmadan sonra bu kadar sessizlik -> parçayı kapat (v1.1.0: 0.7)
MIN_VAD_SPEECH_SEC = 0.5    # Silero VAD'in parçada bulduğu konuşma bundan azsa Whisper'a hiç gönderilmez
MIN_SPEECH_SEC = 0.3        # toplam konuşma bundan azsa parça atılır (gürültüyü Whisper VAD ayrıca eler)
PREROLL_SEC = 0.3           # konuşma başlamadan önceki bu kadar ses de parçaya eklenir
NOISE_WINDOW_SEC = 15.0     # gürültü tabanı bu pencerenin alt NOISE_PERCENTILE'ından hesaplanır
NOISE_PERCENTILE = 5        # (aralıksız konuşmada bile kelime arası sessizlikler bu dilime düşer)
SPEECH_RMS_FLOOR = 0.0008   # mutlak alt eşik (eskiden 0.006 idi; kısık kayıtları kaçırıyordu)


def resource_dir() -> Path:
    """Paketlenmiş uygulamada ve geliştirmede model klasörünün yeri."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS", Path(sys.executable).parent))
    return Path(__file__).resolve().parent.parent


def models_dir() -> Path:
    return resource_dir() / "models"


def available_whisper_models():
    """Pakette bulunan Whisper modelleri: [(etiket, klasör)]."""
    labels = {
        "large-v3-turbo": "Doğru (turbo)",
        "medium": "Dengeli (medium)",
        "small": "Hızlı (small)",
    }
    found = []
    for key, label in labels.items():
        p = models_dir() / f"whisper-{key}"
        if (p / "model.bin").exists():
            found.append((label, p))
    return found


# Desteklenen diller: kod -> (Türkçe ad, kısaltma, NLLB kodu). Whisper ISO kodunu kullanır.
LANGUAGES = {
    "ro": ("Romence", "RO", "ron_Latn"),
    "en": ("İngilizce", "EN", "eng_Latn"),
    "tr": ("Türkçe", "TR", "tur_Latn"),
}


def lang_name(code: str) -> str:
    return LANGUAGES[code][0]


def lang_short(code: str) -> str:
    return LANGUAGES[code][1]


@dataclass
class Result:
    t_start: float      # kayıt başlangıcına göre saniye
    t_end: float
    source: str         # konuşulan dildeki metin
    target: str         # çeviri
    src: str = "ro"     # dil kodları
    tgt: str = "tr"
    latency: float = 0.0  # parça kapandıktan çeviri hazır olana kadar geçen süre (sn)


# --------------------------------------------------------------------------------------
# Whisper "uydurma" (halüsinasyon) filtresi
#
# Whisper, konuşma olmayan seste (nefes, hışırtı, mikrofonun yükselttiği oda gürültüsü)
# eğitim verisindeki video altyazılarından ezberlediği kapanış cümlelerini yazabilir:
# "teşekkürler", "görüşürüz", "abone olun / takip edin"... ve bunları tekrar tekrar.
# Kullanıcının gerçek ekran görüntüsünde görülen örnekler (sessiz oda, Romence):
#   "Vă mulțumesc!" / "Aștepți, să ne vedem! Să ne vedem! Să ne vedem!"
#   "Nu am încărți-vă, în urmărți-vă, în urmărți-vă."
# --------------------------------------------------------------------------------------
HALLUCINATION_PATTERNS = {
    "ro": [r"mul[țţt]umesc", r"ne vedem", r"pe cur[aâ]nd", r"la revedere", r"vizionare",
           r"abon[aă]", r"urm[aă]r", r"subtitr", r"pa pa"],
    "en": [r"thank(s| you)", r"for watching", r"subscribe", r"see you", r"\bbye\b",
           r"like and", r"subtitles?"],
    "tr": [r"te[şs]ekk[üu]r", r"izledi[ğg]iniz", r"abone ol", r"altyaz[ıi]", r"g[öo]r[üu][şs][üu]r[üu]z",
           r"g[öo]r[üu][şs]mek [üu]zere", r"ho[şs][çc]a kal", r"takip ed"],
}
SHORT_UTTERANCE_WORDS = 8      # kalıp filtresi yalnızca bu kadar kısa sözlere uygulanır
HALLUC_MAX_VAD_SPEECH = 1.5    # ...ve VAD'in bulduğu konuşma bundan azsa
HALLUC_MAX_LOGPROB = -0.6      # ...ya da Whisper'ın güveni bundan düşükse


def _norm(s: str) -> str:
    return re.sub(r"[^\w\s]", "", s.lower()).strip()


def collapse_repeats(text: str) -> str:
    """Art arda tekrarlanan cümle/öbekleri tek kopyaya indirir.

    "Să ne vedem! Să ne vedem! Să ne vedem!" -> "Să ne vedem!"
    "în urmărți-vă, în urmărți-vă" -> "în urmărți-vă"
    """
    out = []
    for sent in re.findall(r"[^.!?]+[.!?]*", text):
        parts, kept = [p for p in sent.split(",")], []
        for p in parts:
            if kept and _norm(p) and _norm(p) == _norm(kept[-1]):
                continue
            kept.append(p)
        sent = ",".join(kept)
        if out and _norm(sent) and _norm(sent) == _norm(out[-1]):
            continue
        out.append(sent)
    return re.sub(r"\s+", " ", "".join(out)).strip()


def clean_hallucinations(text: str, lang: str, vad_speech_sec: float, avg_logprob: float) -> str:
    """Tekrarları siler; kısa ve yalnızca bilinen kalıplardan oluşan, düşük kanıtlı sözü atar.

    Bilinçli ödünleşim: gerçekten tek başına söylenmiş kısa bir "Mulțumesc." da, VAD az konuşma
    bulduysa veya Whisper emin değilse atılabilir. Uzun cümlelerin içindeki "teşekkür" vb. etkilenmez.
    """
    text = collapse_repeats(text)
    if not text:
        return ""
    weak_evidence = vad_speech_sec < HALLUC_MAX_VAD_SPEECH or avg_logprob < HALLUC_MAX_LOGPROB
    pats = [re.compile(p, re.IGNORECASE) for p in HALLUCINATION_PATTERNS.get(lang, [])]
    if weak_evidence and len(_norm(text).split()) <= SHORT_UTTERANCE_WORDS and pats:
        # 2 harften kısa parçalar ("M.K." gibi kısaltmaların harfleri) cümle sayılmaz
        sentences = [s for s in re.findall(r"[^.!?]+", text) if len(_norm(s)) > 2]
        if sentences and all(any(p.search(s) for p in pats) for s in sentences):
            return ""
    return text


class Translator:
    """NLLB-200 (CTranslate2, int8) ile diller arası çeviri (RO / EN / TR, her yönde).

    transformers/torch kullanmamak için tokenizasyon doğrudan sentencepiece ile yapılır.
    """

    def __init__(self, model_path: Path, threads: int):
        import ctranslate2
        import sentencepiece as spm

        self.model = ctranslate2.Translator(
            str(model_path), device="cpu", compute_type="int8",
            inter_threads=1, intra_threads=threads,
        )
        self.sp = spm.SentencePieceProcessor(
            model_file=str(model_path / "sentencepiece.bpe.model"))

    def translate(self, text: str, src: str = "ro", tgt: str = "tr") -> str:
        text = text.strip()
        if not text or src == tgt:
            return text
        s, t = LANGUAGES[src][2], LANGUAGES[tgt][2]
        tokens = [s] + self.sp.encode(text, out_type=str) + ["</s>"]
        res = self.model.translate_batch(
            [tokens], target_prefix=[[t]], beam_size=2,
            max_decoding_length=400, repetition_penalty=1.1,
        )
        out = [tok for tok in res[0].hypotheses[0] if tok not in (t, "</s>")]
        return self.sp.decode(out).strip()


class Recognizer:
    """faster-whisper ile konuşma tanıma (dil sabitlenir; otomatik algılama yok)."""

    def __init__(self, model_path: Path, threads: int):
        from faster_whisper import WhisperModel

        self.model = WhisperModel(
            str(model_path), device="cpu", compute_type="int8",
            cpu_threads=threads, local_files_only=True,
        )

    @staticmethod
    def speech_seconds(audio: np.ndarray) -> float:
        """Silero VAD'e göre parçadaki toplam konuşma süresi (sn). Whisper'dan çok daha ucuzdur."""
        from faster_whisper.vad import VadOptions, get_speech_timestamps

        ts = get_speech_timestamps(audio, VadOptions(
            threshold=0.5, min_speech_duration_ms=250, min_silence_duration_ms=300, speech_pad_ms=100))
        return sum(t["end"] - t["start"] for t in ts) / SAMPLE_RATE

    def transcribe(self, audio: np.ndarray, language: str = "ro") -> str:
        return self.transcribe_ex(audio, language)[0]

    def transcribe_ex(self, audio: np.ndarray, language: str = "ro"):
        """(metin, ortalama avg_logprob) döndürür. Güveni düşük / tekrarlı segmentler atılır."""
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            beam_size=1,                           # anlık kullanım için hız; kalite farkı testte küçüktü
            vad_filter=True,                       # gürültü/sessizlikte uydurmayı azaltır
            vad_parameters={"min_silence_duration_ms": 400},
            condition_on_previous_text=False,      # tekrar döngülerini engeller
            no_speech_threshold=0.6,
            temperature=[0.0, 0.2, 0.4],
        )
        parts, logprobs = [], []
        for s in segments:
            # v1.1.0'da yalnızca (no_speech > 0.8 VE logprob < -1.0) atılıyordu; sessiz odada
            # uydurmalar bu süzgeçten geçti. Artık koşullardan HERHANGİ biri yeter.
            if s.no_speech_prob > 0.6 or s.avg_logprob < -1.0 or s.compression_ratio > 2.4:
                continue
            parts.append(s.text.strip())
            logprobs.append(s.avg_logprob)
        text = " ".join(p for p in parts if p)
        return text, (float(np.mean(logprobs)) if logprobs else -10.0)


def _resample(x: np.ndarray, sr: int) -> np.ndarray:
    if sr == SAMPLE_RATE:
        return x.astype(np.float32)
    # Basit alçak geçiren (hareketli ortalama) + doğrusal ara değer. Konuşma için yeterli.
    k = max(1, int(round(sr / SAMPLE_RATE)))
    if k > 1:
        x = np.convolve(x, np.ones(k, dtype=np.float32) / k, mode="same")
    n_out = int(len(x) * SAMPLE_RATE / sr)
    xp = np.linspace(0, len(x) - 1, n_out)
    return np.interp(xp, np.arange(len(x)), x).astype(np.float32)


class Segmenter:
    """Gelen sesi sessizliklere göre cümle benzeri parçalara böler."""

    def __init__(self):
        self.frame = int(FRAME_SEC * SAMPLE_RATE)
        self.buf = np.zeros(0, dtype=np.float32)      # henüz çerçevelenmemiş ses
        self.chunk = []                                # mevcut parçanın çerçeveleri
        self.chunk_start = 0.0
        self.speech_frames = 0
        self.silence_run = 0
        self.t = 0.0                                   # işlenen toplam süre (sn)
        self.noise = 0.001                             # uyarlanabilir gürültü tabanı (hızla uyum sağlar)
        self.level = 0.0                               # arayüz için son ses seviyesi
        self.preroll = collections.deque(maxlen=int(PREROLL_SEC / FRAME_SEC))
        self.hist = collections.deque(maxlen=int(NOISE_WINDOW_SEC / FRAME_SEC))
        self._n = 0

    def _is_speech(self, rms: float, in_chunk: bool) -> bool:
        # Gürültü tabanı = son NOISE_WINDOW_SEC'teki çerçevelerin alt yüzdeliği. Kelime ve cümle
        # aralarındaki sessizlikler bu dilime düştüğü için konuşma gürültü tahminini yukarı çekmez.
        self.hist.append(rms)
        self._n += 1
        if self._n % 10 == 0 or len(self.hist) < 10:
            self.noise = max(float(np.percentile(self.hist, NOISE_PERCENTILE)), 1e-4)
        # Histerezis: konuşmayı BAŞLATMAK için yüksek eşik, cümlenin İÇİNDE devam etmek için
        # düşük eşik — kısık heceler cümleyi erken kesmesin.
        # Alt sınır bilerek çok düşük: kısık mikrofon / uzaktaki konuşmacı da yakalansın.
        # Yanlışlıkla yakalanan gürültüyü Whisper'ın VAD filtresi zaten eler.
        if in_chunk:
            return rms > max(self.noise * 1.8, SPEECH_RMS_FLOOR * 0.75)
        return rms > max(self.noise * 3.0, SPEECH_RMS_FLOOR)

    def push(self, samples: np.ndarray):
        """Ses ekle; tamamlanan parçaları (başlangıç, bitiş, ses) olarak döndür."""
        out = []
        self.buf = np.concatenate([self.buf, samples])
        while len(self.buf) >= self.frame:
            f, self.buf = self.buf[: self.frame], self.buf[self.frame:]
            rms = float(np.sqrt(np.mean(f * f)))
            self.level = rms
            speech = self._is_speech(rms, in_chunk=bool(self.chunk))
            if not self.chunk:
                if speech:
                    # Ön tampon: eşiği aşmadan hemen önceki sesi de ekle (ilk hece kaybolmasın).
                    self.chunk_start = max(0.0, self.t - len(self.preroll) * FRAME_SEC)
                    self.chunk = list(self.preroll) + [f]
                    self.preroll.clear()
                    self.speech_frames = 1
                    self.silence_run = 0
                else:
                    self.preroll.append(f)
            else:
                self.chunk.append(f)
                if speech:
                    self.speech_frames += 1
                    self.silence_run = 0
                else:
                    self.silence_run += 1
                dur = len(self.chunk) * FRAME_SEC
                ended = (self.silence_run * FRAME_SEC >= SILENCE_END_SEC and dur >= MIN_CHUNK_SEC)
                if ended or dur >= MAX_CHUNK_SEC:
                    out.extend(self._close())
            self.t += FRAME_SEC
        return out

    def _close(self):
        chunk, self.chunk = self.chunk, []
        if self.speech_frames * FRAME_SEC < MIN_SPEECH_SEC:
            return []
        audio = np.concatenate(chunk)
        return [(self.chunk_start, self.chunk_start + len(audio) / SAMPLE_RATE, audio)]

    def flush(self):
        return self._close() if self.chunk else []


class Engine:
    """Arka planda çalışan kayıt + tanıma + çeviri hattı.

    Sonuçlar ve durum mesajları `events` kuyruğuna konur; arayüz buradan okur:
      ("status", str) | ("result", Result) | ("error", str) | ("level", float)
      | ("backlog", int) | ("done", None)
    """

    def __init__(self):
        self.events: "queue.Queue" = queue.Queue()
        self._audio_q: "queue.Queue" = queue.Queue()
        self._chunk_q: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._stream = None
        self._threads = []
        self.recognizer = None
        self.translator = None
        self._loaded_whisper = None
        self.src, self.tgt = "ro", "tr"   # konuşulan dil, çeviri dili
        self.stats = {"vad_skipped": 0, "halluc_dropped": 0}   # tanılama sayaçları
        # Tanıma ve çeviri aynı işçide sırayla çalışır; ikisi de tüm çekirdekleri kullanabilir.
        self.threads = max(1, os.cpu_count() or 2)

    # ---- model yükleme ----
    def load(self, whisper_path: Path):
        if self._loaded_whisper != whisper_path:
            self.events.put(("status", "Konuşma tanıma modeli yükleniyor…"))
            self.recognizer = Recognizer(whisper_path, self.threads)
            self._loaded_whisper = whisper_path
        if self.translator is None:
            self.events.put(("status", "Çeviri modeli yükleniyor…"))
            self.translator = Translator(models_dir() / "nllb", self.threads)

    # ---- mikrofon ----
    @staticmethod
    def input_devices():
        import sounddevice as sd
        devs = []
        for i, d in enumerate(sd.query_devices()):
            if d["max_input_channels"] > 0:
                devs.append((i, d["name"]))
        try:
            default = sd.default.device[0]
        except Exception:
            default = None
        return devs, default

    def set_languages(self, src: str, tgt: str):
        if src not in LANGUAGES or tgt not in LANGUAGES:
            raise ValueError(f"Desteklenmeyen dil: {src} → {tgt}")
        if src == tgt:
            raise ValueError("Konuşulan dil ile çeviri dili aynı olamaz.")
        self.src, self.tgt = src, tgt

    def start(self, device=None):
        import sounddevice as sd

        self._stop.clear()
        self._run(self._open_mic(sd, device))

    def _open_mic(self, sd, device):
        def callback(indata, frames, time_info, status):
            self._audio_q.put(indata[:, 0].copy() if indata.ndim > 1 else indata.copy())

        # Önce doğrudan 16 kHz dene; olmazsa cihazın varsayılan hızıyla açıp dönüştür.
        try:
            stream = sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype="float32",
                                    device=device, callback=callback, blocksize=0)
            sr = SAMPLE_RATE
        except Exception:
            info = sd.query_devices(device if device is not None else sd.default.device[0])
            sr = int(info["default_samplerate"])
            stream = sd.InputStream(samplerate=sr, channels=1, dtype="float32",
                                    device=device, callback=callback, blocksize=0)
        self._stream = stream
        stream.start()
        return sr

    def start_from_array(self, audio: np.ndarray, sr: int, realtime: bool = False):
        """Test için: mikrofon yerine hazır bir ses dizisini akış gibi besler."""
        self._stop.clear()

        def feeder():
            step = int(sr * 0.1)
            for i in range(0, len(audio), step):
                if self._stop.is_set():
                    break
                self._audio_q.put(audio[i:i + step].astype(np.float32))
                if realtime:
                    time.sleep(0.1)
            self._stop.set()

        threading.Thread(target=feeder, daemon=True).start()
        self._run(sr)

    def _run(self, sr: int):
        seg = Segmenter()
        self.stats = {"vad_skipped": 0, "halluc_dropped": 0}

        def segment_loop():
            last_level = 0.0
            while True:
                try:
                    x = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                for c in seg.push(_resample(x, sr)):
                    self._chunk_q.put((*c, time.monotonic()))   # kapanış anı: gecikme ölçümü için
                    self.events.put(("backlog", self._chunk_q.qsize()))
                now = time.monotonic()
                if now - last_level > 0.1:
                    self.events.put(("level", seg.level))
                    last_level = now
            for c in seg.flush():
                self._chunk_q.put((*c, time.monotonic()))
            self._chunk_q.put(None)  # işçiye bitiş sinyali

        self._threads = [
            threading.Thread(target=segment_loop, daemon=True),
            threading.Thread(target=self._worker, daemon=True),
        ]
        for t in self._threads:
            t.start()
        self.events.put(("status", "Dinleniyor…"))

    def _worker(self):
        while True:
            item = self._chunk_q.get()
            if item is None:
                self.events.put(("status", "Durduruldu."))
                self.events.put(("done", None))
                return
            t0, t1, audio, closed_at = item
            self.events.put(("backlog", self._chunk_q.qsize()))
            src, tgt = self.src, self.tgt   # oturum boyunca sabit (arayüz kayıtta seçimi kilitler)
            try:
                # 1) Ucuz kapı: içinde gerçek konuşma yoksa Whisper'a hiç gönderme.
                #    Hem uydurmayı hem işlemci yükünü (ve dolayısıyla gecikmeyi) azaltır.
                speech = self.recognizer.speech_seconds(audio)
                if speech < MIN_VAD_SPEECH_SEC:
                    self.stats["vad_skipped"] += 1
                    continue
                # 2) Tanıma + uydurma filtresi
                text, logprob = self.recognizer.transcribe_ex(audio, language=src)
                cleaned = clean_hallucinations(text, src, speech, logprob)
                if not cleaned:
                    if text:
                        self.stats["halluc_dropped"] += 1
                    continue
                out = self.translator.translate(cleaned, src, tgt)
                self.events.put(("result", Result(t0, t1, cleaned, out, src, tgt,
                                                  latency=time.monotonic() - closed_at)))
            except Exception as e:  # bir parçadaki hata tüm oturumu düşürmesin
                self.events.put(("error", f"İşleme hatası: {e}"))

    def stop(self):
        """Kaydı durdurur; kuyruktaki parçalar işlenmeye devam eder ('done' gelene kadar)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._stop.set()
