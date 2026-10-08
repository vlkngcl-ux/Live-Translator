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
# v1.2.0 değerleri ölçümle seçildi (cümle içinde doğal duraksamalı Türkçe kayıtta kelime hata oranı):
#   0.6 sn / 2 sn (v1.1.1): %20.2 (30 parça)  ->  0.8 sn / 3 sn: %12.9 (17 parça)
# Kısa sessizlik eşiği cümleleri ortasından bölüyordu; parçalar bağlamsız kalınca hem tanıma hem
# çeviri bozuluyordu. Bekleme süresinin yarattığı gecikmeyi ön çeviri (interim) telafi eder.
MIN_CHUNK_SEC = 3.0         # bu süreden kısa parçalar, sessizlik olsa bile bekletilir
MAX_CHUNK_SEC = 15.0        # bu süreye ulaşan parça zorla gönderilir (Whisper sınırı 30 sn)
SILENCE_END_SEC = 0.8       # konuşmadan sonra bu kadar sessizlik -> parçayı kapat
# Dile göre arama genişliği: Romencede (Whisper'ın az veriyle eğitildiği dil) beam 3 hata oranını
# %26.3 -> %22.7 indirdi; Türkçede iyileştirmedi (%12.9 -> %15.7). Süreye etkisi ölçülemeyecek kadar az.
BEAM_BY_LANG = {"ro": 3}
MIN_VAD_SPEECH_SEC = 0.5    # Silero VAD'in parçada bulduğu konuşma bundan azsa Whisper'a hiç gönderilmez
# Whisper segment süzgeci: herhangi biri aşılırsa segment atılır
SEG_MAX_NO_SPEECH = 0.6     # "bu segmentte konuşma yok" olasılığı
SEG_MIN_LOGPROB = -1.0      # ortalama güven (log olasılık)
SEG_MAX_COMPRESSION = 2.4   # metin tekrarı göstergesi (gzip sıkıştırma oranı)
PARALLEL_INTERIM_MIN_CPUS = 6   # bu kadar çekirdek varsa ön çeviri ayrı iş parçacığında çalışır
INTERIM_MIN_SEC = 1.5       # devam eden cümle en az bu uzunluktaysa ön çeviri yapılır
INTERIM_EVERY_SEC = 1.5     # ön çeviri en fazla bu sıklıkta güncellenir
INTERIM_GAP_SEC = 1.0       # bir ön çeviri bittikten sonra bir sonrakine kadar en az bu kadar bekle
INTERIM_STOP_ON_SILENCE_SEC = 0.2   # bu kadar sessizlik başladıysa yeni ön çeviri başlatma
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


MLX_TURBO_DIR = "mlx-whisper-large-v3-turbo"


def mlx_status():
    """(kullanılabilir_mi, açıklama). Yalnızca Apple Silicon Mac'te ve MLX yüklenebiliyorsa True."""
    if sys.platform != "darwin":
        return False, "macOS değil"
    if not (models_dir() / MLX_TURBO_DIR / "weights.safetensors").exists():
        return False, "MLX modeli pakette yok"
    try:
        import mlx.core as mx
        if not mx.metal.is_available():
            return False, "Metal GPU bulunamadı"
        import mlx_whisper  # noqa: F401
        return True, f"MLX {getattr(mx, '__version__', '?')}, {mx.default_device()}"
    except Exception as e:  # ör. macOS 14'ten eski sürüm: kütüphane yüklenemez
        return False, f"MLX yüklenemedi: {e}"


def available_whisper_models():
    """Kullanılabilir Whisper modelleri: [(etiket, klasör)]. İlk eleman varsayılandır.

    Mac'te MLX (Apple GPU) kullanılabiliyorsa turbo onunla çalışır: GitHub'ın Apple M1 Mac'inde
    ölçüm, cümle başına 9.3 sn (CPU) -> 1.8 sn (GPU), aynı doğrulukta.
    """
    found = []
    if mlx_status()[0]:
        found.append(("Doğru (turbo · GPU)", models_dir() / MLX_TURBO_DIR))
    labels = {
        "large-v3-turbo": "Doğru (turbo)",
        "medium": "Dengeli (medium)",
        "small": "Hızlı (small)",
    }
    for key, label in labels.items():
        p = models_dir() / f"whisper-{key}"
        if (p / "model.bin").exists():
            found.append((label, p))
    return found


def is_mlx_path(path) -> bool:
    return Path(path).name.startswith("mlx-")


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
    wall: float = 0.0     # konuşmanın başladığı sistem saati (time.time(), epoch sn)
    rid: int = 0          # parça kimliği (ön çeviri ile kesin çeviriyi eşleştirmek için)
    interim: bool = False # True: hızlı ön çeviri (sonradan kesin çeviriyle değiştirilir)


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

    def translate(self, text: str, src: str = "ro", tgt: str = "tr", beam_size: int = 2) -> str:
        text = text.strip()
        if not text or src == tgt:
            return text
        s, t = LANGUAGES[src][2], LANGUAGES[tgt][2]
        tokens = [s] + self.sp.encode(text, out_type=str) + ["</s>"]
        res = self.model.translate_batch(
            [tokens], target_prefix=[[t]], beam_size=beam_size,
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

    def transcribe_ex(self, audio: np.ndarray, language: str = "ro", prompt: str = None,
                      beam_size: int = 1):
        """(metin, ortalama avg_logprob) döndürür. Güveni düşük / tekrarlı segmentler atılır.

        prompt: önceki cümle(ler); Whisper'a bağlam olarak verilir (cümle ortasından bölünen
        konuşmada kelime ve yazım tutarlılığını artırır).
        """
        segments, _info = self.model.transcribe(
            audio,
            language=language,
            task="transcribe",
            initial_prompt=prompt or None,
            beam_size=beam_size,
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
            if (s.no_speech_prob > SEG_MAX_NO_SPEECH or s.avg_logprob < SEG_MIN_LOGPROB
                    or s.compression_ratio > SEG_MAX_COMPRESSION):
                continue
            parts.append(s.text.strip())
            logprobs.append(s.avg_logprob)
        text = " ".join(p for p in parts if p)
        return text, (float(np.mean(logprobs)) if logprobs else -10.0)


class MLXRecognizer:
    """Apple Silicon GPU'sunda (MLX) konuşma tanıma. Recognizer ile aynı arayüz.

    Tüm MLX işlemleri TEK ve kalıcı bir iş parçacığında yapılır: uygulama oturumlar arasında
    farklı iş parçacıkları kullandığı için, GPU kaynaklarının iş parçacıkları arasında
    paylaşılmasından doğabilecek sorunlar böylece baştan önlenir.
    Not: mlx-whisper ışın araması (beam search) desteklemez; her dilde açgözlü (greedy) çözümleme.
    """

    def __init__(self, model_path: Path, threads: int = 0):
        from concurrent.futures import ThreadPoolExecutor

        self.path = str(model_path)
        self._ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._ex.submit(self._warmup).result()     # modeli yükle + GPU çekirdeklerini hazırla

    @staticmethod
    def speech_seconds(audio: np.ndarray) -> float:
        return Recognizer.speech_seconds(audio)    # VAD her iki motorda da aynı (CPU, çok ucuz)

    def _warmup(self):
        import mlx_whisper
        mlx_whisper.transcribe(np.zeros(SAMPLE_RATE, np.float32), path_or_hf_repo=self.path,
                               language="en", verbose=None)

    def transcribe(self, audio: np.ndarray, language: str = "ro") -> str:
        return self.transcribe_ex(audio, language)[0]

    def transcribe_ex(self, audio: np.ndarray, language: str = "ro", prompt: str = None,
                      beam_size: int = 1):
        return self._ex.submit(self._transcribe, audio, language, prompt).result()

    def _transcribe(self, audio, language, prompt):
        import mlx_whisper
        out = mlx_whisper.transcribe(
            np.asarray(audio, np.float32), path_or_hf_repo=self.path, language=language,
            task="transcribe", initial_prompt=prompt or None, verbose=None,
            condition_on_previous_text=False, temperature=(0.0, 0.2, 0.4),
            no_speech_threshold=0.6, compression_ratio_threshold=2.4, logprob_threshold=-1.0)
        parts, logprobs = [], []
        for s in out.get("segments", []):
            # faster-whisper yolundakiyle AYNI segment süzgeci
            if (s.get("no_speech_prob", 0) > SEG_MAX_NO_SPEECH or s.get("avg_logprob", 0) < SEG_MIN_LOGPROB
                    or s.get("compression_ratio", 0) > SEG_MAX_COMPRESSION):
                continue
            t = s.get("text", "").strip()
            if t:
                parts.append(t)
                logprobs.append(s.get("avg_logprob", 0.0))
        return " ".join(parts), (float(np.mean(logprobs)) if logprobs else -10.0)


def make_recognizer(model_path, threads: int):
    """Klasör adına göre doğru tanıyıcıyı oluşturur (mlx-* -> GPU, diğerleri -> CPU)."""
    if is_mlx_path(model_path):
        return MLXRecognizer(model_path, threads)
    return Recognizer(model_path, threads)


class StreamResampler:
    """Mikrofon hızını (ör. 48 / 44.1 kHz) 16 kHz'e çeviren, DURUM TUTAN dönüştürücü.

    v1.1.1'e kadar her küçük mikrofon bloğu ayrı ayrı dönüştürülüyordu: blok sınırlarında
    süreksizlik oluşuyor ve her blokta kesirli örnekler atılıyordu. Bu sınıf filtre geçmişini
    ve kesirli konumu bloklar arasında taşır; çıktı tek seferde dönüştürülmüş sesle aynıdır.
    """

    def __init__(self, sr_in: int, sr_out: int = SAMPLE_RATE, taps: int = 63):
        self.passthrough = sr_in == sr_out
        self.ratio = sr_in / sr_out                       # çıkış başına giriş örneği
        fc = 0.45 * sr_out / sr_in                        # kesim (giriş örneği başına devir)
        n = np.arange(taps) - (taps - 1) / 2
        h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
        self.h = (h / h.sum()).astype(np.float32)         # alçak geçiren (örtüşme önleyici)
        self.hist = np.zeros(taps - 1, np.float32)        # filtre geçmişi
        self.prev = 0.0                                   # önceki bloğun son süzülmüş örneği
        # sonraki çıkışın konumu (ext koordinatı). y[k] ≈ x[k - (taps-1)/2] olduğundan filtre
        # gecikmesi kadar ileriden başla: çıkış girişle aynı zaman çizgisinde kalır.
        self.pos = 1.0 + (taps - 1) / 2

    def process(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, np.float32)
        if self.passthrough or len(x) == 0:
            return x
        buf = np.concatenate([self.hist, x])
        y = np.convolve(buf, self.h, mode="valid")        # len(y) == len(x)
        self.hist = buf[-(len(self.h) - 1):]
        ext = np.concatenate([[self.prev], y])             # ext[0] = önceki son örnek
        last = len(ext) - 1
        if self.pos > last:
            out = np.zeros(0, np.float32)
        else:
            pts = np.arange(self.pos, last + 1e-9, self.ratio)
            out = np.interp(pts, np.arange(len(ext)), ext).astype(np.float32)
            self.pos = pts[-1] + self.ratio
        self.pos -= last                                   # yeni bloğun ext koordinatına kaydır
        self.prev = float(y[-1])
        return out


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
        self.chunk_id = 0                              # her yeni parça için artan kimlik

    def current(self):
        """Devam eden (henüz bitmemiş) parça: (kimlik, başlangıç sn, süre sn, ses) ya da None."""
        if not self.chunk:
            return None
        return (self.chunk_id, self.chunk_start, len(self.chunk) * FRAME_SEC, np.concatenate(self.chunk))

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
                    self.chunk_id += 1
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
        return [(self.chunk_start, self.chunk_start + len(audio) / SAMPLE_RATE, audio, self.chunk_id)]

    def flush(self):
        return self._close() if self.chunk else []


class Engine:
    """Arka planda çalışan kayıt + tanıma + çeviri hattı.

    Sonuçlar ve durum mesajları `events` kuyruğuna konur; arayüz buradan okur:
      ("status", str) | ("result", Result) | ("error", str) | ("level", float)
      | ("backlog", int) | ("done", None)
      | ("interim", Result)  -> devam eden cümlenin hızlı ön çevirisi (aynı rid'li kesin sonuçla değişir)
      | ("discard", rid)     -> bu parçadan kesin sonuç çıkmadı; varsa ön çevirisi silinmeli
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
        # Önceki metni Whisper'a bağlam olarak vermek ölçümde ZARARLI çıktı (%14.0 -> %18.0;
        # önceki cümleyi tekrar ediyordu). Varsayılan kapalı.
        self.use_context = False
        self.final_beam = None     # None: BEAM_BY_LANG / 1
        self._context = ""         # son tanınan metin (bağlam için)
        self.interim_recognizer = None   # ön çeviri için hızlı model (load() ayarlar)
        self._small_cache = None         # yüklenmiş 'small' modeli (oturumlar arasında saklanır)
        self.gpu = False                 # seçili model MLX (Apple GPU) mi
        # Tanıma ve çeviri aynı işçide sırayla çalışır; ikisi de tüm çekirdekleri kullanabilir.
        self.threads = max(1, os.cpu_count() or 2)
        # Yeterli çekirdek varsa ön çeviri AYRI bir iş parçacığında, kendi 2 çekirdeğiyle çalışır;
        # böylece kesin çeviriyi bekletmez. Az çekirdekte (ör. 2) sırayla çalışmak daha iyidir:
        # 2 çekirdekli test makinesinde paralel çalışma kesin çevirileri yavaşlatıyordu.
        self.parallel_interim = self.threads >= PARALLEL_INTERIM_MIN_CPUS
        self.interim_threads = 2 if self.parallel_interim else self.threads
        self.final_threads = max(1, self.threads - 2) if self.parallel_interim else self.threads

    # ---- model yükleme ----
    def load(self, whisper_path: Path, interim: bool = True):
        whisper_path = Path(whisper_path)
        self.gpu = is_mlx_path(whisper_path)
        if self._loaded_whisper != whisper_path:
            self.events.put(("status", "Konuşma tanıma modeli yükleniyor…"))
            self.recognizer = make_recognizer(
                whisper_path, self.threads if self.gpu else self.final_threads)
            self._loaded_whisper = whisper_path
        # Ön çeviri modeli:
        #  - GPU (MLX) seçiliyse aynı model: kesin çeviri ~2 sn sürdüğü için ikinci modele gerek yok
        #  - 'small' seçiliyse aynı model
        #  - aksi halde 'small' (bir kez yüklenir, oturumlar arasında saklanır)
        small = models_dir() / "whisper-small"
        if not interim:
            self.interim_recognizer = None
        elif self.gpu or whisper_path == small:
            self.interim_recognizer = self.recognizer
        elif (small / "model.bin").exists():
            if self._small_cache is None:
                self.events.put(("status", "Ön çeviri modeli yükleniyor…"))
                self._small_cache = Recognizer(small, self.interim_threads)
            self.interim_recognizer = self._small_cache
        else:
            self.interim_recognizer = None
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

    def start_from_array(self, audio: np.ndarray, sr: int, realtime: bool = False, block: int = 0):
        """Test için: mikrofon yerine hazır bir ses dizisini akış gibi besler.

        block: örnek sayısı olarak blok boyutu (0 = 0.1 sn). Gerçek mikrofon blokları küçüktür
        (ör. 48 kHz'de 512 örnek); dönüştürme hatalarını yakalamak için bunu taklit edin.
        """
        self._stop.clear()

        def feeder():
            step = block or int(sr * 0.1)
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
        resampler = StreamResampler(sr)
        self.stats = {"vad_skipped": 0, "halluc_dropped": 0, "interim": 0}
        self._context = ""
        self._wall0 = time.time()           # akış zamanı 0 = bu sistem saati
        self._interim_slot = None           # en son ön çeviri isteği (yalnızca en yenisi tutulur)
        self._interim_lock = threading.Lock()
        self._last_final_id = 0
        self._last_interim_done = 0.0

        def segment_loop():
            last_level = 0.0
            last_interim_t = -1e9
            while True:
                try:
                    x = self._audio_q.get(timeout=0.2)
                except queue.Empty:
                    if self._stop.is_set():
                        break
                    continue
                for c in seg.push(resampler.process(x)):
                    self._chunk_q.put((*c, time.monotonic()))   # kapanış anı: gecikme ölçümü için
                    self.events.put(("backlog", self._chunk_q.qsize()))
                # Devam eden uzun cümle için ön çeviri isteği (işçi boştaysa yapılır)
                cur = seg.current() if self.interim_recognizer is not None else None
                # Konuşmacı durmaya başladıysa yeni ön çeviri başlatma: cümle birazdan kapanacak ve
                # sürmekte olan bir ön çeviri kesin çeviriyi bekletirdi (Mac ölçümü: 4.7 sn bekleme).
                speaking = seg.silence_run * FRAME_SEC < INTERIM_STOP_ON_SILENCE_SEC
                if (cur and speaking and cur[2] >= INTERIM_MIN_SEC
                        and seg.t - last_interim_t >= INTERIM_EVERY_SEC):
                    with self._interim_lock:
                        self._interim_slot = cur
                    last_interim_t = seg.t
                now = time.monotonic()
                if now - last_level > 0.1:
                    self.events.put(("level", seg.level))
                    last_level = now
            for c in seg.flush():
                self._chunk_q.put((*c, time.monotonic()))
            self._chunk_q.put(None)  # işçiye bitiş sinyali

        self._worker_done = threading.Event()
        self._threads = [
            threading.Thread(target=segment_loop, daemon=True),
            threading.Thread(target=self._worker, daemon=True),
        ]
        # Paralel ön çeviri yalnızca CPU'da ve ayrı bir model varken; GPU'da (tek MLX iş parçacığı)
        # ve aynı model paylaşılırken sırayla çalışılır.
        self._parallel_now = (self.parallel_interim and not self.gpu
                              and self.interim_recognizer is not None
                              and self.interim_recognizer is not self.recognizer)
        if self._parallel_now:
            self._threads.append(threading.Thread(target=self._interim_loop, daemon=True))
        for t in self._threads:
            t.start()
        self.events.put(("status", "Dinleniyor…"))

    def _interim_loop(self):
        """Paralel mod: ön çeviriler kesin çevirilerden bağımsız olarak üretilir."""
        while not self._worker_done.is_set():
            iv = self._take_interim()
            if iv:
                self._do_interim(iv)
            else:
                time.sleep(0.05)

    def _take_interim(self):
        # Ön çeviriler arasında boşluk bırak: işçi sürekli meşgul olmasın, cümle kapandığında
        # kesin çeviri boşta bir işçi bulsun.
        if time.monotonic() - self._last_interim_done < INTERIM_GAP_SEC:
            return None
        with self._interim_lock:
            item, self._interim_slot = self._interim_slot, None
        if item and item[0] > self._last_final_id:   # kesin çevirisi çıkmış parçanın önizlemesi gereksiz
            return item
        return None

    def _worker(self):
        while True:
            try:
                # Kesin çeviri her zaman önceliklidir; kuyruk boşsa ön çeviri yapılır.
                item = self._chunk_q.get(timeout=0.05)
            except queue.Empty:
                if not self._parallel_now:         # sıralı mod: boşta kalınca ön çeviri yap
                    iv = self._take_interim()
                    if iv:
                        self._do_interim(iv)
                continue
            if item is None:
                self._worker_done.set()
                self.events.put(("status", "Durduruldu."))
                self.events.put(("done", None))
                return
            t0, t1, audio, cid, closed_at = item
            self.events.put(("backlog", self._chunk_q.qsize()))
            self._last_final_id = cid
            src, tgt = self.src, self.tgt   # oturum boyunca sabit (arayüz kayıtta seçimi kilitler)
            try:
                tm = {"bekleme": round(time.monotonic() - closed_at, 2), "ses_sn": round(len(audio) / SAMPLE_RATE, 1)}
                # 1) Ucuz kapı: içinde gerçek konuşma yoksa Whisper'a hiç gönderme.
                #    Hem uydurmayı hem işlemci yükünü (ve dolayısıyla gecikmeyi) azaltır.
                _t = time.monotonic()
                speech = self.recognizer.speech_seconds(audio)
                tm["vad"] = round(time.monotonic() - _t, 2)
                if speech < MIN_VAD_SPEECH_SEC:
                    self.stats["vad_skipped"] += 1
                    self.events.put(("discard", cid))        # varsa ön çeviriyi kaldır
                    continue
                # 2) Tanıma (önceki metin bağlam olarak) + uydurma filtresi
                prompt = self._context[-200:] if self.use_context else None
                beam = self.final_beam or BEAM_BY_LANG.get(src, 1)
                _t = time.monotonic()
                text, logprob = self.recognizer.transcribe_ex(
                    audio, language=src, prompt=prompt, beam_size=beam)
                tm["tanima"] = round(time.monotonic() - _t, 2)
                cleaned = clean_hallucinations(text, src, speech, logprob)
                if not cleaned:
                    if text:
                        self.stats["halluc_dropped"] += 1
                    self.events.put(("discard", cid))
                    continue
                self._context = (self._context + " " + cleaned).strip()[-400:]
                _t = time.monotonic()
                out = self.translator.translate(cleaned, src, tgt)
                tm["ceviri"] = round(time.monotonic() - _t, 2)
                self.stats.setdefault("sureler", []).append(tm)   # tanılama: adım adım süreler
                self.events.put(("result", Result(
                    t0, t1, cleaned, out, src, tgt, latency=time.monotonic() - closed_at,
                    wall=self._wall0 + t0, rid=cid)))
            except Exception as e:  # bir parçadaki hata tüm oturumu düşürmesin
                self.events.put(("error", f"İşleme hatası: {e}"))

    def _do_interim(self, item):
        """Devam eden cümlenin hızlı (küçük model) ön tanıma + ön çevirisi."""
        cid, t0, dur, audio = item
        src, tgt = self.src, self.tgt
        try:
            _t0 = time.monotonic()
            if self.recognizer.speech_seconds(audio) < MIN_VAD_SPEECH_SEC:
                return
            prompt = self._context[-200:] if self.use_context else None
            text, logprob = self.interim_recognizer.transcribe_ex(audio, language=src, prompt=prompt)
            self.stats.setdefault("on_ceviri_sureleri", []).append(
                {"ses_sn": round(dur, 1), "sure": round(time.monotonic() - _t0, 2)})
            text = clean_hallucinations(text, src, 2.0, logprob)
            if not text or cid <= self._last_final_id:   # bu arada kesin çevirisi geldiyse gösterme
                return
            out = self.translator.translate(text, src, tgt, beam_size=1)
            if cid <= self._last_final_id:
                return
            self.stats["interim"] += 1
            self.events.put(("interim", Result(t0, t0 + dur, text, out, src, tgt,
                                               wall=self._wall0 + t0, rid=cid, interim=True)))
        except Exception as e:
            self.events.put(("error", f"Ön çeviri hatası: {e}"))
        finally:
            self._last_interim_done = time.monotonic()

    def stop(self):
        """Kaydı durdurur; kuyruktaki parçalar işlenmeye devam eder ('done' gelene kadar)."""
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._stop.set()
