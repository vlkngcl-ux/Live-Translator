# Canlı Çevirmen (Romence · İngilizce · Türkçe)

Ortamdaki konuşmayı mikrofondan dinler, seçtiğiniz dile çevirip **metin** olarak ekranda gösterir
ve **TXT** ya da **DOCX** olarak kaydetmenizi sağlar. **Tamamen offline çalışır**: kurulumdan sonra
internet gerekmez, ses hiçbir yere gönderilmez.

**Dil çiftleri:** Romence → Türkçe, Romence → İngilizce, İngilizce → Romence,
İngilizce → Türkçe, Türkçe → İngilizce, Türkçe → Romence (üç dilin her yönü).

- Konuşma tanıma: OpenAI Whisper (large-v3-turbo ve small), CTranslate2 int8
- Çeviri: Meta NLLB-200 (600M), dil çiftleri arasında doğrudan çeviri (aradan dil kullanmaz), CTranslate2 int8

## Desteklenen sistemler

| Sistem | Durum |
|---|---|
| Windows 10/11 (64-bit) | Desteklenir |
| macOS 11+ — Apple Silicon (M1/M2/M3/M4) | Desteklenir |
| macOS — Intel işlemcili Mac | **Desteklenmez** (çeviri motorunun Intel Mac paketi yok) |

Kurulum boyutu yaklaşık 1,5–2 GB'tır (modeller dahil).

---

## Kullanıcı için kurulum

Kurulum dosyaları GitHub'daki **Releases** sayfasındadır.

### Windows
1. `CanliCevirmen-Setup-Windows.exe` dosyasını indirip çalıştırın.
2. Windows "bilinmeyen yayımcı" uyarısı verirse **Ek bilgi → Yine de çalıştır** deyin
   (uygulama imzalı değildir).
3. Yönetici izni gerekmez. Başlat menüsünde ve masaüstünde "Canlı Çevirmen" oluşur.

### macOS (Apple Silicon)
1. `CanliCevirmen-macOS-AppleSilicon.dmg` dosyasını açın, uygulamayı **Applications**
   klasörüne sürükleyin.
2. İlk açılışta macOS "geliştirici doğrulanamadı" diyebilir (uygulama Apple hesabıyla imzalı
   değildir). Bu durumda: **Sistem Ayarları → Gizlilik ve Güvenlik** sayfasının altındaki
   **"Yine de Aç"** düğmesine basın.
3. Mikrofon izni sorulduğunda **İzin Ver** deyin. Yanlışlıkla reddettiyseniz:
   Sistem Ayarları → Gizlilik ve Güvenlik → Mikrofon → Canlı Çevirmen'i açın.

## Kullanım

1. **Mikrofon** listesinden kullanılacak mikrofonu seçin.
2. **Konuşulan dil** ve **Çeviri dili**'ni seçin. **⇄** düğmesi ikisinin yerini değiştirir.
   Diller kayıt sırasında değiştirilemez; değiştirmek için önce durdurun.
3. **Mod**:
   - **Doğru (turbo)**: daha isabetli, daha çok işlemci ister (varsayılan).
   - **Hızlı (small)**: zayıf bilgisayarlar için; hatalar daha fazladır.
4. **▶ Başlat**. Konuşurken cümlenin **ön çevirisi** en altta gri-italik ve "…" ile görünür ve
   konuşma sürdükçe güncellenir (hızlı modelle). Cümle bitince yerini doğru modelin
   **kesin çevirisi** alır. Ön çeviriyi istemiyorsanız **"Ön çeviri"** kutusunu kapatın
   (kapalıyken başlatılırsa hızlı model hiç yüklenmez, işlemci rahatlar).
5. **■ Durdur** dediğinizde sırada kalan konuşma da işlenir.
6. **TXT kaydet** / **DOCX kaydet** ile metni kaydedin. Dosyaya yalnızca kesin çeviriler yazılır.
   "Konuşulan metni de göster/kaydet" işaretliyse orijinal metin de eklenir.
   **"Saat"** işaretliyse her cümlenin başında konuşmanın **sistem saati** (ör. `[18:12:05]`) yazar.

Sağ alttaki **"Sırada bekleyen"** sayısı sürekli artıyorsa bilgisayar konuşmaya yetişemiyordur;
**Hızlı (small)** moduna geçin.

## Bilinen sınırlar

- Çeviri makine çevirisidir; özel isimler, rakamlar ve uzun cümlelerde hata yapabilir.
  Önemli metinleri aslıyla karşılaştırın.
- Konuşulan dil otomatik algılanmaz; seçilen dil dışında konuşulursa sonuç bozuk olur.
- Romence tanıma, İngilizce ve Türkçe'ye göre daha zayıftır (Whisper'ın eğitim verisinde
  Romence çok az).
- Gürültülü ortam, uzak mikrofon ve aynı anda konuşan kişiler tanıma kalitesini düşürür.
- Bu bir "simültane çeviri" değildir: konuşmacı durakladıkça parça parça çevirir.
- **Gecikme** çoğunlukla bilgisayarın Whisper'ı çalıştırma hızına bağlıdır; Whisper her parçayı
  30 saniyeye tamamlayarak işler, bu yüzden kısa bir cümle de uzun bir cümle kadar sürer.
  Sağ alttaki **"Gecikme"** göstergesi her cümlenin konuşma bittikten kaç saniye sonra geldiğini
  gösterir. Çok yüksekse **Hızlı (small)** modu yaklaşık 3 kat daha hızlıdır (ama daha çok hata yapar).
- **Uydurma filtresi:** Whisper, konuşma olmayan seste "teşekkürler / görüşürüz / abone olun"
  gibi ezber cümleler yazabilir. Uygulama bunları elemeye çalışır; yan etkisi olarak tek başına
  söylenmiş kısa ve zor duyulan bir "teşekkürler" de elenebilir.

---

## Kurulum dosyalarını üretmek (geliştirici)

Kurulum dosyaları GitHub Actions ile otomatik üretilir; kendi bilgisayarınızda derleme gerekmez.

**Otomatik:** `main` dalına uygulamayı etkileyen bir değişiklik (`app/`, `build/`, `installer/`,
`scripts/`, `requirements.txt`, `VERSION` veya iş akışı dosyası) gönderildiğinde derleme kendiliğinden
başlar. Sürüm numarası `VERSION` dosyasından alınır.

- **Yeni sürüm çıkarmak için `VERSION`'ı artırın** (ör. `1.1.0` → `1.1.1`). O sürüm zaten
  yayınlanmışsa otomatik derleme üzerine yazmaz, hata verip durur.
- Yalnızca README gibi dosyalar değişirse derleme başlamaz.
- Bir gönderimde derleme istemiyorsanız commit mesajına `[skip ci]` yazın.

**Elle:** Actions → **"Kurulum dosyalarını oluştur"** → **Run workflow** (sürüm kutusu boş = `VERSION`).
⚠️ Eski bir çalışmadaki **"Re-run jobs"** düğmesi o çalışmanın **eski kodunu** yeniden derler; yeni kod
için her zaman **Run workflow** kullanın.

Her derlemede:
1. İlk çalıştırmada modeller indirilip dönüştürülür; sonra önbellekten gelir.
2. Her sistemde paketlenmiş uygulama, gerçek Romence, İngilizce ve Türkçe ses kayıtlarını **offline**
   tanıyıp çeviren bir kendi kendine testten geçer. Test başarısız olursa dosya yayınlanmaz.
3. Bittiğinde **Releases** sayfasında `v<sürüm>` altında `.exe` ve `.dmg` dosyaları belirir.

### Yerelde çalıştırma (geliştirme)

```bash
pip install -r requirements.txt
pip install torch --index-url https://download.pytorch.org/whl/cpu    # yalnızca model dönüştürme için
pip install ctranslate2==4.8.2 transformers==4.57.6 soundfile
python scripts/prepare_models.py          # models/ klasörünü oluşturur (internet gerekir)
python app/main.py
```

Test (mikrofonsuz): `python tests/test_engine.py <wav_klasoru> large-v3-turbo <kaynak> <hedef>` (ör. `en tr`)

## Proje yapısı

```
app/main.py                  Arayüz (tkinter)
app/engine.py                Mikrofon, konuşma bölme, tanıma, çeviri
app/export.py                TXT / DOCX kaydetme (ek kütüphane gerektirmez)
scripts/prepare_models.py    Modelleri indirip int8'e çevirir (yalnızca derlemede)
build/app.spec               PyInstaller paket tanımı
installer/windows.iss        Windows kurulum sihirbazı (Inno Setup)
.github/workflows/build.yml  Otomatik derleme + test + yayın
```

## Lisanslar

- **NLLB-200** (çeviri modeli): CC-BY-NC 4.0, **yalnızca ticari olmayan kullanım.**
  Ticari kullanım gerekirse çeviri modeli değiştirilmelidir.
- **Whisper** (konuşma tanıma modeli): MIT.
- CI testindeki örnek sesler (Romence, İngilizce, Türkçe): Google FLEURS veri seti (CC-BY 4.0); uygulama paketine dahil edilmez.
