# object-track-first

[![CI](https://github.com/kdemirwarehouse/object-track-first/actions/workflows/ci.yml/badge.svg)](https://github.com/kdemirwarehouse/object-track-first/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)

YOLOv11 ve ByteTrack kullanarak videoda gerçek zamanlı nesne tespiti, takip ve sayım.
Nesne takibine giriş projesi.

---

## Özellikler

- **Sınıfa göre sayım** — Her nesne türü için benzersiz ID takibi
- **Sanal sayım çizgisi** — Çizgiyi geçen nesneleri ayrıca sayar
- **Takip izleri** — Her nesnenin hareket geçmişini görselleştirir
- **FPS göstergesi** — Anlık işlem hızı paneli
- **Çıktı videosu** — İşlenmiş videoyu diske kaydedebilme

---

## Proje Yapısı

```
object-track-first/
├── .github/
│   └── workflows/
│       └── ci.yml          # lint + syntax check
├── tracker/
│   ├── __init__.py
│   ├── draw.py             # görselleştirme yardımcıları
│   └── engine.py           # takip döngüsü ve ID yönetimi
├── main.py                 # CLI giriş noktası
├── requirements.txt
└── requirements-dev.txt
```

---

## Kurulum

```bash
git clone https://github.com/kdemirwarehouse/object-track-first.git
cd object-track-first
pip install -r requirements.txt
```

> Model dosyası (`yolo11m.pt`) ilk çalıştırmada otomatik indirilir.

---

## Kullanım

```bash
# Temel kullanım
python main.py

# Özel video ve model
python main.py --video video.mp4 --model yolo11s.pt

# Çıktıyı kaydet
python main.py --video traffic.mp4 --output sonuc.mp4

# Sanal çizgiyi özelleştir (frame yüksekliğinin %40'ı)
python main.py --line-ratio 0.40

# Tüm seçenekler
python main.py --help
```

### Argümanlar

| Argüman        | Varsayılan    | Açıklama                              |
|----------------|---------------|---------------------------------------|
| `--video`      | `traffic.mp4` | İşlenecek video dosyası               |
| `--model`      | `yolo11m.pt`  | YOLO model dosyası                    |
| `--output`     | —             | Çıktı video yolu (opsiyonel)          |
| `--conf`       | `0.3`         | Minimum güven eşiği (0.0 – 1.0)      |
| `--line-ratio` | `0.60`        | Sanal çizginin dikey konumu (0.0–1.0) |

Çalarken **`q`** tuşuna basarak çıkabilirsiniz.

---

## Nasıl Çalışır

```
Video karesi
    │
    ▼
YOLOv11  ──►  Bounding box + sınıf + güven skoru
    │
    ▼
ByteTrack ──►  Her tespite kalıcı track_id atar
    │
    ▼
engine.py ──►  Sayım | Çizgi geçişi | Durum yönetimi
    │
    ▼
draw.py   ──►  İzler | Panel | Sanal çizgi | Etiketler
```

1. **Tespit** — YOLOv11 her karede nesneleri bulur.  
2. **Takip** — ByteTrack, kareler arasında nesneleri eşleştirir ve benzersiz `track_id` atar.  
3. **Sayım** — `track_id` setleri tutularak benzersiz nesne sayısı hesaplanır.  
4. **Çizgi geçişi** — Merkez noktanın sanal çizgiyi aştığı an tespit edilir; her ID yalnızca bir kez sayılır.

---

## Örnek Çıktı

```
=== SONUCLAR ===
  car:        47 benzersiz nesne | 31 tanesi cizgiden gecti
  motorcycle: 12 benzersiz nesne |  9 tanesi cizgiden gecti
  truck:       8 benzersiz nesne |  5 tanesi cizgiden gecti
```

---

## Geliştirici Kurulumu

```bash
pip install -r requirements-dev.txt
ruff check .
```

---

## Kullanılan Teknolojiler

| Araç | Görev |
|------|-------|
| [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) | Nesne tespiti |
| ByteTrack | Çok nesne takibi (YOLO içinde entegre) |
| OpenCV | Video okuma / görselleştirme |
| NumPy | Sayısal işlemler |

---

## Lisans

[MIT](LICENSE)
