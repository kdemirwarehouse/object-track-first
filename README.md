# object-track-first

YOLOv11 ve ByteTrack kullanarak videoda gerçek zamanlı nesne tespiti ve takibi. Nesne takibine giriş projesi.

## Özellikler

- **Sınıfa göre sayım** — Her nesne türü için benzersiz ID takibi
- **Sanal sayım çizgisi** — Belirli bir çizgiyi geçen nesneleri ayrıca sayar
- **Takip izleri** — Her nesnenin hareket yolunu görselleştirir
- **FPS göstergesi** — Anlık işlem hızı paneli
- **Çıktı videosu** — İşlenmiş videoyu diske kaydedebilme

## Kurulum

```bash
pip install -r requirements.txt
```

> Model dosyası (`yolo11m.pt`) ilk çalıştırmada otomatik indirilir.

## Kullanım

```bash
# Temel kullanım
python track.py

# Özel video ve model
python track.py --video video.mp4 --model yolo11s.pt

# Çıktıyı kaydet
python track.py --video traffic.mp4 --output sonuc.mp4

# Tüm seçenekler
python track.py --video traffic.mp4 --model yolo11m.pt --output sonuc.mp4 --conf 0.4
```

### Argümanlar

| Argüman    | Varsayılan    | Açıklama                         |
|------------|---------------|----------------------------------|
| `--video`  | `traffic.mp4` | İşlenecek video dosyası          |
| `--model`  | `yolo11m.pt`  | YOLO model dosyası               |
| `--output` | —             | Çıktı video yolu (opsiyonel)     |
| `--conf`   | `0.3`         | Minimum güven eşiği (0.0 – 1.0) |

Çalarken **`q`** tuşuna basarak çıkabilirsiniz.

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
Uygulama  ──►  İz çizimi | Çizgi geçişi | Sayım paneli
```

1. **Tespit**: YOLOv11 her karede nesneleri bulur.
2. **Takip**: ByteTrack, kareler arasında aynı nesneyi eşleştirir ve benzersiz `track_id` atar.
3. **Sayım**: `track_id` setleri tutularak görülen toplam benzersiz nesne sayısı hesaplanır.
4. **Çizgi geçişi**: Nesnenin merkez noktasının sanal çizgiyi geçip geçmediği kontrol edilir.

## Örnek Çıktı

```
=== SONUCLAR ===
  car: 47 benzersiz nesne tespit edildi, 31 tanesi cizgiden gecti
  truck: 8 benzersiz nesne tespit edildi, 5 tanesi cizgiden gecti
  motorcycle: 12 benzersiz nesne tespit edildi, 9 tanesi cizgiden gecti
```

## Kullanılan Teknolojiler

| Araç | Görev |
|------|-------|
| [Ultralytics YOLOv11](https://github.com/ultralytics/ultralytics) | Nesne tespiti |
| ByteTrack | Çok nesne takibi (YOLO içinde entegre) |
| OpenCV | Video okuma / görselleştirme |
| NumPy | Sayısal işlemler |
