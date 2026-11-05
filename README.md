# Spiral Temizlemeli & XY-Cut Çift Sütun Markdown OCR

Sayfa kenarlarındaki defter/kitap spirallerini görüntü işleme aşamasında dışlayan, üstte tam genişlikli başlık ve altta çift sütun bulunan karma sayfa düzenlerini **Sütun Öncelikli Özyinelemeli XY-Cut** algoritmasıyla doğru okuma sırasına dizerek **Markdown (`.md`)** belgesine dönüştüren OCR boru hattı.

## GitHub Üzerinden Otomatik Kullanım (Sunucusuz)

1. `input/` klasörüne taranmış **PDF** dosyanızı veya sayfa fotoğraflarınızı (`.jpg`, `.png`) yükleyip **Commit changes** butonuna basın.
2. **Actions** sekmesinde `Spiral & XY-Cut Markdown OCR` iş akışı otomatik olarak başlar.
3. İşlem tamamlandığında:
   - Çözümlenen Markdown dosyaları `output/` klasörüne,
   - Başlık ve sütunların nasıl ayrıldığını gösteren renk kodlu doğrulama görselleri `output/debug/` klasörüne otomatik kaydedilir.

## Yerel Bilgisayarda Çalıştırma

```bash
pip install -r requirements.txt
python ocr_pipeline.py --input input --output output --debug-dir output/debug
