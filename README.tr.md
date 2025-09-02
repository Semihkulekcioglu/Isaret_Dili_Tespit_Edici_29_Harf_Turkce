# İşaret Dili (Harfler) Tanıma

Bu proje; MediaPipe ile el işaretleri noktalarını (landmark) çıkarıp, scikit-learn kullanarak basit bir MLP sınıflandırıcı eğiten ve eğitilen modeli kullanarak webcam üzerinden gerçek zamanlı tahmin yapan bir boru hattı içerir.

<img width="320" height="320" alt="ABC_pict" src="https://github.com/user-attachments/assets/f21e8054-d796-4903-83e3-7f1924ac26c7" />

## Proje Yapısı
- `veri_toplama.py`: Webcaminizdan her harf için görüntü toplar ve `data/` klasörüne kaydeder.
- `veri_isleme.py`: Görüntülerden el landmark noktalarını çıkarır ve `data.pickle` dosyasını oluşturur.
- `model_egitimi.py`: MLP modelini eğitir ve `model.p` dosyasını kaydeder.
- `kameradan_tahmin.py`: Eğitilen modeli kullanarak webcam üzerinden gerçek zamanlı tahmin yapar.
- `data/`: Toplanan görüntülerin saklandığı klasör (Git tarafından yok sayılır).
- `data.pickle`: İşlenmiş veri seti (Git tarafından yok sayılır).
- `model.p`: Eğitilen model ve etiket kodlayıcı (Git tarafından yok sayılır).

## Gereksinimler
`requirements.txt` içindeki paketleri sanal ortama kurunuz.

## Kurulum
```bash
# (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Kullanım
1) Veri toplama (Türk alfabesi için klasörler oluşturur ve kareler kaydeder):
```bash
python veri_toplama.py
```

2) Görselleri işleme (landmark çıkarma ve veri seti oluşturma):
```bash
python veri_isleme.py
```

3) Modeli eğitme ve `model.p` çıktısını alma:
```bash
python model_egitimi.py
```

4) Webcameradan gerçek zamanlı tahmin:
```bash
python kameradan_tahmin.py
```

## Notlar
- Klasörler, Türk alfabesindeki harfleri temsil eder: `A, B, C, Ç, D, E, F, G, Ğ, H, I, İ, J, K, L, M, N, O, Ö, P, R, S, Ş, T, U, Ü, V, Y, Z`.
- `data/`, `data.pickle` ve `model.p` dosyaları `.gitignore` ile sürüm kontrolü dışında bırakılmıştır.
- OpenCV'nin Windows üzerinde kameranıza erişebildiğinden emin olun.
