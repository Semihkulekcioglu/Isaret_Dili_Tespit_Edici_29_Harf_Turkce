# Turkish Sign Language (Letters) Recognition

A simple pipeline to collect hand images, extract landmarks with MediaPipe, train an MLP classifier with scikit-learn, and run real-time predictions from the webcam for the Turkish alphabet letters.

<img width="640" height="640" alt="ABC_pict" src="https://github.com/user-attachments/assets/f21e8054-d796-4903-83e3-7f1924ac26c7" />

## Project Structure
- `veri_toplama.py`: Collect images per letter from your webcam into `data/`.
- `veri_isleme.py`: Extract hand landmarks from images into `data.pickle`.
- `model_egitimi.py`: Train an MLP model and save `model.p`.
- `kameradan_tahmin.py`: Run real-time predictions from your webcam using the trained model.
- `data/`: Collected images per letter (ignored in Git).
- `data.pickle`: Processed dataset (ignored in Git).
- `model.p`: Trained model and label encoder (ignored in Git).

## Requirements
See `requirements.txt` and install into a virtual environment.

## Setup
```bash
# (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage
1) Data collection (creates folders for Turkish letters and captures frames):
```bash
python veri_toplama.py
```

2) Preprocess images to extract landmarks and build dataset:
```bash
python veri_isleme.py
```

3) Train the model and export `model.p`:
```bash
python model_egitimi.py
```

4) Run real-time prediction from webcam:
```bash
python kameradan_tahmin.py
```

## Notes
- The dataset folders represent Turkish letters: `A, B, C, Ç, D, E, F, G, Ğ, H, I, İ, J, K, L, M, N, O, Ö, P, R, S, Ş, T, U, Ü, V, Y, Z`.
- `data/`, `data.pickle`, and `model.p` are excluded from version control via `.gitignore`.
- Make sure your camera is accessible by OpenCV on Windows.
