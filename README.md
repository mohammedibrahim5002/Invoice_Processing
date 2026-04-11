# Invoice OCR

Extract structured data from scanned invoices and receipts using Tesseract OCR and a trained CRF model.

---

## Live demo

Try the deployed app here: https://invoice-processing-ocr.streamlit.app/



## What it does

Upload a scanned invoice image (JPG, PNG, PDF) and get back structured fields:

| Field | Source |
|---|---|
| Invoice number | Regex parser |
| Date | Regex parser + CRF |
| Due date | Regex parser |
| Vendor name | Regex parser (company suffix detection) |
| Company | CRF model |
| Address | CRF model |
| Total amount | Regex parser + spatial fallback |
| Tax amount | Regex parser |
| Currency | Regex parser |
| Email | Regex parser |
| Phone | Regex parser |

Results download as Excel (with confidence scores and batch stats) or CSV.

---

## Project structure

```
invoice-ocr/
├── app.py                  # Streamlit web app
├── notebook.ipynb          # EDA + CRF training + evaluation
├── requirements.txt
├── README.md
│
├── src/
│   ├── ocr_engine.py       # Image preprocessing + Tesseract OCR
│   ├── parser.py           # Regex field extraction
│   ├── model.py            # CRF model — train, predict, evaluate
│   └── exporter.py         # Excel and CSV export
│
├── data/
│   ├── raw/                # SROIE 2019 dataset (not committed)
│   └── processed/          # labelled_corpus.csv and EDA outputs
│
├── models/
│   ├── crf_model.pkl       # Model metadata
│   └── crf_model.crfsuite  # Trained CRF weights
│
├── samples/                # Test invoice images
└── outputs/                # Extraction results (not committed)
```

---

## Setup

### 1. Clone and create environment

```bash
git clone https://github.com/yourusername/invoice-ocr.git
cd invoice-ocr

conda create -n invoice-ocr python=3.11
conda activate invoice-ocr
```

### 2. Install Tesseract

```bash
# Anaconda (recommended — handles PATH automatically)
conda install -c conda-forge tesseract

# macOS
brew install tesseract

# Ubuntu / Debian
sudo apt install tesseract-ocr
```

### 3. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Tesseract path (Windows only)

Find your tessdata folder:

```powershell
Get-ChildItem -Path "C:\miniconda3" -Recurse -Filter "eng.traineddata" -ErrorAction SilentlyContinue | Select-Object FullName
```

Then add these two lines at the top of `src/ocr_engine.py`:

```python
import os
os.environ['TESSDATA_PREFIX'] = r'C:\miniconda3\envs\invoice-ocr\share\tessdata'
```

---

## Dataset

**SROIE 2019** — Scanned Receipts OCR and Information Extraction  
ICDAR 2019 competition dataset. 973 annotated receipt images.

```bash
# Download via Kaggle CLI
kaggle datasets download -d urbikn/sroie-datasetv2 -p data/raw --unzip

# Or clone directly
git clone https://github.com/zzzDavid/ICDAR-2019-SROIE.git data/raw/sroie
```

Dataset structure after download:

```
data/raw/SROIE2019/
├── train/
│   ├── img/        # 626 receipt images
│   ├── box/        # Word-level bounding box annotations
│   └── entities/   # Ground truth JSON per receipt
└── test/
    ├── img/        # 347 receipt images
    ├── box/
    └── entities/
```

---

## Training the model

Open `notebook.ipynb` and run all cells in order. The notebook covers:

1. Dataset integrity check
2. Image quality EDA (brightness, contrast, resolution)
3. Entity field analysis (presence rates, format distributions)
4. BIO label assignment — every word gets labelled `B-COMPANY`, `I-ADDRESS`, `O`, etc.
5. CRF training with 35+ spatial and linguistic features
6. Evaluation on held-out test set

The trained model saves to `models/crf_model.crfsuite`.

**Training takes 2–5 minutes on CPU. No GPU required.**

---

## Running the app

```bash
conda activate invoice-ocr
streamlit run app.py
```

Opens at `http://localhost:8501`.

### Single invoice
Upload one image. Extracted fields appear immediately with confidence scores. Download as Excel or CSV.

### Batch processing
Upload multiple invoices. Progress bar shows processing status. Download all results in one Excel file with a summary sheet and per-field confidence sheet.

---

## Model performance

Evaluated on the SROIE 2019 test set (347 receipts):

| Entity | Precision | Recall | F1 |
|---|---|---|---|
| Company | 0.96 | 0.96 | **0.96** |
| Date | 0.89 | 0.88 | **0.88** |
| Address | 0.83 | 0.78 | **0.80** |
| Total | 0.74 | 0.51 | 0.60 |
| **Overall (weighted)** | **0.85** | **0.77** | **0.81** |

The model uses a Conditional Random Field (CRF) with features including:

- Token text, prefix, suffix, shape flags
- Normalised x/y position on the page (0.0–1.0)
- Keyword proximity (distance to TOTAL, DATE, ADDRESS keywords)
- Duplicate value detection (for receipts where the same amount appears multiple times)
- Combined spatial+semantic features (e.g. `amount_far_right_bottom_no_skip`)

**Why CRF over a neural network:** trains in minutes on CPU, single `.crfsuite` file for deployment, interpretable features, achieves competitive F1 without GPU or large pretrained models. LayoutLM achieves ~0.95 F1 on the same benchmark but requires a GPU and 340M parameter model.

---

## How it works

```
Invoice image
    |
    v
ocr_engine.py
  - Grayscale + upscale
  - Deskew (Hough line detection)
  - Denoise (fastNlMeans)
  - Adaptive threshold
  - Tesseract PSM 6 -> token list with bounding boxes
    |
    v
parser.py (regex layer)
  - Matches date, total, tax, invoice number, email, phone
  - Handles 5+ date formats, 4+ total formats, currency symbols
  - Spatial fallback for total (topmost right-side amount above CASH line)
    |
    v
model.py (CRF layer)
  - 50+ features per token
  - Predicts: B-COMPANY, I-COMPANY, B-DATE, B-ADDRESS, I-ADDRESS, B-TOTAL, O
  - extract_entities() joins BIO spans into strings
    |
    v
Merge (highest confidence wins per field)
    |
    v
exporter.py
  - Excel: Summary + Confidence + Stats sheets
  - CSV: flat file with confidence columns
```

---

## Limitations

- **Total field** (F1 0.60): some receipt layouts print the same amount 3–4 times (subtotal, total, cash amount, change). The model uses keyword proximity and spatial position to pick the right one but still misses ~40% in these edge cases.
- **Handwritten invoices**: Tesseract accuracy drops significantly on handwriting. Use Claude Vision API for those.
- **Non-English**: SROIE is English-only. Malay receipts (common in the dataset) partially work. Other languages will have lower accuracy.
- **Company names**: no standard format — the CRF uses positional features (top 20% of receipt) and company suffixes (SDN BHD, LTD, etc.) but struggles on receipts where the company name is not at the top.

---

## Tech stack

| Component | Library | Version |
|---|---|---|
| OCR engine | Tesseract | 5.x |
| Image processing | OpenCV | 4.8+ |
| ML model | python-crfsuite | 0.9.9 |
| Evaluation | scikit-learn | 1.3+ |
| Data | pandas | 2.0+ |
| Export | openpyxl | 3.1+ |
| Web app | Streamlit | 1.28+ |
| Python | — | 3.11 |

---

## Requirements

```
pytesseract
opencv-python-headless
Pillow
scikit-learn
python-crfsuite
numpy
pandas
openpyxl
matplotlib
seaborn
streamlit
kaggle
tqdm
```

Install: `pip install -r requirements.txt`

---

## License

MIT
