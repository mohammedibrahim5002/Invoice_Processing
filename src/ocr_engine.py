"""
src/ocr_engine.py
=================
Takes a raw image or PDF path.
Returns a clean list of token dicts with bounding boxes.

Two modes:
  1. box_path given  -> use SROIE ground-truth box file (training/evaluation)
  2. box_path None   -> preprocess image + run Tesseract (inference on new invoices)

Output token format (same as labelled_corpus.csv):
    {
        'stem'         : 'X00016469612',
        'text'         : 'TOTAL',
        'x_min'        : 42,
        'y_min'        : 310,
        'x_max'        : 118,
        'y_max'        : 334,
        'width'        : 76,
        'height'       : 24,
        'conf'         : 91,
        'line_num'     : 12,
        'word_num'     : 3,
        'x_norm'       : 0.062,
        'y_norm'       : 0.216,
        'x_center_norm': 0.118,
        'y_center_norm': 0.228,
        'width_norm'   : 0.112,
        'height_norm'  : 0.017,
    }

Usage:
    from src.ocr_engine import extract_tokens

    # Mode 1: SROIE box file (use during training)
    tokens = extract_tokens(
        image_path = 'data/raw/SROIE2019/train/img/X00016469612.jpg',
        box_path   = 'data/raw/SROIE2019/train/box/X00016469612.txt'
    )

    # Mode 2: Tesseract on new invoice (use in web app)
    tokens = extract_tokens(
        image_path = 'samples/my_invoice.jpg'
    )
"""

import os
import sys
import platform
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from PIL import Image


# ── Tesseract path — hardcoded for this machine ───────────────────────────────
# eng.traineddata found at: C:\miniconda3\envs\ml_finance\share\tessdata
# If you move environments or reinstall, update these two lines.

import platform
if platform.system() == 'Windows':
    os.environ['TESSDATA_PREFIX'] = r'C:\miniconda3\envs\ml_finance\share\tessdata'
    pytesseract.pytesseract.tesseract_cmd = r'C:\miniconda3\envs\ml_finance\Library\bin\tesseract.exe'
# Linux (Streamlit Cloud): tesseract is on PATH after packages.txt install

# ── Auto-detect Tesseract path (fallback for other machines) ──────────────────

def _set_tesseract_path():
    """
    Set pytesseract.tesseract_cmd to the correct path for this OS.
    Only runs if the hardcoded path above does not exist.
    """
    # If hardcoded path works, skip auto-detection
    hardcoded = Path(pytesseract.pytesseract.tesseract_cmd)
    if hardcoded.exists():
        return

    system = platform.system()

    if system == 'Windows':
        candidates = [
            r'C:\Program Files\Tesseract-OCR\tesseract.exe',
            r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe',
            r'C:\miniconda3\envs\ml_finance\Library\bin\tesseract.exe',
            r'C:\miniconda3\Library\bin\tesseract.exe',
        ]
        tessdata_candidates = [
            r'C:\miniconda3\envs\ml_finance\share\tessdata',
            r'C:\miniconda3\share\tessdata',
            r'C:\Program Files\Tesseract-OCR\tessdata',
        ]
        for path in candidates:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                break
        for td in tessdata_candidates:
            if os.path.exists(td):
                os.environ['TESSDATA_PREFIX'] = td
                break

    elif system == 'Darwin':  # macOS
        candidates = [
            '/usr/local/bin/tesseract',
            '/opt/homebrew/bin/tesseract',
        ]
        for path in candidates:
            if os.path.exists(path):
                pytesseract.pytesseract.tesseract_cmd = path
                return

    # Linux / conda: tesseract is on PATH


_set_tesseract_path()


# ── Preprocessing ─────────────────────────────────────────────────────────────

def load_image(image_path):
    """
    Load image from disk as a numpy array (BGR).
    Handles image files and single-page PDFs.
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f'Image not found: {path}')

    if path.suffix.lower() == '.pdf':
        try:
            import fitz  # PyMuPDF
            doc  = fitz.open(str(path))
            page = doc[0]
            mat  = fitz.Matrix(2.0, 2.0)
            pix  = page.get_pixmap(matrix=mat)
            img  = np.frombuffer(pix.samples, dtype=np.uint8)
            img  = img.reshape(pix.height, pix.width, pix.n)
            if pix.n == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            elif pix.n == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            return img
        except ImportError:
            raise ImportError('PDF support requires: pip install pymupdf')
    else:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f'Could not read image: {path}')
        return img


def preprocess(img):
    """
    Clean up the image so Tesseract reads it accurately.

    Steps:
      1. Grayscale       - remove colour
      2. Upscale         - improve OCR on small images
      3. Deskew          - correct tilt/rotation
      4. Denoise         - remove noise speckles
      5. Threshold       - binarize: black text on white background
    """
    # 1. Grayscale
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # 2. Upscale if too small
    h, w = gray.shape
    if max(h, w) < 1000:
        scale = 1500 / max(h, w)
        gray  = cv2.resize(gray, None, fx=scale, fy=scale,
                           interpolation=cv2.INTER_CUBIC)

    # 3. Deskew
    gray = _deskew(gray)

    # 4. Denoise
    gray = cv2.fastNlMeansDenoising(gray, h=10, templateWindowSize=7,
                                    searchWindowSize=21)

    # 5. Adaptive threshold
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        blockSize=31,
        C=10
    )

    return binary


def _deskew(gray):
    """
    Detect tilt angle and rotate image to correct it.
    Returns corrected image, or original if no correction needed.
    """
    try:
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return gray

        angles = []
        for line in lines[:20]:
            rho, theta = line[0]
            angle = (theta * 180 / np.pi) - 90
            if -45 < angle < 45:
                angles.append(angle)

        if not angles:
            return gray

        skew_angle = float(np.median(angles))

        if abs(skew_angle) < 0.3:
            return gray

        h, w   = gray.shape
        center = (w // 2, h // 2)
        M      = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        result = cv2.warpAffine(
            gray, M, (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )
        return result

    except Exception:
        return gray


# ── Tesseract OCR ─────────────────────────────────────────────────────────────

def run_tesseract(img, stem=''):
    """
    Run Tesseract on a preprocessed image.
    Returns list of token dicts with bounding boxes and confidence.
    """
    config = '--psm 6 --oem 3'

    data = pytesseract.image_to_data(
        img,
        config=config,
        output_type=pytesseract.Output.DICT
    )

    tokens = []
    n      = len(data['text'])

    for i in range(n):
        text = data['text'][i].strip()
        conf = int(data['conf'][i])

        if not text or conf < 10:
            continue

        x = data['left'][i]
        y = data['top'][i]
        w = data['width'][i]
        h = data['height'][i]

        if w == 0 or h == 0:
            continue

        tokens.append({
            'stem'    : stem,
            'text'    : text,
            'x_min'   : x,
            'y_min'   : y,
            'x_max'   : x + w,
            'y_max'   : y + h,
            'width'   : w,
            'height'  : h,
            'conf'    : conf,
            'line_num': data['line_num'][i],
            'word_num': data['word_num'][i],
        })

    return tokens


# ── SROIE box file loader ─────────────────────────────────────────────────────

def load_box_file(box_path, stem=''):
    """
    Load ground-truth bounding boxes from a SROIE .txt box file.
    Line format: x1,y1,x2,y2,x3,y3,x4,y4,TEXT
    """
    path   = Path(box_path)
    tokens = []

    with open(path, encoding='utf-8', errors='replace') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(',')
            if len(parts) < 9:
                continue

            try:
                coords = list(map(int, parts[:8]))
            except ValueError:
                continue

            text = ','.join(parts[8:]).strip()
            if not text:
                continue

            xs = coords[0::2]
            ys = coords[1::2]

            tokens.append({
                'stem'    : stem,
                'text'    : text,
                'x_min'   : min(xs),
                'y_min'   : min(ys),
                'x_max'   : max(xs),
                'y_max'   : max(ys),
                'width'   : max(xs) - min(xs),
                'height'  : max(ys) - min(ys),
                'conf'    : -1,
                'line_num': 0,
                'word_num': 0,
            })

    return tokens


# ── Normalise coordinates ─────────────────────────────────────────────────────

def normalise_coords(tokens, img_w, img_h):
    """
    Add normalised x/y positions (0.0 to 1.0) to each token.
    Used as features by the CRF model.
    """
    for t in tokens:
        t['x_norm']        = round(t['x_min'] / img_w, 4) if img_w else 0
        t['y_norm']        = round(t['y_min'] / img_h, 4) if img_h else 0
        t['x_center_norm'] = round((t['x_min'] + t['x_max']) / 2 / img_w, 4) if img_w else 0
        t['y_center_norm'] = round((t['y_min'] + t['y_max']) / 2 / img_h, 4) if img_h else 0
        t['width_norm']    = round(t['width']  / img_w, 4) if img_w else 0
        t['height_norm']   = round(t['height'] / img_h, 4) if img_h else 0
    return tokens


# ── Main public function ──────────────────────────────────────────────────────

def extract_tokens(image_path, box_path=None):
    """
    Main entry point. Call this from the notebook, model.py, and app.py.

    Args:
        image_path : path to invoice image (.jpg .png .pdf)
        box_path   : (optional) path to SROIE .txt box file
                     If provided, uses ground-truth boxes instead of Tesseract.
                     Use during training for accurate features.
                     Leave None for inference on new invoices.

    Returns:
        List of token dicts with bounding boxes + normalised coordinates.
    """
    image_path = Path(image_path)
    stem       = image_path.stem

    img          = load_image(image_path)
    img_h, img_w = img.shape[:2]

    if box_path is not None:
        tokens = load_box_file(box_path, stem=stem)
    else:
        processed = preprocess(img)
        tokens    = run_tesseract(processed, stem=stem)

    tokens = normalise_coords(tokens, img_w, img_h)

    return tokens


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print('Testing OCR engine...\n')

    # Verify Tesseract
    try:
        version = pytesseract.get_tesseract_version()
        print(f'Tesseract version : {version}')
        print(f'Tesseract cmd     : {pytesseract.pytesseract.tesseract_cmd}')
        print(f'TESSDATA_PREFIX   : {os.environ.get("TESSDATA_PREFIX", "not set")}')
    except Exception as e:
        print(f'ERROR: Tesseract not found: {e}')
        sys.exit(1)

    sroie_img = Path('data/raw/SROIE2019/train/img')
    sroie_box = Path('data/raw/SROIE2019/train/box')

    if not sroie_img.exists():
        print('\nSROIE dataset not found — skipping token tests.')
        sys.exit(0)

    sample_img = sorted(sroie_img.glob('*.jpg'))[0]
    sample_box = sroie_box / f'{sample_img.stem}.txt'

    # Mode 1: box file
    print(f'\nMode 1 — Box file: {sample_img.name}')
    tokens = extract_tokens(sample_img, box_path=sample_box)
    print(f'Tokens: {len(tokens)}')
    print('First 3:')
    for t in tokens[:3]:
        print(f'  text={t["text"]:30s}  x={t["x_min"]:4d}  y={t["y_min"]:4d}'
              f'  x_norm={t["x_norm"]:.3f}  y_norm={t["y_norm"]:.3f}')

    # Mode 2: Tesseract
    print(f'\nMode 2 — Tesseract (preprocessing + OCR)')
    tokens2 = extract_tokens(sample_img)
    print(f'Tokens: {len(tokens2)}')
    print('First 3:')
    for t in tokens2[:3]:
        print(f'  text={t["text"]:30s}  conf={t["conf"]:3d}'
              f'  x_norm={t["x_norm"]:.3f}  y_norm={t["y_norm"]:.3f}')

    print('\nOCR engine OK.')
