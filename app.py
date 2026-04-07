"""
app.py — Invoice OCR
Clean, professional Streamlit UI inspired by tools like ilovepdf.
Run: streamlit run app.py
"""

import os
import sys
import re
import tempfile
import pickle
from pathlib import Path
from datetime import datetime

import streamlit as st
import pandas as pd

sys.path.insert(0, '.')

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title='Invoice OCR',
    page_icon='assets/favicon.ico' if Path('assets/favicon.ico').exists() else None,
    layout='wide',
    initial_sidebar_state='collapsed',
)

# ── CSS ───────────────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Mono:wght@400;500&display=swap');

/* ── Reset & base ──────────────────────────────────────────────────────── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    font-family: 'DM Sans', sans-serif;
    background: #F7F7F5;
    color: #1A1A1A;
}

[data-testid="stAppViewContainer"] > .main {
    background: #F7F7F5;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }
.stDeployButton { display: none; }
[data-testid="collapsedControl"] { display: none; }

/* ── Top nav bar ───────────────────────────────────────────────────────── */
.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 0 28px 0;
    border-bottom: 1px solid #E5E5E2;
    margin-bottom: 48px;
}
.nav-logo {
    font-size: 17px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.3px;
}
.nav-logo span {
    color: #D64000;
}
.nav-tag {
    font-size: 11px;
    font-weight: 500;
    color: #888;
    letter-spacing: 0.8px;
    text-transform: uppercase;
}

/* ── Hero section ──────────────────────────────────────────────────────── */
.hero {
    text-align: center;
    padding: 0 0 56px 0;
}
.hero h1 {
    font-size: 42px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -1.2px;
    line-height: 1.15;
    margin: 0 0 14px 0;
}
.hero h1 em {
    font-style: normal;
    color: #D64000;
}
.hero p {
    font-size: 16px;
    color: #666;
    font-weight: 400;
    margin: 0;
    line-height: 1.6;
}

/* ── Upload zone ───────────────────────────────────────────────────────── */
.upload-zone {
    background: #FFFFFF;
    border: 1.5px dashed #D0D0CC;
    border-radius: 12px;
    padding: 52px 40px;
    text-align: center;
    transition: border-color 0.2s;
    margin-bottom: 32px;
}
.upload-zone:hover {
    border-color: #D64000;
}
.upload-zone-title {
    font-size: 15px;
    font-weight: 500;
    color: #1A1A1A;
    margin-bottom: 6px;
}
.upload-zone-sub {
    font-size: 13px;
    color: #999;
}

/* Streamlit file uploader override */
[data-testid="stFileUploader"] {
    background: #FFFFFF;
    border: 1.5px dashed #D0D0CC;
    border-radius: 12px;
    padding: 0;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover {
    border-color: #D64000;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
    border: none !important;
    padding: 48px 40px !important;
}
[data-testid="stFileUploadDropzone"] > div {
    gap: 8px !important;
}
.stFileUploader label {
    font-size: 15px !important;
    font-weight: 500 !important;
    color: #1A1A1A !important;
}

/* ── Tabs ──────────────────────────────────────────────────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #E5E5E2;
    gap: 0;
    padding: 0;
    margin-bottom: 40px;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    color: #888 !important;
    background: transparent !important;
    border: none !important;
    border-bottom: 2px solid transparent !important;
    padding: 12px 24px !important;
    margin-right: 4px;
    transition: all 0.15s;
}
.stTabs [aria-selected="true"] {
    color: #1A1A1A !important;
    border-bottom-color: #D64000 !important;
}

/* ── Cards ─────────────────────────────────────────────────────────────── */
.result-card {
    background: #FFFFFF;
    border-radius: 12px;
    border: 1px solid #E5E5E2;
    overflow: hidden;
}
.result-card-header {
    padding: 20px 24px;
    border-bottom: 1px solid #F0F0EE;
    display: flex;
    align-items: center;
    justify-content: space-between;
}
.result-card-title {
    font-size: 13px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: 0.3px;
    text-transform: uppercase;
}
.result-card-body {
    padding: 0;
}

/* ── Field rows ─────────────────────────────────────────────────────────── */
.field-row {
    display: flex;
    align-items: flex-start;
    padding: 14px 24px;
    border-bottom: 1px solid #F5F5F3;
    transition: background 0.12s;
}
.field-row:last-child { border-bottom: none; }
.field-row:hover { background: #FAFAF8; }

.field-label {
    font-size: 12px;
    font-weight: 500;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    width: 140px;
    flex-shrink: 0;
    padding-top: 1px;
}
.field-value {
    font-size: 14px;
    font-weight: 400;
    color: #1A1A1A;
    flex: 1;
    font-family: 'DM Mono', monospace;
}
.field-value.empty {
    color: #CCC;
    font-family: 'DM Sans', sans-serif;
    font-style: italic;
    font-size: 13px;
}
.field-conf {
    font-size: 11px;
    font-weight: 500;
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.3px;
    margin-left: 16px;
    flex-shrink: 0;
}
.conf-high { background: #EBF5E6; color: #2D6A1F; }
.conf-mid  { background: #EEF3FF; color: #2B4BAE; }
.conf-low  { background: #FFF3ED; color: #A83800; }
.conf-none { display: none; }

/* ── Stat boxes ─────────────────────────────────────────────────────────── */
.stat-row {
    display: flex;
    gap: 16px;
    margin-bottom: 32px;
}
.stat-box {
    background: #FFFFFF;
    border: 1px solid #E5E5E2;
    border-radius: 10px;
    padding: 20px 24px;
    flex: 1;
}
.stat-value {
    font-size: 28px;
    font-weight: 600;
    color: #1A1A1A;
    letter-spacing: -0.8px;
    line-height: 1;
    margin-bottom: 4px;
}
.stat-label {
    font-size: 12px;
    color: #999;
    font-weight: 500;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}

/* ── Buttons ─────────────────────────────────────────────────────────────── */
.stButton button, .stDownloadButton button {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    border-radius: 8px !important;
    padding: 10px 24px !important;
    transition: all 0.15s !important;
    border: none !important;
}
.stButton > button[kind="primary"],
.stDownloadButton > button {
    background: #1A1A1A !important;
    color: #FFFFFF !important;
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button:hover {
    background: #333 !important;
    transform: translateY(-1px);
}
.stButton > button[kind="secondary"] {
    background: #FFFFFF !important;
    color: #1A1A1A !important;
    border: 1.5px solid #E5E5E2 !important;
}
.stButton > button[kind="secondary"]:hover {
    border-color: #1A1A1A !important;
}

/* ── Progress bar ────────────────────────────────────────────────────────── */
.stProgress > div > div {
    background: #D64000 !important;
    border-radius: 4px !important;
}
.stProgress > div {
    background: #E5E5E2 !important;
    border-radius: 4px !important;
    height: 3px !important;
}

/* ── Image display ───────────────────────────────────────────────────────── */
.preview-card {
    background: #FFFFFF;
    border: 1px solid #E5E5E2;
    border-radius: 12px;
    overflow: hidden;
    height: 100%;
}
.preview-header {
    padding: 16px 20px;
    border-bottom: 1px solid #F0F0EE;
    font-size: 12px;
    font-weight: 600;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.preview-body {
    padding: 16px;
}

/* ── Status badges ───────────────────────────────────────────────────────── */
.status-ok {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    color: #2D6A1F;
    background: #EBF5E6;
    padding: 4px 10px;
    border-radius: 20px;
}
.status-warn {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    font-size: 12px;
    font-weight: 500;
    color: #A83800;
    background: #FFF3ED;
    padding: 4px 10px;
    border-radius: 20px;
}

/* ── Batch table ─────────────────────────────────────────────────────────── */
.batch-table {
    background: #FFFFFF;
    border: 1px solid #E5E5E2;
    border-radius: 12px;
    overflow: hidden;
    margin-bottom: 24px;
}
.batch-table-header {
    display: grid;
    grid-template-columns: 2fr 1fr 2fr 1fr 1fr;
    padding: 12px 20px;
    background: #FAFAF8;
    border-bottom: 1px solid #E5E5E2;
    font-size: 11px;
    font-weight: 600;
    color: #999;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.batch-row {
    display: grid;
    grid-template-columns: 2fr 1fr 2fr 1fr 1fr;
    padding: 14px 20px;
    border-bottom: 1px solid #F5F5F3;
    font-size: 13px;
    color: #1A1A1A;
    align-items: center;
    transition: background 0.12s;
}
.batch-row:last-child { border-bottom: none; }
.batch-row:hover { background: #FAFAF8; }
.batch-filename {
    font-family: 'DM Mono', monospace;
    font-size: 12px;
    color: #555;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}

/* ── Section headings ────────────────────────────────────────────────────── */
.section-heading {
    font-size: 13px;
    font-weight: 600;
    color: #1A1A1A;
    text-transform: uppercase;
    letter-spacing: 0.5px;
    margin: 0 0 16px 0;
}

/* ── Overall confidence bar ──────────────────────────────────────────────── */
.conf-bar-wrap {
    background: #F0F0EE;
    border-radius: 4px;
    height: 4px;
    width: 100%;
    margin-top: 8px;
}
.conf-bar-fill {
    height: 4px;
    border-radius: 4px;
    background: #D64000;
    transition: width 0.4s ease;
}

/* ── Divider ─────────────────────────────────────────────────────────────── */
.divider {
    height: 1px;
    background: #E5E5E2;
    margin: 32px 0;
}

/* ── Model status dot ────────────────────────────────────────────────────── */
.model-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    display: inline-block;
    margin-right: 6px;
}
.dot-green { background: #4CAF50; }
.dot-red   { background: #E53935; }

/* ── Spinner override ────────────────────────────────────────────────────── */
.stSpinner > div {
    border-top-color: #D64000 !important;
}

/* ── Dataframe ───────────────────────────────────────────────────────────── */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    border: 1px solid #E5E5E2;
    overflow: hidden;
}

/* ── Alert ───────────────────────────────────────────────────────────────── */
.stAlert {
    border-radius: 8px !important;
    border: none !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important;
}

/* ── Padding fix ─────────────────────────────────────────────────────────── */
.block-container {
    padding-top: 40px !important;
    padding-bottom: 60px !important;
    max-width: 1100px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    import pycrfsuite
    model_pkl = Path('models/crf_model.pkl')
    if not model_pkl.exists():
        return None
    try:
        with open(model_pkl, 'rb') as f:
            meta = pickle.load(f)
        model_path = meta['model_path']
        if not Path(model_path).exists():
            model_path = str(Path('models/crf_model.crfsuite').resolve())
        tagger = pycrfsuite.Tagger()
        tagger.open(model_path)
        return tagger
    except Exception:
        return None


# ── Pipeline ──────────────────────────────────────────────────────────────────

TOTAL_KEYWORDS   = {'total', 'amount', 'jumlah', 'balance', 'due',
                    'payable', 'subtotal', 'grand', 'net', 'bil'}
DATE_KEYWORDS    = {'date', 'tarikh', 'dated', 'invoice', 'issued'}
ADDRESS_KEYWORDS = {'address', 'addr', 'no.', 'jalan', 'street', 'road',
                    'avenue', 'lane', 'taman', 'lorong'}
COMPANY_SUFFIXES = {'sdn', 'bhd', 'ltd', 'llc', 'inc', 'corp',
                    'pty', 'plc', 'pte', 'gmbh', 'berhad'}
SKIP_KEYWORDS    = {'cash', 'change', 'tender', 'rounding', 'round',
                    'disc', 'discount', 'deposit', 'payment', 'paid'}

def _bin(v, n): return min(int(v * n), n - 1)
def _near(toks, i, kws, w=4):
    for j in range(max(0,i-w), min(len(toks),i+w+1)):
        if j!=i and toks[j]['text'].lower().strip() in kws: return True
    return False
def _prev_kw(toks, i, w=6):
    for j in range(max(0,i-w), i):
        t = toks[j]['text'].lower()
        if t.strip() in TOTAL_KEYWORDS or any(k in t for k in TOTAL_KEYWORDS): return True
    return False
def _next_kw(toks, i, w=5):
    for j in range(i+1, min(len(toks),i+w+1)):
        t = toks[j]['text'].lower()
        if t.strip() in TOTAL_KEYWORDS or any(k in t for k in TOTAL_KEYWORDS): return True
    return False
def _near_skip(toks, i, w=4): return _near(toks, i, SKIP_KEYWORDS, w)
def _same_before(toks, i):
    def norm(t):
        t = re.sub(r'(?i)(rm|myr|\$|£|€)','',t).strip()
        try: return str(round(float(t.replace(',','')),2))
        except: return t
    cur = norm(toks[i]['text']); c = 0
    for j in range(i):
        if norm(toks[j]['text'])==cur: c+=1
    return c
def _same_after(toks, i):
    def norm(t):
        t = re.sub(r'(?i)(rm|myr|\$|£|€)','',t).strip()
        try: return str(round(float(t.replace(',','')),2))
        except: return t
    cur = norm(toks[i]['text']); c = 0
    for j in range(i+1,len(toks)):
        if norm(toks[j]['text'])==cur: c+=1
    return c

def word_features(toks, i):
    t=toks[i]; text=t['text'].strip(); text_l=text.lower()
    x=t.get('x_norm',0.5); y=t.get('y_norm',0.5)
    is_amount=bool(re.match(r'^(?:rm|myr|usd|sgd|gbp|eur|inr|\$|£|€|₹)?\s*[\d,]+\.\d{1,2}$',text,re.IGNORECASE))
    sb=_same_before(toks,i) if is_amount else 0
    sa=_same_after(toks,i)  if is_amount else 0
    feats=['bias',
        f'word={text_l}',f'word_len={len(text)}',
        f'prefix2={text_l[:2]}',f'prefix3={text_l[:3]}',
        f'suffix2={text_l[-2:]}',f'suffix3={text_l[-3:]}',
        f'is_upper={text.isupper()}',f'is_digit={text.isdigit()}',
        f'is_alnum={text.isalnum()}',
        f'has_digit={any(c.isdigit() for c in text)}',
        f'has_special={any(not c.isalnum() and c!=" " for c in text)}',
        f'is_amount={is_amount}',
        f'has_currency={any(c in text.upper() for c in ["$","£","€","₹","RM"])}',
        f'is_amount_with_currency={bool(re.match(r"^(?:rm|myr|\$|£|€)[\d,]+",text,re.IGNORECASE))}',
        f'is_company_suffix={text_l in COMPANY_SUFFIXES}',
        f'is_total_kw={text_l in TOTAL_KEYWORDS}',
        f'is_date_kw={text_l in DATE_KEYWORDS}',
        f'is_skip_kw={text_l in SKIP_KEYWORDS}',
        f'like_date={bool(re.match(r".*\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}.*",text))}',
        f'x_bin={_bin(x,5)}',f'y_bin={_bin(y,10)}',
        f'in_top_20={y<0.20}',f'in_top_40={y<0.40}',
        f'in_bottom_30={y>0.70}',f'in_bottom_20={y>0.80}',
        f'on_right={x>0.55}',f'on_far_right={x>0.75}',f'on_left={x<0.30}',
        f'near_total={_near(toks,i,TOTAL_KEYWORDS)}',
        f'near_date={_near(toks,i,DATE_KEYWORDS)}',
        f'near_address={_near(toks,i,ADDRESS_KEYWORDS)}',
        f'near_suffix={_near(toks,i,COMPANY_SUFFIXES)}',
        f'prev_is_total_kw={_prev_kw(toks,i)}',
        f'next_is_total_kw={_next_kw(toks,i)}',
        f'near_skip={_near_skip(toks,i)}',
        f'amount_on_right={is_amount and x>0.55}',
        f'amount_in_bottom={is_amount and y>0.55}',
        f'amount_near_total={is_amount and _near(toks,i,TOTAL_KEYWORDS,6)}',
        f'amount_prev_total={is_amount and _prev_kw(toks,i)}',
        f'amount_not_near_skip={is_amount and not _near_skip(toks,i)}',
        f'same_before={sb}',f'same_after={sa}',
        f'is_first_occurrence={sb==0}',f'is_last_occurrence={sa==0}',
        f'has_duplicates={sb+sa>0}',
        f'amount_is_last={is_amount and sa==0}',
        f'amount_after_total_kw_and_last={is_amount and _prev_kw(toks,i) and sa==0}',
        f'amount_far_right_bottom={is_amount and x>0.65 and y>0.65}',
        f'amount_far_right_bottom_no_skip={is_amount and x>0.65 and y>0.65 and not _near_skip(toks,i,5)}',
        f'rm_amount_bottom={bool(re.match(r"^(?:rm|myr)[\d,]+\.\d{1,2}$",text,re.IGNORECASE)) and y>0.60}',
        f'prev_token_contains_total={any(k in toks[i-1]["text"].lower() for k in TOTAL_KEYWORDS) if i>0 else False}',
        f'prev2_token_contains_total={any(k in toks[i-2]["text"].lower() for k in TOTAL_KEYWORDS) if i>1 else False}',
        f'prev3_token_contains_total={any(k in toks[i-3]["text"].lower() for k in TOTAL_KEYWORDS) if i>2 else False}',
        f'next_token_contains_total={any(k in toks[i+1]["text"].lower() for k in TOTAL_KEYWORDS) if i<len(toks)-1 else False}',
        f'next2_token_contains_total={any(k in toks[i+2]["text"].lower() for k in TOTAL_KEYWORDS) if i<len(toks)-2 else False}',
        f'next3_token_contains_total={any(k in toks[i+3]["text"].lower() for k in TOTAL_KEYWORDS) if i<len(toks)-3 else False}',
        f'near_cash_label={any(toks[j]["text"].lower().strip() in {"cash","change"} and toks[j].get("x_norm",0)<0.40 for j in range(max(0,i-5),min(len(toks),i+5)) if j!=i)}',
        f'near_cash_value={any(toks[j]["text"].lower().strip() in {"cash","change"} and toks[j].get("x_norm",0)>=0.40 for j in range(max(0,i-5),min(len(toks),i+5)) if j!=i)}',
    ]
    if i>0:
        pt=toks[i-1]; pt_l=pt['text'].lower().strip()
        feats+=[f'prev={pt_l}',f'prev_is_upper={pt["text"].isupper()}',
                f'prev_is_digit={pt["text"].isdigit()}',
                f'prev_is_total_kw2={pt_l in TOTAL_KEYWORDS}',
                f'prev_is_suffix={pt_l in COMPANY_SUFFIXES}',
                f'prev_is_skip={pt_l in SKIP_KEYWORDS}',
                f'prev_is_zero={pt["text"].strip() in {"0.00","0.0","0"}}']
    else: feats.append('BOS')
    if i>1:
        ppt_l=toks[i-2]['text'].lower().strip()
        feats+=[f'prev2={ppt_l}',f'prev2_is_total_kw={ppt_l in TOTAL_KEYWORDS}',
                f'prev2_is_skip={ppt_l in SKIP_KEYWORDS}',
                f'prev2_is_zero={toks[i-2]["text"].strip() in {"0.00","0.0","0"}}']
    else: feats.append('BOS2')
    if i>2: feats.append(f'prev3_is_total_kw={toks[i-3]["text"].lower().strip() in TOTAL_KEYWORDS}')
    if i<len(toks)-1:
        nt=toks[i+1]; nt_l=nt['text'].lower().strip()
        feats+=[f'next={nt_l}',f'next_is_upper={nt["text"].isupper()}',
                f'next_is_digit={nt["text"].isdigit()}',
                f'next_is_total_kw2={nt_l in TOTAL_KEYWORDS}',
                f'next_is_suffix={nt_l in COMPANY_SUFFIXES}',
                f'next_is_skip={nt_l in SKIP_KEYWORDS}',
                f'next_is_zero={nt["text"].strip() in {"0.00","0.0","0"}}']
    else: feats.append('EOS')
    if i<len(toks)-2:
        nnt_l=toks[i+2]['text'].lower().strip()
        feats+=[f'next2={nnt_l}',f'next2_is_total_kw={nnt_l in TOTAL_KEYWORDS}',
                f'next2_is_skip={nnt_l in SKIP_KEYWORDS}',
                f'next2_is_zero={toks[i+2]["text"].strip() in {"0.00","0.0","0"}}']
    else: feats.append('EOS2')
    if i<len(toks)-3:
        feats+=[f'next3_is_skip={toks[i+3]["text"].lower().strip() in SKIP_KEYWORDS}',
                f'next3_is_zero={toks[i+3]["text"].strip() in {"0.00","0.0","0"}}']
    return feats


def run_pipeline(image_path: Path, tagger) -> dict:
    from src.ocr_engine import extract_tokens
    from src.parser     import parse_fields

    tokens        = extract_tokens(str(image_path))
    parser_fields = parse_fields(tokens)
    crf_entities  = {}
    crf_conf      = {}

    if tagger is not None:
        features = [word_features(tokens, i) for i in range(len(tokens))]
        labels   = tagger.tag(features)
        tagger.set(features)
        probs    = [tagger.marginal(l, i) for i, l in enumerate(labels)]

        spans     = {'company': [], 'date': [], 'address': [], 'total': []}
        conf_sums = {'company': [], 'date': [], 'address': [], 'total': []}

        for tok, label, prob in zip(tokens, labels, probs):
            if label == 'O': continue
            parts = label.split('-', 1)
            if len(parts) != 2: continue
            entity = parts[1].lower()
            if entity not in spans: continue
            spans[entity].append(tok['text'])
            conf_sums[entity].append(round(prob, 3))

        for key in spans:
            crf_entities[key] = ' '.join(spans[key]).strip()
            crf_conf[key]     = round(
                sum(conf_sums[key])/len(conf_sums[key]), 3
            ) if conf_sums[key] else 0.0

    parser_conf = parser_fields.pop('_confidence', {})
    merged      = dict(parser_fields)

    for key in ['company', 'date', 'address', 'total']:
        crf_val    = crf_entities.get(key, '')
        parser_val = parser_fields.get(key, '')
        crf_c      = crf_conf.get(key, 0.0)
        parser_c   = parser_conf.get(key, 0.0)
        if crf_val and parser_val:
            merged[key] = crf_val if crf_c >= parser_c else parser_val
        elif crf_val:
            merged[key] = crf_val
        elif parser_val:
            merged[key] = parser_val

    all_conf = {}
    for k in set(list(parser_conf.keys()) + list(crf_conf.keys())):
        all_conf[k] = max(parser_conf.get(k, 0.0), crf_conf.get(k, 0.0))

    merged['_confidence'] = all_conf
    merged['_method']     = 'parser + crf' if tagger else 'parser only'
    return merged


# ── UI helpers ────────────────────────────────────────────────────────────────

FIELDS = [
    ('invoice_number', 'Invoice No.'),
    ('date',           'Date'),
    ('due_date',       'Due Date'),
    ('vendor_name',    'Vendor'),
    ('company',        'Company'),
    ('address',        'Address'),
    ('total',          'Total'),
    ('tax',            'Tax'),
    ('tax_rate',       'Tax Rate'),
    ('currency',       'Currency'),
    ('email',          'Email'),
    ('phone',          'Phone'),
]


def render_fields(fields: dict):
    conf   = fields.get('_confidence', {})
    html   = '<div class="result-card">'
    html  += '<div class="result-card-header"><div class="result-card-title">Extracted Fields</div></div>'
    html  += '<div class="result-card-body">'

    for key, label in FIELDS:
        val = str(fields.get(key, '')).strip()
        c   = conf.get(key, 0.0)

        if c >= 0.85:
            badge_cls = 'conf-high'
            badge_txt = f'{c:.0%}'
        elif c >= 0.60:
            badge_cls = 'conf-mid'
            badge_txt = f'{c:.0%}'
        elif c > 0:
            badge_cls = 'conf-low'
            badge_txt = f'{c:.0%}'
        else:
            badge_cls = 'conf-none'
            badge_txt = ''

        val_html = (
            f'<span class="field-value">{val}</span>'
            if val else
            '<span class="field-value empty">Not found</span>'
        )

        html += f'''
        <div class="field-row">
            <div class="field-label">{label}</div>
            {val_html}
            <span class="field-conf {badge_cls}">{badge_txt}</span>
        </div>'''

    html += '</div></div>'
    st.markdown(html, unsafe_allow_html=True)


def overall_conf(fields: dict) -> float:
    conf     = fields.get('_confidence', {})
    non_zero = [v for v in conf.values() if v > 0]
    return round(sum(non_zero) / len(non_zero), 2) if non_zero else 0.0


def get_excel_bytes(results, source_files):
    from src.exporter import export_batch
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as tmp:
        tmp_path = tmp.name
    export_batch(results, tmp_path, source_files=source_files)
    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


def get_csv_bytes(results, source_files):
    from src.exporter import export_csv
    with tempfile.NamedTemporaryFile(suffix='.csv', delete=False) as tmp:
        tmp_path = tmp.name
    export_csv(results, tmp_path, source_files=source_files)
    with open(tmp_path, 'rb') as f:
        data = f.read()
    os.unlink(tmp_path)
    return data


# ── App ───────────────────────────────────────────────────────────────────────

def main():
    tagger = load_model()

    # Nav bar
    model_status = (
        '<span class="status-ok"><span class="model-dot dot-green"></span>Model ready</span>'
        if tagger else
        '<span class="status-warn"><span class="model-dot dot-red"></span>Model not found</span>'
    )
    st.markdown(f'''
    <div class="nav-bar">
        <div class="nav-logo">Invoice<span>OCR</span></div>
        <div style="display:flex;align-items:center;gap:20px;">
            {model_status}
            <span class="nav-tag">SROIE 2019 · CRF · 0.81 F1</span>
        </div>
    </div>
    ''', unsafe_allow_html=True)

    # Hero
    st.markdown('''
    <div class="hero">
        <h1>Extract data from<br><em>any invoice</em> instantly</h1>
        <p>Upload a scanned receipt or invoice image.<br>
        Get structured fields — vendor, date, total, address — in seconds.</p>
    </div>
    ''', unsafe_allow_html=True)

    # Tabs
    tab1, tab2 = st.tabs(['Single Invoice', 'Batch Processing'])

    # ── Single ────────────────────────────────────────────────────────────────
    with tab1:
        uploaded = st.file_uploader(
            'Drop your invoice here, or click to browse',
            type=['jpg', 'jpeg', 'png', 'pdf'],
            label_visibility='collapsed',
        )

        if uploaded is None:
            st.markdown('''
            <div style="text-align:center;padding:16px 0 0 0;">
                <span style="font-size:13px;color:#BBB;">
                    Supports JPG, PNG, PDF &nbsp;·&nbsp; Max 200MB
                </span>
            </div>
            ''', unsafe_allow_html=True)

        if uploaded is not None:
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

            col_img, col_res = st.columns([1, 1.4], gap='large')

            with col_img:
                st.markdown('''
                <div class="preview-card">
                    <div class="preview-header">Preview</div>
                </div>
                ''', unsafe_allow_html=True)
                if uploaded.type != 'application/pdf':
                    st.image(uploaded, use_container_width=True)
                else:
                    st.info('PDF uploaded. Preview not available for PDFs.')

            with col_res:
                with st.spinner('Extracting fields...'):
                    suffix = Path(uploaded.name).suffix
                    with tempfile.NamedTemporaryFile(
                            suffix=suffix, delete=False) as tmp:
                        tmp.write(uploaded.read())
                        tmp_path = Path(tmp.name)
                    try:
                        fields  = run_pipeline(tmp_path, tagger)
                        overall = overall_conf(fields)

                        # Stats row
                        filled = sum(
                            1 for k, _ in FIELDS
                            if fields.get(k, '').strip()
                        )
                        st.markdown(f'''
                        <div class="stat-row">
                            <div class="stat-box">
                                <div class="stat-value">{filled}</div>
                                <div class="stat-label">Fields found</div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value">{overall:.0%}</div>
                                <div class="stat-label">Avg confidence</div>
                                <div class="conf-bar-wrap">
                                    <div class="conf-bar-fill" style="width:{overall*100:.0f}%"></div>
                                </div>
                            </div>
                            <div class="stat-box">
                                <div class="stat-value" style="font-size:14px;letter-spacing:0;">{fields.get("_method","—")}</div>
                                <div class="stat-label">Method</div>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)

                        render_fields(fields)

                        # Downloads
                        st.markdown('<div style="height:20px"></div>',
                                    unsafe_allow_html=True)
                        dc1, dc2, _ = st.columns([1, 1, 1.5])
                        with dc1:
                            st.download_button(
                                'Download Excel',
                                data=get_excel_bytes([fields], [uploaded.name]),
                                file_name=f'{Path(uploaded.name).stem}_result.xlsx',
                                mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                                use_container_width=True,
                            )
                        with dc2:
                            st.download_button(
                                'Download CSV',
                                data=get_csv_bytes([fields], [uploaded.name]),
                                file_name=f'{Path(uploaded.name).stem}_result.csv',
                                mime='text/csv',
                                use_container_width=True,
                            )

                    except Exception as e:
                        st.error(f'Extraction failed: {e}')
                    finally:
                        if tmp_path.exists():
                            os.unlink(tmp_path)

    # ── Batch ─────────────────────────────────────────────────────────────────
    with tab2:
        st.markdown('''
        <p style="color:#666;font-size:15px;margin:0 0 28px 0;">
            Upload multiple invoices and download all results in one file.
            Each invoice is processed independently.
        </p>
        ''', unsafe_allow_html=True)

        uploaded_files = st.file_uploader(
            'Drop your invoices here, or click to browse',
            type=['jpg', 'jpeg', 'png', 'pdf'],
            accept_multiple_files=True,
            label_visibility='collapsed',
        )

        if uploaded_files:
            n = len(uploaded_files)
            st.markdown(f'''
            <p style="font-size:13px;color:#999;margin:12px 0 24px 0;">
                {n} file{"s" if n>1 else ""} selected
            </p>
            ''', unsafe_allow_html=True)

            if st.button('Extract all invoices', type='primary'):
                results      = []
                source_files = []
                errors       = []

                prog    = st.progress(0)
                status  = st.empty()

                for i, f in enumerate(uploaded_files):
                    status.markdown(
                        f'<p style="font-size:13px;color:#666;">'
                        f'Processing {f.name} &nbsp; ({i+1} of {n})</p>',
                        unsafe_allow_html=True
                    )
                    suffix = Path(f.name).suffix
                    with tempfile.NamedTemporaryFile(
                            suffix=suffix, delete=False) as tmp:
                        tmp.write(f.read())
                        tmp_path = Path(tmp.name)
                    try:
                        fields = run_pipeline(tmp_path, tagger)
                        results.append(fields)
                        source_files.append(f.name)
                    except Exception as e:
                        errors.append(f'{f.name}: {e}')
                    finally:
                        if tmp_path.exists():
                            os.unlink(tmp_path)
                    prog.progress((i + 1) / n)

                status.empty()
                prog.empty()

                if errors:
                    st.warning(f'{len(errors)} file(s) failed to process.')

                if results:
                    st.markdown('<div class="divider"></div>',
                                unsafe_allow_html=True)

                    # Summary stats
                    avg_conf = sum(overall_conf(r) for r in results) / len(results)
                    st.markdown(f'''
                    <div class="stat-row">
                        <div class="stat-box">
                            <div class="stat-value">{len(results)}</div>
                            <div class="stat-label">Invoices processed</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{avg_conf:.0%}</div>
                            <div class="stat-label">Avg confidence</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-value">{len(errors)}</div>
                            <div class="stat-label">Errors</div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)

                    # Results table
                    st.markdown(
                        '<div class="section-heading">Results</div>',
                        unsafe_allow_html=True
                    )

                    table_rows = []
                    for fname, f in zip(source_files, results):
                        table_rows.append({
                            'File'      : fname,
                            'Date'      : f.get('date', '—') or '—',
                            'Vendor'    : (f.get('vendor_name') or f.get('company') or '—'),
                            'Total'     : f.get('total', '—') or '—',
                            'Confidence': f'{overall_conf(f):.0%}',
                        })

                    st.dataframe(
                        pd.DataFrame(table_rows),
                        use_container_width=True,
                        hide_index=True,
                        height=min(400, 56 + 35 * len(table_rows)),
                    )

                    # Downloads
                    st.markdown('<div style="height:8px"></div>',
                                unsafe_allow_html=True)
                    ts   = datetime.now().strftime('%Y%m%d_%H%M')
                    dc1, dc2, _ = st.columns([1, 1, 1.5])
                    with dc1:
                        st.download_button(
                            f'Download Excel ({len(results)} invoices)',
                            data=get_excel_bytes(results, source_files),
                            file_name=f'invoices_{ts}.xlsx',
                            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
                            use_container_width=True,
                        )
                    with dc2:
                        st.download_button(
                            f'Download CSV ({len(results)} invoices)',
                            data=get_csv_bytes(results, source_files),
                            file_name=f'invoices_{ts}.csv',
                            mime='text/csv',
                            use_container_width=True,
                        )


if __name__ == '__main__':
    main()
