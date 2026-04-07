"""
src/model.py
============
CRF (Conditional Random Field) model for token-level entity labelling.

Improvements over v1:
  - Keyword proximity features (near_total_kw, near_date_kw, etc.)
  - Stronger positional binning
  - Tuned hyperparameters (c1=0.01, c2=0.1, max_iter=300)
  - Windows-safe temp path for training
  - Better entity extraction with span merging

Usage:
    from src.model import train_crf, save_model, load_model, predict

    # Training (in notebook)
    corpus = pd.read_csv('data/processed/labelled_corpus.csv')
    tagger = train_crf(corpus)
    save_model(tagger, 'models/crf_model.pkl')

    # Inference (in app.py)
    tagger = load_model('models/crf_model.pkl')
    tokens = extract_tokens('invoice.jpg')
    result = predict(tagger, tokens)
"""

import re
import os
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pycrfsuite


# ── Keyword sets ──────────────────────────────────────────────────────────────
# Used by _near_keyword() to detect proximity to trigger words

TOTAL_KEYWORDS   = {'total', 'amount', 'jumlah', 'balance', 'due',
                    'payable', 'subtotal', 'grand', 'net', 'bil'}
DATE_KEYWORDS    = {'date', 'tarikh', 'dated', 'invoice', 'issued'}
ADDRESS_KEYWORDS = {'address', 'addr', 'no.', 'jalan', 'street', 'road',
                    'avenue', 'lane', 'taman', 'lorong', 'apt', 'suite'}
COMPANY_KEYWORDS = {'from', 'vendor', 'company', 'supplier', 'sold',
                    'billed', 'by', 'to'}
COMPANY_SUFFIXES = {'sdn', 'bhd', 'ltd', 'llc', 'inc', 'corp', 'co',
                    'pty', 'plc', 'pte', 'gmbh', 'berhad', 'enterprise'}


# ── Helper functions ──────────────────────────────────────────────────────────

def _bin(value: float, n_bins: int) -> int:
    """Discretise a 0.0-1.0 float into n_bins integer bins."""
    return min(int(value * n_bins), n_bins - 1)


def _near_keyword(tokens: list, i: int, keywords: set, window: int = 4) -> bool:
    """
    Check if any keyword appears within `window` tokens of position i.
    Used to give the CRF context about what field a token is near.
    """
    start = max(0, i - window)
    end   = min(len(tokens), i + window + 1)
    for j in range(start, end):
        if j == i:
            continue
        if tokens[j]['text'].lower().strip() in keywords:
            return True
    return False


def _find_total_keyword_idx(tokens: list) -> int:
    """
    Find the index of the token most likely to be a TOTAL keyword.
    Returns -1 if not found.
    """
    for idx, t in enumerate(tokens):
        if t['text'].lower().strip() in TOTAL_KEYWORDS:
            return idx
    return -1


# ── Feature engineering ───────────────────────────────────────────────────────

def _word_features(tokens: list, i: int) -> list:
    """
    Compute features for token at position i.
    Looks at neighbours (i-2, i-1, i+1, i+2) for context.
    """
    t      = tokens[i]
    text   = t['text'].strip()
    text_l = text.lower()

    x_norm = t.get('x_norm', 0.5)
    y_norm = t.get('y_norm', 0.5)

    features = [
        'bias',

        # ── Text features ──────────────────────────────────────────────────
        f'word={text_l}',
        f'word_upper={text.upper()}',
        f'word_len={len(text)}',
        f'prefix2={text_l[:2]}',
        f'prefix3={text_l[:3]}',
        f'suffix2={text_l[-2:]}',
        f'suffix3={text_l[-3:]}',

        # ── Shape features ─────────────────────────────────────────────────
        f'is_upper={text.isupper()}',
        f'is_lower={text.islower()}',
        f'is_title={text.istitle()}',
        f'is_digit={text.isdigit()}',
        f'is_alnum={text.isalnum()}',
        f'has_digit={any(c.isdigit() for c in text)}',
        f'has_upper={any(c.isupper() for c in text)}',
        f'has_special={any(not c.isalnum() and c != " " for c in text)}',
        f'char_count={min(len(text), 30)}',

        # ── Regex / domain flags ───────────────────────────────────────────
        f'like_date={bool(re.match(r"^\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4}$", text))}',
        f'like_amount={bool(re.match(r"^\$?[\d,]+\.?\d{0,2}$", text))}',
        f'like_phone={bool(re.match(r"^\+?[\d\s\-\(\)]{7,}$", text))}',
        f'like_email={bool(re.match(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", text))}',
        f'like_invoice={bool(re.match(r"^[A-Z]{2,4}[\-\/]\w+$", text))}',
        f'has_currency={any(c in text for c in ["$", "£", "€", "₹", "RM"])}',
        f'is_company_suffix={text_l in COMPANY_SUFFIXES}',
        f'is_total_kw={text_l in TOTAL_KEYWORDS}',
        f'is_date_kw={text_l in DATE_KEYWORDS}',
        f'is_address_kw={text_l in ADDRESS_KEYWORDS}',

        # ── Position features (coarse bins) ───────────────────────────────
        f'x_bin={_bin(x_norm, 5)}',
        f'y_bin={_bin(y_norm, 10)}',
        f'x_center_bin={_bin(t.get("x_center_norm", x_norm), 5)}',
        f'y_center_bin={_bin(t.get("y_center_norm", y_norm), 10)}',
        f'width_bin={_bin(t.get("width_norm", 0.1), 5)}',

        # ── Position features (fine-grained) ──────────────────────────────
        f'in_top_10={y_norm < 0.10}',
        f'in_top_20={y_norm < 0.20}',
        f'in_top_40={y_norm < 0.40}',
        f'in_mid={0.40 <= y_norm < 0.70}',
        f'in_bottom_30={y_norm > 0.70}',
        f'in_bottom_20={y_norm > 0.80}',
        f'on_far_right={x_norm > 0.75}',
        f'on_right={x_norm > 0.55}',
        f'on_left={x_norm < 0.30}',
        f'on_far_left={x_norm < 0.15}',
        f'is_centered={0.35 < x_norm < 0.65}',

        # ── Keyword proximity features (NEW) ──────────────────────────────
        # Is a trigger word within 4 tokens of this one?
        f'near_total={_near_keyword(tokens, i, TOTAL_KEYWORDS)}',
        f'near_date={_near_keyword(tokens, i, DATE_KEYWORDS)}',
        f'near_address={_near_keyword(tokens, i, ADDRESS_KEYWORDS)}',
        f'near_company={_near_keyword(tokens, i, COMPANY_KEYWORDS)}',
        f'near_suffix={_near_keyword(tokens, i, COMPANY_SUFFIXES)}',
    ]

    # ── Previous token (i-1) ──────────────────────────────────────────────
    if i > 0:
        pt   = tokens[i - 1]
        pt_l = pt['text'].lower().strip()
        features += [
            f'prev_word={pt_l}',
            f'prev_is_upper={pt["text"].isupper()}',
            f'prev_is_digit={pt["text"].isdigit()}',
            f'prev_like_amount={bool(re.match(r"^\$?[\d,]+\.?\d{0,2}$", pt["text"]))}',
            f'prev_is_total_kw={pt_l in TOTAL_KEYWORDS}',
            f'prev_is_date_kw={pt_l in DATE_KEYWORDS}',
            f'prev_is_suffix={pt_l in COMPANY_SUFFIXES}',
        ]
    else:
        features.append('BOS')

    # ── Two tokens back (i-2) ─────────────────────────────────────────────
    if i > 1:
        ppt_l = tokens[i - 2]['text'].lower().strip()
        features += [
            f'prev2_word={ppt_l}',
            f'prev2_is_total_kw={ppt_l in TOTAL_KEYWORDS}',
        ]

    # ── Next token (i+1) ─────────────────────────────────────────────────
    if i < len(tokens) - 1:
        nt   = tokens[i + 1]
        nt_l = nt['text'].lower().strip()
        features += [
            f'next_word={nt_l}',
            f'next_is_upper={nt["text"].isupper()}',
            f'next_is_digit={nt["text"].isdigit()}',
            f'next_is_total_kw={nt_l in TOTAL_KEYWORDS}',
            f'next_is_suffix={nt_l in COMPANY_SUFFIXES}',
            f'next_like_amount={bool(re.match(r"^\$?[\d,]+\.?\d{0,2}$", nt["text"]))}',
        ]
    else:
        features.append('EOS')

    # ── Two tokens ahead (i+2) ───────────────────────────────────────────
    if i < len(tokens) - 2:
        nnt_l = tokens[i + 2]['text'].lower().strip()
        features += [
            f'next2_word={nnt_l}',
            f'next2_is_total_kw={nnt_l in TOTAL_KEYWORDS}',
        ]

    return features


def build_features(tokens: list) -> list:
    """
    Build feature list for an entire receipt.
    Returns list of feature lists — one per token.
    """
    return [_word_features(tokens, i) for i in range(len(tokens))]


# ── Corpus preparation ────────────────────────────────────────────────────────

def corpus_to_sequences(corpus_df: pd.DataFrame) -> tuple:
    """
    Convert labelled_corpus.csv into pycrfsuite training sequences.
    Returns (X, y) where X = features, y = labels.
    """
    X = []
    y = []

    for stem, group in corpus_df.groupby('stem'):
        tokens = group.to_dict('records')
        X.append(build_features(tokens))
        y.append(group['label'].tolist())

    return X, y


# ── Training ──────────────────────────────────────────────────────────────────

def train_crf(
    corpus_df : pd.DataFrame,
    c1        : float = 0.01,
    c2        : float = 0.10,
    max_iter  : int   = 300,
    verbose   : bool  = True,
) -> pycrfsuite.Tagger:
    """
    Train a CRF model on the labelled corpus.

    Args:
        corpus_df : DataFrame from labelled_corpus.csv
        c1        : L1 regularisation (sparse features)
        c2        : L2 regularisation (prevent overfitting)
        max_iter  : number of training iterations
        verbose   : print training progress

    Returns:
        Trained pycrfsuite.Tagger ready for prediction.
    """
    # Add default normalised coords if missing
    if 'x_norm' not in corpus_df.columns:
        corpus_df = corpus_df.copy()
        corpus_df['x_norm']        = 0.5
        corpus_df['y_norm']        = 0.5
        corpus_df['x_center_norm'] = 0.5
        corpus_df['y_center_norm'] = 0.5
        corpus_df['width_norm']    = 0.1
        corpus_df['height_norm']   = 0.03

    print('Preparing training sequences...')
    X_train, y_train = corpus_to_sequences(corpus_df)

    n_tokens = sum(len(s) for s in y_train)
    print(f'  Receipts : {len(X_train)}')
    print(f'  Tokens   : {n_tokens:,}')

    from collections import Counter
    all_labels = [l for seq in y_train for l in seq]
    label_counts = dict(Counter(all_labels).most_common())
    print(f'  Labels   : {label_counts}')

    # Set up trainer
    trainer = pycrfsuite.Trainer(verbose=verbose)
    for x_seq, y_seq in zip(X_train, y_train):
        trainer.append(x_seq, y_seq)

    trainer.set_params({
        'c1'                           : c1,
        'c2'                           : c2,
        'max_iterations'               : max_iter,
        'feature.possible_transitions' : True,
        'feature.possible_states'      : True,
    })

    # Windows-safe temp file (no /tmp/ on Windows)
    model_tmp = os.path.join(tempfile.gettempdir(), 'crf_tmp.model')
    print(f'\nTraining CRF (c1={c1}, c2={c2}, max_iter={max_iter})...')
    trainer.train(model_tmp)

    tagger = pycrfsuite.Tagger()
    tagger.open(model_tmp)

    print('Training complete.')
    return tagger


# ── Save / Load ───────────────────────────────────────────────────────────────

def save_model(tagger: pycrfsuite.Tagger, path: str) -> None:
    """Save trained CRF tagger to disk."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    model_path = str(path).replace('.pkl', '.crfsuite')
    tagger.dump(model_path)

    meta = {'model_path': model_path}
    with open(path, 'wb') as f:
        pickle.dump(meta, f)

    size_kb = Path(model_path).stat().st_size / 1024
    print(f'Saved: {model_path}  ({size_kb:.0f} KB)')


def load_model(path: str) -> pycrfsuite.Tagger:
    """Load a saved CRF model from disk."""
    with open(path, 'rb') as f:
        meta = pickle.load(f)

    tagger = pycrfsuite.Tagger()
    tagger.open(meta['model_path'])
    return tagger


# ── Prediction ────────────────────────────────────────────────────────────────

def predict(tagger: pycrfsuite.Tagger, tokens: list) -> list:
    """
    Predict BIO labels for a list of tokens.
    Returns same token list with 'label' and 'label_conf' added.
    """
    if not tokens:
        return tokens

    features = build_features(tokens)
    labels   = tagger.tag(features)

    # Marginal probabilities for confidence scores
    tagger.set(features)
    probs = [tagger.marginal(label, i) for i, label in enumerate(labels)]

    result = []
    for token, label, prob in zip(tokens, labels, probs):
        t               = dict(token)
        t['label']      = label
        t['label_conf'] = round(prob, 3)
        result.append(t)

    return result


# ── Entity extraction ─────────────────────────────────────────────────────────

def extract_entities(predicted_tokens: list) -> dict:
    """
    Convert BIO-labelled tokens into clean entity strings.

    B-COMPANY I-COMPANY I-COMPANY -> 'BOOK TA .K SDN BHD'

    Returns dict with: company, date, address, total, _crf_confidence
    """
    spans      = {k: [] for k in ['company', 'date', 'address', 'total']}
    conf_sums  = {k: [] for k in ['company', 'date', 'address', 'total']}

    for t in predicted_tokens:
        label = t.get('label', 'O')
        if label == 'O':
            continue

        parts = label.split('-', 1)
        if len(parts) != 2:
            continue

        prefix, entity = parts
        entity = entity.lower()
        if entity not in spans:
            continue

        spans[entity].append(t['text'])
        conf_sums[entity].append(t.get('label_conf', 0.0))

    result   = {}
    crf_conf = {}

    for key in spans:
        text          = ' '.join(spans[key]).strip()
        result[key]   = text
        crf_conf[key] = round(
            sum(conf_sums[key]) / len(conf_sums[key]), 3
        ) if conf_sums[key] else 0.0

    result['_crf_confidence'] = crf_conf
    return result


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate(tagger: pycrfsuite.Tagger, corpus_df: pd.DataFrame) -> pd.DataFrame:
    """
    Evaluate CRF on a labelled corpus. Returns F1 per label.
    """
    from sklearn.metrics import classification_report

    # Add coords if missing
    if 'x_norm' not in corpus_df.columns:
        corpus_df = corpus_df.copy()
        corpus_df['x_norm']        = 0.5
        corpus_df['y_norm']        = 0.5
        corpus_df['x_center_norm'] = 0.5
        corpus_df['y_center_norm'] = 0.5
        corpus_df['width_norm']    = 0.1
        corpus_df['height_norm']   = 0.03

    X, y_true_seqs = corpus_to_sequences(corpus_df)

    y_true = []
    y_pred = []

    for x_seq, y_seq in zip(X, y_true_seqs):
        y_true.extend(y_seq)
        y_pred.extend(tagger.tag(x_seq))

    labels = sorted(set(y_true + y_pred) - {'O'})
    report = classification_report(
        y_true, y_pred,
        labels=labels,
        output_dict=True,
        zero_division=0
    )

    rows = []
    for label, metrics in report.items():
        if label in ('accuracy', 'macro avg', 'weighted avg'):
            continue
        rows.append({
            'label'    : label,
            'precision': round(metrics['precision'], 3),
            'recall'   : round(metrics['recall'],    3),
            'f1'       : round(metrics['f1-score'],  3),
            'support'  : int(metrics['support']),
        })

    wa = report.get('weighted avg', {})
    rows.append({
        'label'    : 'OVERALL (weighted)',
        'precision': round(wa.get('precision', 0), 3),
        'recall'   : round(wa.get('recall',    0), 3),
        'f1'       : round(wa.get('f1-score',  0), 3),
        'support'  : int(wa.get('support',     0)),
    })

    return pd.DataFrame(rows)


# ── Feature importance ────────────────────────────────────────────────────────

def top_features(tagger: pycrfsuite.Tagger, n: int = 20) -> pd.DataFrame:
    """Show most important features learned by the CRF."""
    info = tagger.info()
    rows = []

    for (attr, label), weight in info.transitions.items():
        rows.append({'type': 'transition', 'feature': f'{attr} -> {label}', 'weight': weight})

    for (attr, label), weight in info.state_features.items():
        rows.append({'type': 'state', 'feature': f'{attr} | {label}', 'weight': weight})

    df = pd.DataFrame(rows)
    df = df.reindex(df['weight'].abs().sort_values(ascending=False).index)
    return df.head(n).reset_index(drop=True)


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')

    corpus_path = Path('data/processed/labelled_corpus.csv')

    if not corpus_path.exists():
        print('labelled_corpus.csv not found. Run Phase 1 first.')
        sys.exit(1)

    print('Loading corpus...')
    corpus_df = pd.read_csv(corpus_path)
    print(f'Corpus: {len(corpus_df):,} tokens, {corpus_df["stem"].nunique()} receipts\n')

    # Train / test split 80/20
    stems       = corpus_df['stem'].unique()
    n_train     = int(len(stems) * 0.8)
    train_df    = corpus_df[corpus_df['stem'].isin(stems[:n_train])]
    test_df     = corpus_df[corpus_df['stem'].isin(stems[n_train:])]

    print(f'Train: {len(stems[:n_train])} receipts ({len(train_df):,} tokens)')
    print(f'Test : {len(stems[n_train:])} receipts ({len(test_df):,} tokens)\n')

    # Train
    Path('models').mkdir(exist_ok=True)
    tagger = train_crf(train_df, verbose=False)
    save_model(tagger, 'models/crf_model.pkl')

    # Evaluate
    print('\nEvaluating on test set...')
    results = evaluate(tagger, test_df)
    print('\nResults:')
    print(results.to_string(index=False))

    # Top features
    print('\nTop 15 most important features:')
    print(top_features(tagger, n=15).to_string(index=False))
