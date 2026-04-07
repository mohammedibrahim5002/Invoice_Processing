"""
src/parser.py
=============
Takes the token list from ocr_engine.extract_tokens()
and extracts structured fields using regex patterns.

Fields extracted:
    invoice_number  - INV-001, DOCUMENT NO, RECEIPT NO, etc.
    date            - many date formats
    due_date        - payment due date (B2B invoices)
    vendor_name     - heuristic: company suffix detection
    total           - final amount due
    tax             - tax amount
    tax_rate        - tax percentage
    currency        - RM, USD, EUR, etc.
    email           - email address
    phone           - phone number

company and address are left blank for the CRF model (model.py).

Usage:
    from src.parser import parse_fields
    tokens = extract_tokens('invoice.jpg')
    fields = parse_fields(tokens)
"""

import re
from pathlib import Path


# ── Date patterns ─────────────────────────────────────────────────────────────

DATE_PATTERNS = [
    # DD/MM/YYYY or DD-MM-YYYY or DD.MM.YYYY
    (re.compile(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{4})\b'), 0.95),

    # DD/MM/YY
    (re.compile(r'\b(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2})\b'), 0.85),

    # 25 December 2018 or December 25, 2018
    (re.compile(
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'[a-z]*\.?\s+\d{4})\b', re.IGNORECASE), 0.92),
    (re.compile(
        r'\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+'
        r'\d{1,2},?\s+\d{4})\b', re.IGNORECASE), 0.92),

    # YYYY-MM-DD
    (re.compile(r'\b(\d{4}[\/\-]\d{2}[\/\-]\d{2})\b'), 0.90),

    # 25 Dec 2018
    (re.compile(
        r'\b(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)'
        r'\w*\s+\d{4})\b', re.IGNORECASE), 0.88),
]

# ── Due date patterns ─────────────────────────────────────────────────────────

DUE_DATE_PATTERNS = [
    # "Due Date: 15/01/2024"
    (re.compile(
        r'(?:due\s*date|payment\s*due|due\s*by|pay\s*by)'
        r'[\s:]+(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        re.IGNORECASE), 0.95),

    # "Due Date: January 15, 2024"
    (re.compile(
        r'(?:due\s*date|payment\s*due|due\s*by)'
        r'[\s:]+([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
        re.IGNORECASE), 0.90),

    # "Net 30" — payment terms (not a date but useful)
    (re.compile(
        r'\b(net\s*\d+)\b',
        re.IGNORECASE), 0.75),
]

# ── Total patterns ────────────────────────────────────────────────────────────

TOTAL_PATTERNS = [
    # "TOTAL (RM): 9.00" or "ROUNDED TOTAL (RM): 9.00" — SROIE format
    (re.compile(
        r'total\s*(?:\([A-Z]+\))?\s*:\s*(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
        re.IGNORECASE), 0.98),

    # "GRAND TOTAL: 123.45" or "TOTAL AMOUNT: 1,234.56"
    (re.compile(
        r'(?:grand\s*total|total\s*amount|amount\s*due|total\s*payable|'
        r'total\s*bill|net\s*total|balance\s*due)'
        r'[\s:]*(?:[A-Z]{2,3}\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
        re.IGNORECASE), 0.97),

    # "TOTAL: 123.45"
    (re.compile(
        r'\btotal[\s:]+(?:[A-Z]{2,3}\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
        re.IGNORECASE), 0.93),

    # "MYR 9.00" or "RM 9.00" — space required to avoid matching barcodes
    (re.compile(
        r'(?:MYR|RM|USD|SGD|GBP|EUR|INR)\s+(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
        re.IGNORECASE), 0.88),

    # "SUBTOTAL: 123.45"
    (re.compile(
        r'\bsubtotal[\s:]+(?:[A-Z]{2,3}\s*)?(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
        re.IGNORECASE), 0.80),
]

# ── Tax patterns ──────────────────────────────────────────────────────────────

TAX_PATTERNS = [
    # "GST 6% : 0.54" or "TAX: 1.08"
    (re.compile(
        r'(?:gst|sst|vat|tax|service\s*charge)'
        r'[\s\d%\(\)]*?[\s:]+(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)',
        re.IGNORECASE), 0.92),

    # "6% TAX: 0.54"
    (re.compile(
        r'\d+(?:\.\d+)?%\s*(?:gst|sst|vat|tax)[\s:]+(\d+(?:\.\d{1,2})?)',
        re.IGNORECASE), 0.90),
]

TAX_RATE_PATTERNS = [
    # "GST 6%" or "TAX RATE: 8%"
    (re.compile(
        r'(?:gst|sst|vat|tax)[\s\w]*?(\d+(?:\.\d+)?)\s*%',
        re.IGNORECASE), 0.90),
]

# ── Invoice number patterns ───────────────────────────────────────────────────

INVOICE_NO_PATTERNS = [
    # "DOCUMENT NO : TD01167104" — common in SROIE receipts
    (re.compile(
        r'(?:document\s*no|doc\s*no|doc\s*number)'
        r'[\s:#]*([A-Z0-9][A-Z0-9\-\/]{2,20})',
        re.IGNORECASE), 0.92),

    # "Invoice No: INV-2024-001"
    (re.compile(
        r'(?:invoice\s*(?:no|num|number|#)|inv\s*(?:no|#))'
        r'[\s:#]*([A-Z0-9][A-Z0-9\-\/]{2,20})',
        re.IGNORECASE), 0.95),

    # "Receipt No: 12345"
    (re.compile(
        r'(?:receipt\s*(?:no|num|number|#))'
        r'[\s:#]*([A-Z0-9][A-Z0-9\-\/]{2,20})',
        re.IGNORECASE), 0.90),

    # "Bill No: 12345" or "Ref No: 12345"
    (re.compile(
        r'(?:bill\s*(?:no|number)|ref\s*(?:no|number)|order\s*(?:no|number))'
        r'[\s:#]*([A-Z0-9][A-Z0-9\-\/]{2,20})',
        re.IGNORECASE), 0.88),

    # Standalone INV- or RCP- prefix
    (re.compile(r'\b(INV[\-\/][A-Z0-9\-]{3,20})\b'), 0.88),
    (re.compile(r'\b(RCP[\-\/][A-Z0-9\-]{3,20})\b'), 0.85),
]

# ── Vendor name patterns ──────────────────────────────────────────────────────
# Looks for company suffixes — works well on B2B invoices
# SROIE receipts often have the company name at the top without a label

VENDOR_NAME_PATTERNS = [
    # "From: ACME Sdn Bhd" or "Vendor: XYZ Ltd"
    (re.compile(
        r'(?:from|vendor|supplier|billed?\s*by|sold\s*by)'
        r'[\s:]+([A-Z][A-Za-z0-9\s\.\,\&\(\)]{3,50}'
        r'(?:sdn[\s.]*bhd|sdn|bhd|ltd|llc|inc|corp|co\.|pty|plc|gmbh|pte))',
        re.IGNORECASE), 0.90),

    # Company name with suffix anywhere on the page (top 40% more likely)
    (re.compile(
        r'([A-Z][A-Za-z0-9\s\.\,\&\(\)]{3,50}'
        r'(?:\s+(?:sdn[\s.]*bhd|sdn\s*bhd|bhd|ltd|llc|inc|corp|co\.|pty|plc|pte)))',
        re.IGNORECASE), 0.72),
]

# ── Email & phone ─────────────────────────────────────────────────────────────

EMAIL_PATTERN = re.compile(
    r'\b([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})\b'
)

PHONE_PATTERNS = [
    # "Tel: +60-12-345-6789"
    (re.compile(
        r'(?:tel|phone|ph|mob|mobile|fax|contact|hp)'
        r'[\s:.]*(\+?[\d][\d\s\-\(\)\.]{7,20})',
        re.IGNORECASE), 0.92),

    # International: +60123456789
    (re.compile(
        r'\b(\+\d{1,3}[\s\-]?\d{2,4}[\s\-]?\d{3,4}[\s\-]?\d{3,4})\b'),
        0.85),
]

# ── Currency patterns ─────────────────────────────────────────────────────────

CURRENCY_PATTERNS = [
    (re.compile(r'\b(USD|EUR|GBP|SGD|MYR|INR|AUD|CAD)\b'), 0.95),
    (re.compile(r'\b(RM)\b'), 0.90),
    (re.compile(r'([\$£€₹])'), 0.88),
]


# ── Helper functions ──────────────────────────────────────────────────────────

def _tokens_to_text(tokens: list) -> str:
    """Join all token texts into one string."""
    return ' '.join(t['text'] for t in tokens if t.get('text', '').strip())


def _tokens_to_lines(tokens: list) -> list:
    """Group tokens by line_num, return list of line strings."""
    if not tokens:
        return []
    lines_dict = {}
    for t in tokens:
        line_num = t.get('line_num', 0)
        if line_num not in lines_dict:
            lines_dict[line_num] = []
        lines_dict[line_num].append(t['text'])
    return [' '.join(words) for words in lines_dict.values()]


def _find_first(patterns: list, text: str) -> tuple:
    """Try patterns in order. Return (value, confidence) for first match."""
    for pattern, confidence in patterns:
        match = pattern.search(text)
        if match:
            value = match.group(1).strip()
            if value:
                return value, confidence
    return '', 0.0


def _clean_amount(value: str) -> str:
    """Remove commas. '1,234.56' -> '1234.56'"""
    return value.replace(',', '').strip()


def _find_vendor_from_top(tokens: list) -> tuple:
    """
    Heuristic: vendor name is usually in the top 20% of the receipt.
    Look for a token line in the top section that contains a company suffix.
    Falls back to the very first text line if nothing found.
    """
    if not tokens:
        return '', 0.0

    # Only look at top 20% of page
    top_tokens = [t for t in tokens if t.get('y_norm', 1.0) < 0.20]

    # Build lines from top tokens
    lines_dict = {}
    for t in top_tokens:
        ln = t.get('line_num', 0)
        if ln not in lines_dict:
            lines_dict[ln] = []
        lines_dict[ln].append(t['text'])

    top_lines = [' '.join(words) for words in lines_dict.values()]
    top_text  = '\n'.join(top_lines)

    # Try company suffix patterns on top section
    val, conf = _find_first(VENDOR_NAME_PATTERNS, top_text)
    if val:
        return val.strip(), conf

    # Fallback: return the first non-empty line from the top
    for line in top_lines:
        line = line.strip()
        if len(line) > 3:
            return line, 0.45

    return '', 0.0


def _find_total_from_tokens(tokens: list) -> tuple:
    """
    Fallback total extraction using spatial position.
    Total = topmost currency amount on the right side, above the CASH line.
    """
    if not tokens:
        return '', 0.0

    # Find y position of CASH / CHANGE line
    cash_y = 1.0
    for t in tokens:
        if t['text'].upper().strip() in {'CASH', 'CHANGE', 'TENDER'}:
            cash_y = min(cash_y, t.get('y_norm', 1.0))

    amount_re = re.compile(r'^[\$£€RM]?\d{1,3}(?:,\d{3})*(?:\.\d{2})?$')

    candidates = []
    for t in tokens:
        text   = t['text'].strip()
        x_norm = t.get('x_norm', 0)
        y_norm = t.get('y_norm', 0)

        if not amount_re.match(text):
            continue
        if x_norm < 0.5:        # right side only
            continue
        if y_norm >= cash_y:    # above cash line only
            continue

        numeric_str = re.sub(r'[^\d.]', '', text)
        try:
            numeric = float(numeric_str)
            if numeric > 0:
                candidates.append((y_norm, numeric, text))
        except ValueError:
            continue

    # Relax if nothing found
    if not candidates:
        for t in tokens:
            text   = t['text'].strip()
            x_norm = t.get('x_norm', 0)
            y_norm = t.get('y_norm', 0)

            if not amount_re.match(text) or x_norm < 0.5:
                continue

            numeric_str = re.sub(r'[^\d.]', '', text)
            try:
                numeric = float(numeric_str)
                if numeric > 0:
                    candidates.append((y_norm, numeric, text))
            except ValueError:
                continue

    if not candidates:
        return '', 0.0

    # Topmost = total
    candidates.sort(key=lambda x: x[0])
    return _clean_amount(candidates[0][2]), 0.72


# ── Main function ─────────────────────────────────────────────────────────────

def parse_fields(tokens: list) -> dict:
    """
    Extract all fields from a token list.

    Args:
        tokens : output of ocr_engine.extract_tokens()

    Returns:
        dict with fields + '_confidence' key
    """
    full_text  = _tokens_to_text(tokens)
    lines      = _tokens_to_lines(tokens)
    lines_text = '\n'.join(lines)

    confidence = {}

    # ── Invoice number ────────────────────────────────────────────────────────
    invoice_number, inv_conf = _find_first(INVOICE_NO_PATTERNS, lines_text)
    confidence['invoice_number'] = inv_conf

    # ── Date ──────────────────────────────────────────────────────────────────
    date, date_conf = _find_first(DATE_PATTERNS, lines_text)
    confidence['date'] = date_conf

    # ── Due date ──────────────────────────────────────────────────────────────
    due_date, due_conf = _find_first(DUE_DATE_PATTERNS, lines_text)
    confidence['due_date'] = due_conf

    # ── Vendor name ───────────────────────────────────────────────────────────
    # Try patterns on full text first, then top-of-page heuristic
    vendor_name, vendor_conf = _find_first(VENDOR_NAME_PATTERNS, lines_text)
    if not vendor_name:
        vendor_name, vendor_conf = _find_vendor_from_top(tokens)
    confidence['vendor_name'] = vendor_conf

    # ── Total ─────────────────────────────────────────────────────────────────
    total, total_conf = _find_first(TOTAL_PATTERNS, lines_text)
    if total:
        total = _clean_amount(total)
    else:
        total, total_conf = _find_total_from_tokens(tokens)
    confidence['total'] = total_conf

    # ── Tax ───────────────────────────────────────────────────────────────────
    tax, tax_conf = _find_first(TAX_PATTERNS, lines_text)
    if tax:
        tax = _clean_amount(tax)
    confidence['tax'] = tax_conf

    # ── Tax rate ──────────────────────────────────────────────────────────────
    tax_rate, tax_rate_conf = _find_first(TAX_RATE_PATTERNS, lines_text)
    if tax_rate:
        tax_rate = tax_rate + '%'
    confidence['tax_rate'] = tax_rate_conf

    # ── Currency ──────────────────────────────────────────────────────────────
    currency, cur_conf = _find_first(CURRENCY_PATTERNS, full_text)
    confidence['currency'] = cur_conf

    # ── Email ─────────────────────────────────────────────────────────────────
    email_match    = EMAIL_PATTERN.search(full_text)
    email          = email_match.group(1) if email_match else ''
    confidence['email'] = 0.98 if email else 0.0

    # ── Phone ─────────────────────────────────────────────────────────────────
    phone, phone_conf = _find_first(PHONE_PATTERNS, lines_text)
    confidence['phone'] = phone_conf

    # ── Company & Address — left for CRF ──────────────────────────────────────
    company = ''
    address = ''
    confidence['company'] = 0.0
    confidence['address'] = 0.0

    return {
        'invoice_number' : invoice_number,
        'date'           : date,
        'due_date'       : due_date,
        'vendor_name'    : vendor_name,
        'company'        : company,
        'address'        : address,
        'total'          : total,
        'tax'            : tax,
        'tax_rate'       : tax_rate,
        'currency'       : currency,
        'email'          : email,
        'phone'          : phone,
        '_confidence'    : confidence,
    }


# ── Batch ─────────────────────────────────────────────────────────────────────

def parse_batch(token_lists: list) -> list:
    """Run parse_fields on a list of token lists."""
    return [parse_fields(tokens) for tokens in token_lists]


# ── Quick test ────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    sys.path.insert(0, '.')
    from src.ocr_engine import extract_tokens

    img_dir = Path('data/raw/SROIE2019/train/img')
    box_dir = Path('data/raw/SROIE2019/train/box')

    if not img_dir.exists():
        print('SROIE dataset not found.')
        sys.exit(0)

    samples = sorted(img_dir.glob('*.jpg'))[:5]

    for sample in samples:
        box    = box_dir / f'{sample.stem}.txt'
        tokens = extract_tokens(sample, box_path=box)
        fields = parse_fields(tokens)
        conf   = fields.pop('_confidence', {})

        print(f'\n{"="*65}')
        print(f'Receipt: {sample.name}')
        print(f'{"="*65}')
        print(f'{"Field":15s}  {"Value":38s}  Conf')
        print(f'{"-"*15}  {"-"*38}  ----')
        for key, val in fields.items():
            c   = conf.get(key, 0.0)
            bar = '█' * int(c * 10)
            print(f'{key:15s}  {str(val):38s}  {c:.2f}  {bar}')
