import time
import camelot
import pandas as pd
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import pdfplumber
from multiprocessing import Pool, cpu_count

MAX_PROCESSES = min(6, cpu_count())
OCR_BATCH = 8   # Smaller batch = safer for Streamlit

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    rec_batch_num=16
)

# ---------------- PDF TYPE DETECTION ----------------
def is_scanned_pdf(pdf_path, check_pages=3):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i in range(min(check_pages, len(pdf.pages))):
                text = pdf.pages[i].extract_text()
                if text and len(text.strip()) > 20:
                    return False
        return True
    except:
        return True

# ---------------- TEXT EXTRACTION ----------------
def extract_text_tables(pdf_path):
    tables = camelot.read_pdf(
        pdf_path,
        pages="all",
        flavor="stream"
    )
    dfs = []
    for t in tables:
        df = t.df
        df["Source"] = "Text"
        dfs.append(df)
    return dfs

# ---------------- OCR WORKER ----------------
def ocr_worker(args):
    pdf_path, page_range = args
    images = convert_from_path(
        pdf_path,
        dpi=200,
        first_page=page_range[0],
        last_page=page_range[-1]
    )

    data = []
    for idx, img in enumerate(images):
        result = ocr.ocr(img, cls=True)
        lines = [line[1][0] for line in result[0]]
        if lines:
            df = pd.DataFrame(lines, columns=["Text"])
            df["Page"] = page_range[idx]
            df["Source"] = "OCR"
            data.append(df)
    return data

# ---------------- OCR MANAGER ----------------
def extract_ocr(pdf_path):
    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    page_ranges = [
        list(range(i, min(i + OCR_BATCH, total_pages) + 1))
        for i in range(1, total_pages + 1, OCR_BATCH)
    ]

    results = []
    with Pool(MAX_PROCESSES) as pool:
        for chunk in pool.imap_unordered(ocr_worker, [(pdf_path, r) for r in page_ranges]):
            results.extend(chunk)

    return results

# ---------------- MAIN FUNCTION ----------------
def extract_pdf(pdf_path):
    start = time.time()

    scanned = is_scanned_pdf(pdf_path)

    if not scanned:
        tables = extract_text_tables(pdf_path)
        df = pd.concat(tables, ignore_index=True)
        return df, time.time() - start

    ocr_tables = extract_ocr(pdf_path)
    df = pd.concat(ocr_tables, ignore_index=True)
    return df, time.time() - start
