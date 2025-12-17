import time
import camelot
import pandas as pd
from paddleocr import PaddleOCR
from pdf2image import convert_from_path
import pdfplumber
from multiprocessing import Pool, cpu_count

POPPLER_PATH = None  # Streamlit Cloud auto-detects

MAX_PROCESSES = min(6, cpu_count())
OCR_BATCH = 10

ocr = PaddleOCR(
    use_angle_cls=True,
    lang='en',
    use_gpu=False,
    rec_batch_num=16
)

def extract_text_tables(pdf_path):
    try:
        tables = camelot.read_pdf(pdf_path, pages="all", flavor="stream")
        if tables:
            dfs = []
            for t in tables:
                df = t.df
                df["Source"] = "Text"
                dfs.append(df)
            return dfs
    except:
        pass
    return []

def ocr_chunk(args):
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
        rows = [line[1][0] for line in result[0]]
        if rows:
            df = pd.DataFrame(rows, columns=["Text"])
            df["Page"] = page_range[idx]
            df["Source"] = "OCR"
            data.append(df)
    return data

def extract_pdf(pdf_path):
    start = time.time()

    text_tables = extract_text_tables(pdf_path)
    if text_tables:
        return pd.concat(text_tables, ignore_index=True), time.time() - start

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)

    page_ranges = [
        list(range(i, min(i + OCR_BATCH, total_pages) + 1))
        for i in range(1, total_pages + 1, OCR_BATCH)
    ]

    args = [(pdf_path, r) for r in page_ranges]

    results = []
    with Pool(MAX_PROCESSES) as pool:
        for chunk in pool.imap_unordered(ocr_chunk, args):
            results.extend(chunk)

    return pd.concat(results, ignore_index=True), time.time() - start
