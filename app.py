import streamlit as st
import pdfplumber
import pandas as pd
import numpy as np
import os
import re
import time
import tempfile
import shutil
import gc
from io import BytesIO

from pdf2image import convert_from_path
import pytesseract


# ============================================================
# STREAMLIT CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF Fast Extractor",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CONSTANTS
# ============================================================

DEFAULT_BATCH_SIZE = 150
MAX_BATCH_SIZE = 150

DEFAULT_OCR_DPI = 180
MIN_OCR_DPI = 120
MAX_OCR_DPI = 250


# ============================================================
# SYSTEM CHECK
# ============================================================

def check_system():
    """
    Check whether Tesseract is available.
    Poppler is checked indirectly by pdf2image when processing.
    """

    status = {
        "tesseract": False,
        "pdfplumber": False,
        "pandas": False,
        "pdf2image": False
    }

    try:
        pytesseract.get_tesseract_version()
        status["tesseract"] = True
    except Exception:
        status["tesseract"] = False

    try:
        import pdfplumber
        status["pdfplumber"] = True
    except Exception:
        status["pdfplumber"] = False

    try:
        import pandas
        status["pandas"] = True
    except Exception:
        status["pandas"] = False

    try:
        import pdf2image
        status["pdf2image"] = True
    except Exception:
        status["pdf2image"] = False

    return status


# ============================================================
# PDF TYPE DETECTION
# ============================================================

def detect_pdf_type(pdf_path):
    """
    Detect whether the PDF is searchable or scanned.

    Checks the first few pages for extractable text.
    """

    try:
        with pdfplumber.open(pdf_path) as pdf:

            pages_to_check = min(3, len(pdf.pages))

            text_pages = 0
            total_chars = 0

            for i in range(pages_to_check):

                try:
                    text = pdf.pages[i].extract_text()

                    if text and text.strip():

                        text_pages += 1
                        total_chars += len(text.strip())

                except Exception:
                    continue

            if text_pages > 0 and total_chars > 30:
                return "searchable"

            return "scanned"

    except Exception:
        return "scanned"


# ============================================================
# GET TOTAL PAGES
# ============================================================

def get_total_pages(pdf_path):

    with pdfplumber.open(pdf_path) as pdf:
        return len(pdf.pages)


# ============================================================
# CREATE 150-PAGE BATCHES
# ============================================================

def create_batches(total_pages, batch_size=150):

    batch_size = min(batch_size, MAX_BATCH_SIZE)

    batches = []

    for start in range(0, total_pages, batch_size):

        end = min(start + batch_size, total_pages)

        batches.append(
            list(range(start, end))
        )

    return batches


# ============================================================
# CLEAN CELL
# ============================================================

def clean_cell(value):

    if value is None:
        return ""

    value = str(value)

    value = value.replace("\n", " ")
    value = value.replace("\r", " ")

    value = re.sub(r"\s+", " ", value)

    return value.strip()


# ============================================================
# CLEAN DATAFRAME
# ============================================================

def clean_dataframe(df):

    if df is None:
        return None

    if df.empty:
        return df

    df = df.copy()

    # Clean every cell
    for column in df.columns:
        df[column] = df[column].apply(clean_cell)

    # Remove completely empty rows
    df = df.replace("", np.nan)

    df = df.dropna(
        axis=0,
        how="all"
    )

    df = df.fillna("")

    return df.reset_index(drop=True)


# ============================================================
# SEARCHABLE PDF - TABLE EXTRACTION
# ============================================================

def extract_tables_from_page(page):

    results = []

    try:

        tables = page.extract_tables()

        if tables:

            for table in tables:

                if not table:
                    continue

                try:

                    df = pd.DataFrame(table)

                    if df.empty:
                        continue

                    df = clean_dataframe(df)

                    if df is not None and not df.empty:

                        results.append(df)

                except Exception:
                    continue

    except Exception:
        pass

    return results


# ============================================================
# SEARCHABLE PDF - TEXT FALLBACK
# ============================================================

def extract_text_from_page(page, page_number):

    results = []

    try:

        text = page.extract_text()

        if not text or not text.strip():
            return results

        lines = []

        for line in text.splitlines():

            line = clean_cell(line)

            if line:
                lines.append(line)

        if not lines:
            return results

        rows = []

        for line in lines:

            # First try multiple spaces as column separators.
            parts = re.split(r"\s{2,}", line)

            parts = [
                clean_cell(x)
                for x in parts
                if clean_cell(x)
            ]

            # If there is no obvious column spacing,
            # preserve the whole line.
            if not parts:
                continue

            rows.append(parts)

        if not rows:
            return results

        max_columns = max(
            len(row)
            for row in rows
        )

        normalized_rows = []

        for row in rows:

            row = list(row)

            if len(row) < max_columns:

                row.extend(
                    [""] * (max_columns - len(row))
                )

            elif len(row) > max_columns:

                row = row[:max_columns]

            normalized_rows.append(row)

        df = pd.DataFrame(normalized_rows)

        df["Page"] = page_number

        df = clean_dataframe(df)

        if df is not None and not df.empty:

            results.append(df)

    except Exception:
        pass

    return results


# ============================================================
# PROCESS SEARCHABLE BATCH
# ============================================================

def process_searchable_batch(
    pdf_path,
    page_numbers
):

    batch_results = []

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_index in page_numbers:

                page_number = page_index + 1

                try:

                    page = pdf.pages[page_index]

                    # ----------------------------------------
                    # FIRST: TRY TABLE EXTRACTION
                    # ----------------------------------------

                    tables = extract_tables_from_page(page)

                    if tables:

                        for df in tables:

                            df = df.copy()

                            # Add page number if not already present
                            if "Page" not in df.columns:
                                df["Page"] = page_number

                            batch_results.append(df)

                    else:

                        # ------------------------------------
                        # FALLBACK: TEXT EXTRACTION
                        # ------------------------------------

                        text_results = extract_text_from_page(
                            page,
                            page_number
                        )

                        batch_results.extend(
                            text_results
                        )

                except Exception:
                    continue

    except Exception:
        pass

    return batch_results


# ============================================================
# OCR - SINGLE PAGE
# ============================================================

def ocr_single_page(
    pdf_path,
    page_number,
    dpi
):

    results = []

    image = None

    try:

        # Convert ONLY ONE PAGE.
        # This is important for Cloud RAM usage.

        images = convert_from_path(
            pdf_path,
            dpi=dpi,
            first_page=page_number,
            last_page=page_number,
            use_cropbox=True,
            thread_count=1
        )

        if not images:
            return results

        image = images[0]

        # OCR
        text = pytesseract.image_to_string(
            image,
            lang="eng",
            config="--psm 6"
        )

        if not text or not text.strip():
            return results

        lines = []

        for line in text.splitlines():

            line = clean_cell(line)

            if line:
                lines.append(line)

        if not lines:
            return results

        rows = []

        for line in lines:

            # Detect multiple spaces as possible
            # column boundaries.

            parts = re.split(
                r"\s{2,}",
                line
            )

            parts = [
                clean_cell(x)
                for x in parts
                if clean_cell(x)
            ]

            # If OCR does not produce obvious
            # column spacing, keep the line.

            if not parts:
                continue

            rows.append(parts)

        if not rows:
            return results

        max_columns = max(
            len(row)
            for row in rows
        )

        normalized_rows = []

        for row in rows:

            row = list(row)

            if len(row) < max_columns:

                row.extend(
                    [""] *
                    (max_columns - len(row))
                )

            elif len(row) > max_columns:

                row = row[:max_columns]

            normalized_rows.append(row)

        df = pd.DataFrame(
            normalized_rows
        )

        df["Page"] = page_number

        df = clean_dataframe(df)

        if df is not None and not df.empty:

            results.append(df)

    except Exception:
        pass

    finally:

        try:
            if image is not None:
                image.close()
        except Exception:
            pass

        try:
            del image
        except Exception:
            pass

        try:
            del images
        except Exception:
            pass

        gc.collect()

    return results


# ============================================================
# PROCESS OCR BATCH
# ============================================================

def process_ocr_batch(
    pdf_path,
    page_numbers,
    dpi
):

    batch_results = []

    # IMPORTANT:
    # Pages are processed one at a time.
    # The batch is still 150 pages.

    for page_index in page_numbers:

        page_number = page_index + 1

        results = ocr_single_page(
            pdf_path,
            page_number,
            dpi
        )

        batch_results.extend(results)

    return batch_results


# ============================================================
# COMBINE DATA
# ============================================================

def combine_data(all_data):

    if not all_data:
        return None

    valid_data = []

    for df in all_data:

        if df is None:
            continue

        if not isinstance(df, pd.DataFrame):
            continue

        if df.empty:
            continue

        valid_data.append(df)

    if not valid_data:
        return None

    # Determine maximum column count.
    max_columns = max(
        len(df.columns)
        for df in valid_data
    )

    normalized = []

    for df in valid_data:

        df = df.copy()

        # Convert column names to strings
        df.columns = [
            str(col)
            for col in df.columns
        ]

        current_columns = len(df.columns)

        if current_columns < max_columns:

            for i in range(
                current_columns,
                max_columns
            ):

                df[f"Column_{i + 1}"] = ""

        normalized.append(df)

    combined = pd.concat(
        normalized,
        ignore_index=True,
        sort=False
    )

    # Clean again
    combined = clean_dataframe(
        combined
    )

    return combined


# ============================================================
# EXCEL GENERATION
# ============================================================

def dataframe_to_excel(df):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        # Excel maximum row limit
        MAX_EXCEL_ROWS = 1_048_000

        total_rows = len(df)

        if total_rows <= MAX_EXCEL_ROWS:

            df.to_excel(
                writer,
                index=False,
                sheet_name="Extracted_Data"
            )

        else:

            sheet_number = 1

            for start in range(
                0,
                total_rows,
                MAX_EXCEL_ROWS
            ):

                end = min(
                    start + MAX_EXCEL_ROWS,
                    total_rows
                )

                chunk = df.iloc[
                    start:end
                ]

                chunk.to_excel(
                    writer,
                    index=False,
                    sheet_name=f"Data_{sheet_number}"
                )

                sheet_number += 1

    output.seek(0)

    return output.getvalue()


# ============================================================
# MAIN EXTRACTION
# ============================================================

def run_extraction(
    pdf_path,
    batch_size,
    dpi,
    progress_bar,
    status_text
):

    start_time = time.time()

    # ----------------------------------------
    # DETECT PDF TYPE
    # ----------------------------------------

    status_text.info(
        "🔍 Detecting PDF type..."
    )

    pdf_type = detect_pdf_type(
        pdf_path
    )

    # ----------------------------------------
    # PAGE COUNT
    # ----------------------------------------

    status_text.info(
        "📄 Counting PDF pages..."
    )

    total_pages = get_total_pages(
        pdf_path
    )

    # ----------------------------------------
    # CREATE BATCHES
    # ----------------------------------------

    batches = create_batches(
        total_pages,
        batch_size
    )

    total_batches = len(batches)

    if pdf_type == "searchable":

        mode_text = "Searchable PDF"

    else:

        mode_text = "Scanned PDF / OCR"

    status_text.success(
        f"Detected: **{mode_text}** | "
        f"Total Pages: **{total_pages:,}** | "
        f"Batches: **{total_batches}**"
    )

    all_data = []

    completed_batches = 0

    # ----------------------------------------
    # PROCESS BATCHES
    # ----------------------------------------

    for batch_index, page_numbers in enumerate(
        batches,
        start=1
    ):

        batch_start_time = time.time()

        first_page = page_numbers[0] + 1
        last_page = page_numbers[-1] + 1

        status_text.info(
            f"⚙️ Processing Batch "
            f"{batch_index}/{total_batches} "
            f"| Pages {first_page}–{last_page}"
        )

        # ------------------------------------
        # SEARCHABLE
        # ------------------------------------

        if pdf_type == "searchable":

            batch_data = process_searchable_batch(
                pdf_path,
                page_numbers
            )

        # ------------------------------------
        # OCR
        # ------------------------------------

        else:

            batch_data = process_ocr_batch(
                pdf_path,
                page_numbers,
                dpi
            )

        # ------------------------------------
        # STORE RESULTS
        # ------------------------------------

        if batch_data:

            all_data.extend(
                batch_data
            )

        completed_batches += 1

        batch_time = (
            time.time() -
            batch_start_time
        )

        elapsed = (
            time.time() -
            start_time
        )

        avg_batch_time = (
            elapsed /
            completed_batches
        )

        remaining_batches = (
            total_batches -
            completed_batches
        )

        eta_seconds = (
            remaining_batches *
            avg_batch_time
        )

        progress = (
            completed_batches /
            total_batches
        )

        progress_bar.progress(
            progress
        )

        # Convert ETA
        if eta_seconds < 60:

            eta_text = (
                f"{eta_seconds:.0f} sec"
            )

        else:

            eta_text = (
                f"{eta_seconds / 60:.1f} min"
            )

        status_text.info(
            f"✅ Batch {batch_index}/{total_batches} "
            f"completed | "
            f"Pages {first_page}–{last_page} | "
            f"Batch Time: {batch_time:.1f}s | "
            f"ETA: {eta_text}"
        )

        # Release temporary memory
        gc.collect()

    # ----------------------------------------
    # COMBINE
    # ----------------------------------------

    status_text.info(
        "📊 Combining extracted data..."
    )

    combined_df = combine_data(
        all_data
    )

    # Release individual dataframes
    del all_data

    gc.collect()

    total_time = (
        time.time() -
        start_time
    )

    # ----------------------------------------
    # COMPLETE
    # ----------------------------------------

    progress_bar.progress(
        1.0
    )

    if combined_df is None:

        status_text.warning(
            f"⚠️ Extraction completed, "
            f"but no data was detected. "
            f"Time: {total_time:.1f}s"
        )

        return None, {
            "total_pages": total_pages,
            "total_batches": total_batches,
            "pdf_type": pdf_type,
            "total_time": total_time,
            "rows": 0
        }

    rows = len(combined_df)

    status_text.success(
        f"🎉 Extraction completed! "
        f"{rows:,} rows extracted in "
        f"{total_time:.1f} seconds."
    )

    stats = {
        "total_pages": total_pages,
        "total_batches": total_batches,
        "pdf_type": pdf_type,
        "total_time": total_time,
        "rows": rows
    }

    return combined_df, stats


# ============================================================
# APP UI
# ============================================================

st.title(
    "📄 PDF Fast Extractor"
)

st.caption(
    "150-page batch processing • "
    "Searchable PDF + OCR • "
    "Excel Export"
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "⚙️ Settings"
    )

    batch_size = st.number_input(
        "Batch Size",
        min_value=1,
        max_value=150,
        value=150,
        step=10,
        help=(
            "Maximum 150 pages per batch. "
            "For Cloud, 150 is recommended."
        )
    )

    dpi = st.slider(
        "OCR DPI",
        min_value=MIN_OCR_DPI,
        max_value=MAX_OCR_DPI,
        value=DEFAULT_OCR_DPI,
        step=10,
        help=(
            "Higher DPI can improve OCR "
            "but increases processing time "
            "and memory usage."
        )
    )

    st.divider()

    st.subheader(
        "☁️ Cloud Mode"
    )

    st.write(
        "Multiprocessing: **OFF**"
    )

    st.write(
        "Batch size: **150 pages**"
    )

    st.write(
        "OCR: **1 page at a time**"
    )

    st.divider()

    st.subheader(
        "🔧 System"
    )

    system_status = check_system()

    if system_status["tesseract"]:
        st.success(
            "Tesseract: OK"
        )
    else:
        st.error(
            "Tesseract: Not found"
        )

    if system_status["pdfplumber"]:
        st.success(
            "pdfplumber: OK"
        )
    else:
        st.error(
            "pdfplumber: Error"
        )

    if system_status["pdf2image"]:
        st.success(
            "pdf2image: OK"
        )
    else:
        st.error(
            "pdf2image: Error"
        )


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    help="Upload your counselling/allotment PDF."
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file is not None:

    st.success(
        f"📄 {uploaded_file.name}"
    )

    file_size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.write(
        f"File size: **{file_size_mb:.2f} MB**"
    )

    start_button = st.button(
        "🚀 Start Extraction",
        type="primary",
        use_container_width=True
    )

    if start_button:

        temp_dir = tempfile.mkdtemp(
            prefix="pdf_extractor_"
        )

        pdf_path = os.path.join(
            temp_dir,
            uploaded_file.name
        )

        try:

            # ----------------------------------------
            # SAVE UPLOADED PDF
            # ----------------------------------------

            with open(
                pdf_path,
                "wb"
            ) as f:

                f.write(
                    uploaded_file.getbuffer()
                )

            # ----------------------------------------
            # UI PLACEHOLDERS
            # ----------------------------------------

            progress_bar = st.progress(
                0
            )

            status_text = st.empty()

            # ----------------------------------------
            # EXTRACTION
            # ----------------------------------------

            with st.spinner(
                "Processing PDF..."
            ):

                combined_df, stats = run_extraction(
                    pdf_path=pdf_path,
                    batch_size=int(batch_size),
                    dpi=int(dpi),
                    progress_bar=progress_bar,
                    status_text=status_text
                )

            # ----------------------------------------
            # RESULTS
            # ----------------------------------------

            if combined_df is not None:

                st.divider()

                st.subheader(
                    "📊 Extraction Summary"
                )

                col1, col2, col3, col4 = st.columns(4)

                with col1:
                    st.metric(
                        "Pages",
                        f"{stats['total_pages']:,}"
                    )

                with col2:
                    st.metric(
                        "Batches",
                        f"{stats['total_batches']:,}"
                    )

                with col3:
                    st.metric(
                        "Rows",
                        f"{stats['rows']:,}"
                    )

                with col4:
                    st.metric(
                        "Time",
                        f"{stats['total_time']:.1f}s"
                    )

                st.subheader(
                    "👀 Preview"
                )

                preview_rows = min(
                    100,
                    len(combined_df)
                )

                st.dataframe(
                    combined_df.head(
                        preview_rows
                    ),
                    use_container_width=True,
                    height=500
                )

                # ------------------------------------
                # CREATE EXCEL
                # ------------------------------------

                with st.spinner(
                    "Creating Excel file..."
                ):

                    excel_data = dataframe_to_excel(
                        combined_df
                    )

                output_name = (
                    os.path.splitext(
                        uploaded_file.name
                    )[0]
                    + "_extracted.xlsx"
                )

                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_data,
                    file_name=output_name,
                    mime=(
                        "application/vnd.openxmlformats-"
                        "officedocument.spreadsheetml.sheet"
                    ),
                    type="primary",
                    use_container_width=True
                )

                st.success(
                    "✅ Excel file is ready."
                )

                # Release dataframe after
                # download button is created.
                del combined_df
                gc.collect()

            else:

                st.warning(
                    "No extractable data was found."
                )

        except Exception as e:

            st.error(
                "❌ Extraction failed."
            )

            st.exception(e)

        finally:

            # ----------------------------------------
            # CLEAN TEMP DIRECTORY
            # ----------------------------------------

            try:

                if os.path.exists(
                    temp_dir
                ):

                    shutil.rmtree(
                        temp_dir,
                        ignore_errors=True
                    )

            except Exception:
                pass

            gc.collect()


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "PDF Fast Extractor • "
    "Cloud-safe 150-page batch architecture"
)