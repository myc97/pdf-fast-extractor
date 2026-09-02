import streamlit as st
import pdfplumber
import pandas as pd
import tempfile
import os
import time
import io
from multiprocessing import Pool, cpu_count

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
# SETTINGS
# ============================================================

DEFAULT_BATCH_SIZE = 150

# Streamlit Cloud normally has limited CPU/RAM.
# 2 is a safer default than using all CPUs.
DEFAULT_PROCESSES = min(2, cpu_count())

MAX_BATCH_SIZE = 150


# ============================================================
# PAGE STYLE
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 32px;
    font-weight: 700;
    margin-bottom: 5px;
}

.sub-title {
    color: #666;
    margin-bottom: 20px;
}

.info-box {
    padding: 12px;
    border-radius: 8px;
    background-color: #f5f7fa;
    margin-bottom: 10px;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# PDF TYPE DETECTION
# ============================================================

def detect_pdf_type(pdf_path):

    try:

        with pdfplumber.open(pdf_path) as pdf:

            total_pages = len(pdf.pages)

            pages_to_check = min(3, total_pages)

            text_found = 0

            for i in range(pages_to_check):

                try:

                    text = pdf.pages[i].extract_text()

                    if text and len(text.strip()) > 20:
                        text_found += 1

                except Exception:
                    pass

            if text_found > 0:
                return "searchable", total_pages

            return "scanned", total_pages

    except Exception:

        return "scanned", 0


# ============================================================
# SEARCHABLE PDF - TABLE EXTRACTION
# ============================================================

def process_searchable_batch(args):

    pdf_path, page_numbers = args

    batch_data = []

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_number in page_numbers:

                try:

                    page = pdf.pages[page_number]

                    tables = page.extract_tables()

                    for table in tables:

                        if not table:
                            continue

                        df = pd.DataFrame(table)

                        # Remove completely empty rows
                        df = df.dropna(how="all")

                        if df.empty:
                            continue

                        df["Page"] = page_number + 1

                        batch_data.append(df)

                except Exception:

                    continue

    except Exception:

        pass

    return batch_data


# ============================================================
# SEARCHABLE PDF - TEXT FALLBACK
# ============================================================

def process_text_batch(args):

    pdf_path, page_numbers = args

    batch_data = []

    try:

        with pdfplumber.open(pdf_path) as pdf:

            for page_number in page_numbers:

                try:

                    page = pdf.pages[page_number]

                    text = page.extract_text()

                    if not text:
                        continue

                    rows = []

                    for line in text.splitlines():

                        line = line.strip()

                        if not line:
                            continue

                        # Keep whitespace-separated values
                        row = line.split()

                        if row:
                            rows.append(row)

                    if rows:

                        max_columns = max(len(row) for row in rows)

                        normalized_rows = []

                        for row in rows:

                            row = row + [""] * (
                                max_columns - len(row)
                            )

                            normalized_rows.append(row)

                        df = pd.DataFrame(normalized_rows)

                        df["Page"] = page_number + 1

                        batch_data.append(df)

                except Exception:

                    continue

    except Exception:

        pass

    return batch_data


# ============================================================
# OCR BATCH
# ============================================================

def process_ocr_batch(args):

    pdf_path, page_numbers, dpi = args

    batch_data = []

    try:

        for page_number in page_numbers:

            try:

                # IMPORTANT:
                # Convert only ONE page at a time.
                # This prevents huge RAM usage for 150-page OCR batches.

                images = convert_from_path(
                    pdf_path,
                    dpi=dpi,
                    first_page=page_number + 1,
                    last_page=page_number + 1,
                    use_cropbox=True,
                    thread_count=1
                )

                if not images:
                    continue

                image = images[0]

                text = pytesseract.image_to_string(
                    image,
                    lang="eng"
                )

                del image
                del images

                if not text:
                    continue

                rows = []

                for line in text.splitlines():

                    line = line.strip()

                    if not line:
                        continue

                    row = line.split()

                    if row:
                        rows.append(row)

                if rows:

                    max_columns = max(
                        len(row) for row in rows
                    )

                    normalized_rows = []

                    for row in rows:

                        row = row + [""] * (
                            max_columns - len(row)
                        )

                        normalized_rows.append(row)

                    df = pd.DataFrame(
                        normalized_rows
                    )

                    df["Page"] = page_number + 1

                    batch_data.append(df)

            except Exception:

                continue

    except Exception:

        pass

    return batch_data


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataframe(df):

    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # Convert NaN to blank
    df = df.fillna("")

    # Convert everything to string
    for column in df.columns:
        df[column] = df[column].astype(str)

    # Remove completely empty rows
    df = df[
        df.apply(
            lambda row:
            any(str(x).strip() for x in row),
            axis=1
        )
    ]

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df.reset_index(drop=True)


# ============================================================
# COMBINE DATA
# ============================================================

def combine_data(all_batches):

    valid_data = []

    for batch in all_batches:

        if not batch:
            continue

        for df in batch:

            if df is None:
                continue

            if df.empty:
                continue

            df = clean_dataframe(df)

            if not df.empty:
                valid_data.append(df)

    if not valid_data:
        return pd.DataFrame()

    # Find maximum number of columns
    max_columns = max(
        len(df.columns)
        for df in valid_data
    )

    normalized = []

    for df in valid_data:

        df = df.copy()

        # Rename columns temporarily
        df.columns = [
            f"Column_{i+1}"
            for i in range(len(df.columns))
        ]

        while len(df.columns) < max_columns:

            df[
                f"Column_{len(df.columns)+1}"
            ] = ""

        normalized.append(df)

    combined = pd.concat(
        normalized,
        ignore_index=True
    )

    return combined


# ============================================================
# EXCEL CREATION
# ============================================================

def dataframe_to_excel(df):

    output = io.BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        df.to_excel(
            writer,
            index=False,
            sheet_name="Extracted Data"
        )

    output.seek(0)

    return output


# ============================================================
# CREATE BATCHES
# ============================================================

def create_batches(total_pages, batch_size):

    batches = []

    for start in range(
        0,
        total_pages,
        batch_size
    ):

        end = min(
            start + batch_size,
            total_pages
        )

        page_numbers = list(
            range(start, end)
        )

        batches.append(page_numbers)

    return batches


# ============================================================
# MAIN EXTRACTION
# ============================================================

def run_extraction(
    pdf_path,
    extraction_mode,
    batch_size,
    max_processes,
    ocr_dpi,
    progress_bar,
    status_text,
    metrics_placeholder
):

    start_time = time.time()

    # --------------------------------------------------------
    # DETECT PDF
    # --------------------------------------------------------

    status_text.info(
        "🔍 Detecting PDF structure..."
    )

    detected_type, total_pages = detect_pdf_type(
        pdf_path
    )

    if total_pages == 0:

        raise Exception(
            "Unable to read PDF."
        )

    # --------------------------------------------------------
    # AUTOMATIC MODE
    # --------------------------------------------------------

    if extraction_mode == "Automatic":

        if detected_type == "searchable":
            selected_mode = "Table Extraction"
        else:
            selected_mode = "OCR"

    else:

        selected_mode = extraction_mode

    # --------------------------------------------------------
    # CREATE BATCHES
    # --------------------------------------------------------

    batches = create_batches(
        total_pages,
        batch_size
    )

    total_batches = len(batches)

    # --------------------------------------------------------
    # SHOW INFO
    # --------------------------------------------------------

    status_text.success(
        f"PDF detected: "
        f"{'Searchable' if detected_type == 'searchable' else 'Scanned'}"
    )

    metrics_placeholder.markdown(
        f"""
        **Total Pages:** {total_pages:,}  
        **Batch Size:** {batch_size} pages  
        **Total Batches:** {total_batches}  
        **Extraction:** {selected_mode}  
        **Processes:** {max_processes}
        """
    )

    # --------------------------------------------------------
    # PREPARE ARGUMENTS
    # --------------------------------------------------------

    args_list = []

    if selected_mode == "OCR":

        for pages in batches:

            args_list.append(
                (
                    pdf_path,
                    pages,
                    ocr_dpi
                )
            )

        processor = process_ocr_batch

    elif selected_mode == "Table Extraction":

        for pages in batches:

            args_list.append(
                (
                    pdf_path,
                    pages
                )
            )

        processor = process_searchable_batch

    else:

        for pages in batches:

            args_list.append(
                (
                    pdf_path,
                    pages
                )
            )

        processor = process_text_batch

    # --------------------------------------------------------
    # PROCESS BATCHES
    # --------------------------------------------------------

    all_batches = []

    completed_batches = 0

    batch_times = []

    # We deliberately use a limited number of workers
    # for Streamlit Cloud stability.

    with Pool(
        processes=max_processes
    ) as pool:

        results = pool.imap_unordered(
            processor,
            args_list
        )

        for result in results:

            batch_end_time = time.time()

            completed_batches += 1

            all_batches.append(result)

            # ------------------------------------------------
            # TIME CALCULATION
            # ------------------------------------------------

            elapsed = (
                batch_end_time - start_time
            )

            average_batch_time = (
                elapsed / completed_batches
            )

            remaining_batches = (
                total_batches -
                completed_batches
            )

            eta = (
                remaining_batches *
                average_batch_time
            )

            batch_times.append(
                average_batch_time
            )

            # ------------------------------------------------
            # PAGE RANGE
            # ------------------------------------------------

            # imap_unordered means result order is not guaranteed.
            # We therefore show batch completion rather than
            # pretending the current result is a specific batch.

            progress = (
                completed_batches /
                total_batches
            )

            progress_bar.progress(
                progress
            )

            status_text.info(
                f"⚙️ Processing batch "
                f"{completed_batches}/{total_batches}"
            )

            metrics_placeholder.markdown(
                f"""
                **Progress:** {progress * 100:.1f}%  
                **Completed Batches:** {completed_batches}/{total_batches}  
                **Pages:** {total_pages:,}  
                **Elapsed:** {elapsed:.1f} sec  
                **Estimated Remaining:** {eta:.1f} sec
                """
            )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    status_text.info(
        "📊 Combining extracted data..."
    )

    combined_df = combine_data(
        all_batches
    )

    # --------------------------------------------------------
    # FINAL TIME
    # --------------------------------------------------------

    total_time = (
        time.time() - start_time
    )

    progress_bar.progress(1.0)

    if combined_df.empty:

        status_text.warning(
            "⚠️ No data was extracted."
        )

        return None, total_pages, total_time

    status_text.success(
        f"✅ Extraction completed in "
        f"{total_time:.1f} seconds"
    )

    return (
        combined_df,
        total_pages,
        total_time
    )


# ============================================================
# USER INTERFACE
# ============================================================

st.markdown(
    '<div class="main-title">📄 PDF Fast Extractor</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="sub-title">'
    'Batch-wise PDF → Excel extraction for large files'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.header("⚙️ Settings")

extraction_mode = st.sidebar.selectbox(
    "Extraction Mode",
    [
        "Automatic",
        "Table Extraction",
        "OCR"
    ]
)

batch_size = st.sidebar.number_input(
    "Batch Size",
    min_value=25,
    max_value=150,
    value=DEFAULT_BATCH_SIZE,
    step=25
)

max_processes = st.sidebar.number_input(
    "Processes",
    min_value=1,
    max_value=min(4, cpu_count()),
    value=DEFAULT_PROCESSES,
    step=1
)

ocr_dpi = st.sidebar.selectbox(
    "OCR DPI",
    [
        150,
        200,
        250,
        300
    ],
    index=1
)


# ============================================================
# UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)


# ============================================================
# PROCESS
# ============================================================

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    file_size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.write(
        f"**File Size:** {file_size_mb:.2f} MB"
    )

    # --------------------------------------------------------
    # TEMP FILE
    # --------------------------------------------------------

    temp_file = tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    )

    temp_file.write(
        uploaded_file.getbuffer()
    )

    temp_file.close()

    # --------------------------------------------------------
    # DETECT
    # --------------------------------------------------------

    if st.button(
        "🚀 Start Extraction",
        type="primary",
        use_container_width=True
    ):

        progress_bar = st.progress(0)

        status_text = st.empty()

        metrics_placeholder = st.empty()

        try:

            df, total_pages, total_time = (
                run_extraction(
                    temp_file.name,
                    extraction_mode,
                    int(batch_size),
                    int(max_processes),
                    int(ocr_dpi),
                    progress_bar,
                    status_text,
                    metrics_placeholder
                )
            )

            if df is not None and not df.empty:

                st.subheader(
                    "📊 Extraction Result"
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "Pages",
                        f"{total_pages:,}"
                    )

                with col2:
                    st.metric(
                        "Rows",
                        f"{len(df):,}"
                    )

                with col3:
                    st.metric(
                        "Columns",
                        f"{len(df.columns):,}"
                    )

                st.subheader(
                    "Preview"
                )

                st.dataframe(
                    df.head(100),
                    use_container_width=True
                )

                # ------------------------------------------------
                # EXCEL
                # ------------------------------------------------

                excel_file = dataframe_to_excel(
                    df
                )

                output_name = (
                    os.path.splitext(
                        uploaded_file.name
                    )[0]
                    + "_extracted.xlsx"
                )

                st.download_button(
                    label="⬇️ Download Excel",
                    data=excel_file,
                    file_name=output_name,
                    mime=(
                        "application/vnd.openxmlformats-officedocument."
                        "spreadsheetml.sheet"
                    ),
                    use_container_width=True
                )

        except Exception as e:

            st.error(
                f"❌ Extraction failed: {str(e)}"
            )

        finally:

            try:
                os.remove(
                    temp_file.name
                )
            except Exception:
                pass
