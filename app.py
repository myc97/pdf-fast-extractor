import streamlit as st
import pdfplumber
import pandas as pd
import tempfile
import os
import time
import io

# OCR imports
from pdf2image import convert_from_path
import pytesseract


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="PDF to Excel Converter",
    page_icon="📄",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown("""
<style>

.main-title {
    font-size: 36px;
    font-weight: 700;
    text-align: center;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    color: #666;
    margin-bottom: 30px;
}

.success-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #e8f5e9;
    border: 1px solid #81c784;
}

.info-box {
    padding: 15px;
    border-radius: 10px;
    background-color: #e3f2fd;
    border: 1px solid #64b5f6;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# TITLE
# ============================================================

st.markdown(
    '<div class="main-title">📄 PDF → Excel Converter</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Upload a PDF, extract the data, and download it as Excel.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Settings")

    extraction_mode = st.selectbox(
        "Extraction Mode",
        [
            "Automatic",
            "Table Extraction",
            "OCR"
        ]
    )

    ocr_dpi = st.selectbox(
        "OCR Quality",
        [150, 200, 250, 300],
        index=1
    )

    st.markdown("---")

    st.markdown("""
    ### Supported PDFs

    ✅ Searchable PDFs  
    ✅ Table PDFs  
    ✅ Scanned PDFs  
    ✅ OCR PDFs  

    ### Output

    📊 Excel `.xlsx`
    """)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"],
    help="Upload the PDF that you want to convert to Excel."
)


# ============================================================
# PDF TYPE DETECTION
# ============================================================

def detect_pdf_type(pdf_path):

    try:

        with pdfplumber.open(pdf_path) as pdf:

            total_pages = len(pdf.pages)

            pages_to_check = min(3, total_pages)

            text_pages = 0

            for i in range(pages_to_check):

                try:

                    text = pdf.pages[i].extract_text()

                    if text and text.strip():

                        text_pages += 1

                except Exception:
                    pass

            if text_pages > 0:
                return "Searchable PDF"

            return "Scanned PDF"

    except Exception:

        return "Unknown"


# ============================================================
# TABLE EXTRACTION
# ============================================================

def extract_tables(pdf_path, progress_bar=None):

    all_data = []

    page_info = []

    with pdfplumber.open(pdf_path) as pdf:

        total_pages = len(pdf.pages)

        for page_number, page in enumerate(pdf.pages, start=1):

            try:

                tables = page.extract_tables()

                page_tables = 0

                for table in tables:

                    if not table:
                        continue

                    df = pd.DataFrame(table)

                    # Remove completely empty rows
                    df = df.dropna(
                        how="all"
                    )

                    if df.empty:
                        continue

                    # Add source page
                    df["Page"] = page_number

                    all_data.append(df)

                    page_tables += 1

                page_info.append(
                    f"Page {page_number}: {page_tables} table(s)"
                )

            except Exception as e:

                page_info.append(
                    f"Page {page_number}: Error"
                )

            if progress_bar:

                progress_bar.progress(
                    page_number / total_pages,
                    text=f"Extracting page {page_number}/{total_pages}"
                )

    return all_data, page_info


# ============================================================
# TEXT EXTRACTION
# ============================================================

def extract_text_as_rows(pdf_path, progress_bar=None):

    all_data = []

    with pdfplumber.open(pdf_path) as pdf:

        total_pages = len(pdf.pages)

        for page_number, page in enumerate(
            pdf.pages,
            start=1
        ):

            try:

                text = page.extract_text()

                if not text:
                    continue

                lines = text.splitlines()

                rows = []

                for line in lines:

                    line = line.strip()

                    if not line:
                        continue

                    rows.append(line.split())

                if rows:

                    df = pd.DataFrame(rows)

                    df["Page"] = page_number

                    all_data.append(df)

            except Exception:

                pass

            if progress_bar:

                progress_bar.progress(
                    page_number / total_pages,
                    text=f"Reading page {page_number}/{total_pages}"
                )

    return all_data


# ============================================================
# OCR EXTRACTION
# ============================================================

def extract_ocr(pdf_path, dpi=200, progress_bar=None):

    all_data = []

    # Find PDF page count first
    with pdfplumber.open(pdf_path) as pdf:

        total_pages = len(pdf.pages)

    for page_number in range(1, total_pages + 1):

        try:

            images = convert_from_path(
                pdf_path,
                dpi=dpi,
                first_page=page_number,
                last_page=page_number,
                thread_count=1
            )

            if not images:
                continue

            image = images[0]

            text = pytesseract.image_to_string(
                image,
                lang="eng"
            )

            lines = text.splitlines()

            rows = []

            for line in lines:

                line = line.strip()

                if not line:
                    continue

                # Keep OCR columns separated
                rows.append(line.split())

            if rows:

                df = pd.DataFrame(rows)

                df["Page"] = page_number

                all_data.append(df)

        except Exception as e:

            st.warning(
                f"OCR error on page {page_number}: {e}"
            )

        if progress_bar:

            progress_bar.progress(
                page_number / total_pages,
                text=f"OCR page {page_number}/{total_pages}"
            )

    return all_data


# ============================================================
# CLEAN DATA
# ============================================================

def clean_dataframe(df):

    # Convert everything to string
    df = df.fillna("")

    df = df.astype(str)

    # Remove completely empty columns
    df = df.loc[
        :,
        (df != "").any(axis=0)
    ]

    # Remove completely empty rows
    df = df.loc[
        (df != "").any(axis=1)
    ]

    # Reset index
    df = df.reset_index(drop=True)

    return df


# ============================================================
# COMBINE DATA
# ============================================================

def combine_data(all_data):

    if not all_data:

        return pd.DataFrame()

    cleaned = []

    for df in all_data:

        try:

            df = clean_dataframe(df)

            if not df.empty:

                cleaned.append(df)

        except Exception:

            pass

    if not cleaned:

        return pd.DataFrame()

    # Find maximum number of columns
    max_columns = max(
        len(df.columns)
        for df in cleaned
    )

    normalized = []

    for df in cleaned:

        df = df.copy()

        # Rename columns consistently
        df.columns = [
            f"Column_{i+1}"
            for i in range(len(df.columns))
        ]

        # Add missing columns
        for i in range(
            len(df.columns) + 1,
            max_columns + 1
        ):

            df[f"Column_{i}"] = ""

        # Correct column order
        df = df[
            [
                f"Column_{i}"
                for i in range(1, max_columns + 1)
            ]
        ]

        normalized.append(df)

    return pd.concat(
        normalized,
        ignore_index=True
    )


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

        worksheet = writer.sheets[
            "Extracted Data"
        ]

        # Freeze header
        worksheet.freeze_panes = "A2"

        # Auto width
        for column_cells in worksheet.columns:

            max_length = 0

            column_letter = column_cells[0].column_letter

            for cell in column_cells:

                try:

                    value_length = len(
                        str(cell.value)
                    )

                    max_length = max(
                        max_length,
                        value_length
                    )

                except Exception:

                    pass

            worksheet.column_dimensions[
                column_letter
            ].width = min(
                max_length + 2,
                50
            )

    output.seek(0)

    return output


# ============================================================
# MAIN PROCESS
# ============================================================

if uploaded_file:

    st.success(
        f"Uploaded: {uploaded_file.name}"
    )

    file_size_mb = (
        uploaded_file.size /
        (1024 * 1024)
    )

    st.info(
        f"File size: {file_size_mb:.2f} MB"
    )

    # Save uploaded PDF temporarily
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=".pdf"
    ) as tmp:

        tmp.write(
            uploaded_file.getbuffer()
        )

        pdf_path = tmp.name

    # --------------------------------------------------------
    # Detect PDF
    # --------------------------------------------------------

    with st.spinner("🔍 Detecting PDF type..."):

        pdf_type = detect_pdf_type(
            pdf_path
        )

    st.info(
        f"Detected: **{pdf_type}**"
    )

    # --------------------------------------------------------
    # Start button
    # --------------------------------------------------------

    if st.button(
        "🚀 Start Extraction",
        type="primary",
        use_container_width=True
    ):

        start_time = time.time()

        progress_bar = st.progress(
            0,
            text="Starting..."
        )

        all_data = []

        # ====================================================
        # AUTOMATIC MODE
        # ====================================================

        if extraction_mode == "Automatic":

            if pdf_type == "Searchable PDF":

                st.write(
                    "📊 Using table extraction..."
                )

                all_data, page_info = extract_tables(
                    pdf_path,
                    progress_bar
                )

                # If no tables found, fallback
                # to normal text extraction

                if not all_data:

                    st.warning(
                        "No tables detected. "
                        "Trying text extraction..."
                    )

                    all_data = extract_text_as_rows(
                        pdf_path,
                        progress_bar
                    )

            else:

                st.write(
                    "🖨️ Using OCR..."
                )

                all_data = extract_ocr(
                    pdf_path,
                    dpi=ocr_dpi,
                    progress_bar=progress_bar
                )

        # ====================================================
        # TABLE MODE
        # ====================================================

        elif extraction_mode == "Table Extraction":

            st.write(
                "📊 Extracting tables..."
            )

            all_data, page_info = extract_tables(
                pdf_path,
                progress_bar
            )

        # ====================================================
        # OCR MODE
        # ====================================================

        elif extraction_mode == "OCR":

            st.write(
                "🖨️ Running OCR..."
            )

            all_data = extract_ocr(
                pdf_path,
                dpi=ocr_dpi,
                progress_bar=progress_bar
            )

        # ====================================================
        # COMBINE
        # ====================================================

        progress_bar.progress(
            100,
            text="Finalizing Excel..."
        )

        final_df = combine_data(
            all_data
        )

        elapsed = time.time() - start_time

        # ====================================================
        # RESULT
        # ====================================================

        if not final_df.empty:

            st.success(
                f"✅ Extraction completed in "
                f"{elapsed:.2f} seconds"
            )

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Rows",
                len(final_df)
            )

            col2.metric(
                "Columns",
                len(final_df.columns)
            )

            col3.metric(
                "Time",
                f"{elapsed:.1f}s"
            )

            st.subheader(
                "📊 Extracted Data Preview"
            )

            st.dataframe(
                final_df.head(100),
                use_container_width=True,
                height=500
            )

            # =================================================
            # CREATE EXCEL
            # =================================================

            excel_file = dataframe_to_excel(
                final_df
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
                    "application/vnd.openxmlformats-"
                    "officedocument.spreadsheetml.sheet"
                ),
                type="primary",
                use_container_width=True
            )

        else:

            st.error(
                "❌ No data could be extracted from this PDF."
            )

            st.warning(
                "Try OCR mode if the PDF is scanned "
                "or image-based."
            )

    # Clean temporary file
    try:
        if os.path.exists(pdf_path):
            os.remove(pdf_path)
    except Exception:
        pass


else:

    st.markdown(
        """
        <div class="info-box">

        ### 👆 Upload your PDF

        The application will:

        1. Detect the PDF type
        2. Extract table/text data
        3. Use OCR for scanned PDFs
        4. Convert the result to Excel
        5. Show a preview
        6. Provide an Excel download button

        </div>
        """,
        unsafe_allow_html=True
    )