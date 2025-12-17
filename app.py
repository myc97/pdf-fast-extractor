import streamlit as st
import pandas as pd
import tempfile
from engine import extract_pdf

st.set_page_config(
    page_title="Fast PDF to Excel Extractor",
    layout="wide"
)

st.title("⚡ Ultra-Fast PDF → Excel Extractor")
st.caption("Handles 3000+ pages | Text + Scanned PDFs")

uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    with st.spinner("🚀 Processing PDF (Ultra Fast)..."):
        df, duration = extract_pdf(pdf_path)

    st.success(f"✅ Completed in {duration:.2f} seconds")
    st.dataframe(df.head(50), use_container_width=True)

    excel = df.to_excel(index=False)
    st.download_button(
        "📥 Download Excel",
        data=excel,
        file_name="extracted_data.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
