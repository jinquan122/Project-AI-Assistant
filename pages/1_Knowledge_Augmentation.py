import streamlit as st
import os
from app.controllers.llamaindex.ingestion_pipeline.pdf_ingestion import pdf_ingest

def main():
    st.set_page_config(page_title="Knowledge Augementation", page_icon="🌍")

    st.title("Upload PDF :sunglasses:")
    uploaded_files = st.file_uploader(":+1::+1::+1::+1::+1:", accept_multiple_files=True, type=["pdf"])
    folder_path = "./data"
    for uploaded_file in uploaded_files:
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        file_path = os.path.join(folder_path, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())

    file_names = [file for file in os.listdir(folder_path) if file != '.DS_Store']
    for file_name in file_names:
        st.write("filename:", file_name)

    if uploaded_files:
        with st.spinner("Uploading files..."):
            for uploaded_file in uploaded_files:
                pdf_ingest(uploaded_file.name)
            st.success(f"File(s) uploaded successfully to Qdrant!")
        st.switch_page("./AI_Asistant.py")
    
if __name__ == "__main__":
    main()