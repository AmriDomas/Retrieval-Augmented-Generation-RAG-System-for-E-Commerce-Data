# Import
import streamlit as st
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import openai
import fitz  # PyMuPDF

# ----------------- Backend -----------------
@st.cache_resource
def load_model():
    return SentenceTransformer('paraphrase-MiniLM-L6-v2')

model = load_model()

def build_faiss_index_cosine(texts):
    embeddings = model.encode(texts, convert_to_numpy=True)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings.astype('float32')
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index, embeddings

def retrieve(query, index, df, top_k=5):
    query_vec = model.encode([query], convert_to_numpy=True)
    query_vec = query_vec / np.linalg.norm(query_vec)
    query_vec = query_vec.astype('float32')
    scores, indices = index.search(query_vec, top_k)
    return df.iloc[indices[0]]

def generate_answer(query, context, api_key):
    openai.api_key = api_key
    system_message = "Kamu adalah asisten cerdas yang menjawab pertanyaan berdasarkan data yang diberikan."
    user_message = f"""Pertanyaan: {query}

Data yang relevan:
{context}"""
    response = openai.ChatCompletion.create(
        model="gpt-4.1-mini",
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message}
        ],
        temperature=0.7,
        max_tokens=1000
    )
    return response.choices[0]['message']["content"].strip()

def extract_text_from_pdfs(files):
    data = []
    for file in files:
        doc = fitz.open(stream=file.read(), filetype="pdf")
        for page_num, page in enumerate(doc):
            text = page.get_text()
            if text.strip():  # hanya ambil halaman yang tidak kosong
                data.append({
                    "source": file.name,
                    "page": page_num + 1,
                    "text": text.strip()
                })
    return pd.DataFrame(data)

def transform_csv(df, selected_columns, source_name):
    df["text"] = df[selected_columns].astype(str).agg(" | ".join, axis=1)
    df["source"] = source_name
    df["page"] = None
    return df[["source", "page", "text"]]

# ----------------- UI -----------------
st.title("📄🔍 RAG: CSV + PDF Multi-Input with LLM (Streamlit)")

# --- Sidebar Upload & API Key ---
st.sidebar.markdown("### 📥 Upload Files")
csv_files = st.sidebar.file_uploader("Upload CSV File(s)", type='csv', accept_multiple_files=True)
pdf_files = st.sidebar.file_uploader("Upload PDF File(s)", type='pdf', accept_multiple_files=True)

input_api_key = st.sidebar.text_input("🔑 Input OpenAI API Key", type='password')
button_api = st.sidebar.button("Activate API Key")

if 'api_key' not in st.session_state:
    st.session_state.api_key = None

if input_api_key and button_api:
    st.session_state.api_key = input_api_key
    st.sidebar.success("API Key Activated!")

# --- Process Uploaded Files ---
final_df = pd.DataFrame(columns=["source", "page", "text"])

# 1. Process CSV
if csv_files:
    for file in csv_files:
        try:
            df_csv = pd.read_csv(file, encoding='cp1252')
        except:
            df_csv = pd.read_csv(file)

        st.subheader(f"📄 Select Columns from {file.name}")
        selected_columns = st.multiselect(
            f"Select columns for: {file.name}",
            options=df_csv.columns.tolist(),
            default=df_csv.columns.tolist(),
            key=file.name
        )
        if selected_columns:
            transformed = transform_csv(df_csv, selected_columns, file.name)
            final_df = pd.concat([final_df, transformed], ignore_index=True)

# 2. Process PDFs
if pdf_files:
    st.subheader("📑 Preview of Extracted PDF Content")
    df_pdf = extract_text_from_pdfs(pdf_files)
    if not df_pdf.empty:
        st.dataframe(df_pdf[["source", "page", "text"]].head())
        final_df = pd.concat([final_df, df_pdf], ignore_index=True)

# --- Run RAG if data available ---
if not final_df.empty:
    query = st.text_input("🧠 Enter your question")
    run_query = st.button("Answer the Question")

    if run_query:
        if not st.session_state.api_key:
            st.warning("⚠️ Please activate your API Key first.")
        else:
            try:
                index, _ = build_faiss_index_cosine(final_df["text"].tolist())

                with st.spinner("🔎 Searching for relevant context..."):
                    results = retrieve(query, index, final_df)
                    context = "\n\n".join(results["text"].tolist())

                with st.spinner("✍️ Generating answer..."):
                    answer = generate_answer(query, context, st.session_state.api_key)

                st.subheader("🧾 Answer:")
                st.success(answer)

                st.markdown("### 📌 Source(s):")
                st.write(results[["source", "page"]])

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
else:
    st.info("⬅️ Please upload at least one CSV or PDF file.")
