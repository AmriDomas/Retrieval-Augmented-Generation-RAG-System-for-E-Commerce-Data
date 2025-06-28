# 🧠 Retrieval-Augmented Generation (RAG) System for E-Commerce Data

This project implements a complete **RAG-based question-answering system** using Streamlit, FAISS, Sentence Transformers, and OpenAI's GPT model. Users can upload an e-commerce dataset (CSV format), ask questions in natural language, and get answers generated based on the most relevant data rows.

---

## 🚀 Features

* 📂 **Upload CSV file** via Streamlit interface
* 📊 **Column selection** for text embedding
* 🧠 **Text embedding** using `SentenceTransformer` (MiniLM)
* ⚙️ **FAISS indexing** with cosine similarity
* 🔎 **Top-k semantic retrieval** based on user queries
* 💬 **Answer generation** with OpenAI GPT (e.g. `gpt-4.1-mini`)
* 🔐 **API key input** and session management

---

## 📁 Folder Structure

```
project/
├── app.py                   # Main Streamlit application
├── data/                    # Folder to store input CSV files
│   └── US  E-commerce records 2020.csv
├── requirements.txt         # Python dependencies
└── README.md                # This documentation
```

---

## ⚙️ Installation & Setup

1. **Clone this repository**

```bash
git clone https://github.com/your-username/rag-ecommerce-streamlit.git
cd rag-ecommerce-streamlit
```

2. **Create and activate virtual environment (optional)**

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Run the Streamlit app**

```bash
streamlit run app.py
```

---

## 🔑 OpenAI API Key Setup

To use GPT models, you need an [OpenAI API key](https://platform.openai.com/account/api-keys).
You will be prompted to enter it in the Streamlit sidebar.

---

## 📌 How It Works

### Step 1: Upload CSV File

* User uploads e-commerce data (e.g., US E-commerce 2020).

### Step 2: Column Selection

* User selects relevant columns to include in semantic search.

### Step 3: Embedding & Indexing

* The app creates sentence embeddings and normalizes them for cosine similarity.
* FAISS is used to index and search for top-k similar entries.

### Step 4: Ask a Question

* User types a question in natural language (e.g., "What segment buys the most in California?").
* The app encodes the question, retrieves relevant rows, and feeds both into OpenAI's GPT to generate an answer.

---

## ✅ Conclusion

This project demonstrates how to combine semantic search with generative models to build a smart assistant that answers questions based on structured data. It bridges classical information retrieval (FAISS) with modern LLMs for flexible business intelligence.

> Can be extended to internal company documents, reports, or customer feedback.

---

## 📌 To-Do (Next Steps)

* [ ] Add documet like pdf or website
* [ ] Add answer highlighting from source rows
* [ ] Filter input data by condition (e.g., location)
* [ ] Store query history in local/session storage
* [ ] Add download/export option for retrieved answers

## 🙋‍♀️ Contributing

Pull requests are welcome! For major changes, please open an issue first to discuss what you would like to change.

---

## 📬 Contact

Created by \[Your Name] - feel free to reach out via [[LinkedIn](https://linkedin.com](http://linkedin.com/in/muh-amri-sidiq)) or open an issue.

---
