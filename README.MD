# 🧠 Dr. X Research Assistant - RAG NLP Pipeline

## 🚀 Overview
This project implements a complete **Retrieval-Augmented Generation (RAG)** pipeline to investigate the enigmatic research of Dr. X. It processes a mix of scientific documents, embeds them in a local vector store, and allows for natural language question-answering, summarization, and translation — all locally, GPU-accelerated with CUDA 11.8.

---

## 📂 Project Structure
```
├── models/                     # Local LLaMA + translation models
├── Dr.X Files/                 # Input documents (pdf, docx, xlsx, csv...)
├── vector_store/               # Persistent ChromaDB vector store
├── OSOS Assignment.ipynb       # Full implementation pipeline
├── requirements.txt            # All Python dependencies
├── README.md                   # This file
└── report.pdf                  # 6-page documentation
```

---

## 🧱 Pipeline Components

### 📥 1. Document Extraction
- Supports: `.pdf`, `.docx`, `.csv`, `.xlsx`, `.xls`, `.xlsm`
- Tables are detected using line/tab heuristics and marked as `type: table`
- File content flattened and cleaned using:
  - `PyMuPDF` (PDF)
  - `docx2txt` (Word)
  - `pandas` (Spreadsheets)

### 🧩 2. Chunking
- **Sentence-aware chunking** using `nltk.sent_tokenize`
- Token-safe boundaries using `tiktoken` + overlap logic
- Metadata includes: `source`, `page`, `chunk_number`, `type`

### 🧠 3. Embedding & Vector DB
- Uses `SentenceTransformer("nomic-ai/nomic-embed-text-v1.5)`
- Automatically uses GPU with CUDA 11.8
- Chunks stored in `ChromaDB` with full metadata

### ❓ 4. RAG Q&A System
- Embeds query → retrieves relevant chunks → generates response with LLaMA
- Uses `llama3-8B.gguf` via `llama.cpp`
- Supports `verbose=True` to show retrieved chunks + answer
- Remembers past questions with `previous_qas`
- Detects data-heavy queries (e.g., tables) for smarter chunk selection (WIP)

### 🌍 5. Translation Tool
- Detects source language using `langdetect`
- Attempts LLaMA-based prompt translation
- Falls back to HuggingFace Transformers if needed
- Optional grammar polishing using post-translation prompt

### 📝 6. Summarization
- Prompt styles: `default`, `insight`, `overview`
- Uses LLaMA locally

### 📊 7. Evaluation
- ROUGE-1, ROUGE-2, ROUGE-L (via `rouge_score`)
- Manual QA scoring available (Relevance, Accuracy, Fluency, Confidence)
- Tokens/sec performance logs for all major stages

### 🧹 8. VRAM & Debugging Tools
- Manual GPU cleanup using `torch.cuda.empty_cache()`
- Model reloading utility with `reload_llm()`
- Display top retrieved chunks with `verbose=True`

---

## 🛠️ Setup Instructions
```bash
# Clone repository

# It prefeared to create pip environment:
python -m venv env
env/Scripts/activate.ps1 # for Linux: source env/bin/activate

# Install the requirments
pip install -r requirements.txt

# Ensure CUDA 11.8 is available
# Ensure GPU acceleration availability for llama.cpp & torch

# Download llama3-8B.gguf model  https://huggingface.co/mradermacher/llama3-8B-DarkIdol-2.2-Uncensored-1048K-GGUF/resolve/main/llama3-8B-DarkIdol-2.2-Uncensored-1048K.Q4_K_M.gguf

# Place 'llama3-8B.gguf' in ./models/
# Run the notebook
```

---

## 📈 Example Usage
```python
reset_session()

# Ask a question with debug output
response = ask_question("What were Dr. X’s findings on gene expression?", verbose=True)
print(response)

# Translate answer to Arabic
print(translate_text(response, target_lang="ar"))

# Summarize response
print(summarize_text(response, strategy="insight"))
```

---

## ⚙️ Available Utilities
- `reset_session()` – clear QA memory
- `evaluate_qa_manual()` – manual scoring tool
- `refine_translation()` – clean up translated grammar
- `display_retrieved_chunks()` – see source metadata
- `reload_llm()` – safely reload LLaMA & free VRAM

---


## 📬 Author & Credits
Hamood Al-Shabanoti
Built as part of the OSOS AI Technical Challenge.  


