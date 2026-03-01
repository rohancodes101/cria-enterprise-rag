# CRIA: Enterprise Retrieval-Augmented Generation System

**CRIA (Change Record Intelligent Assistant)** is an enterprise-grade RAG system designed to automate the creation of structured Change Requests. By leveraging semantic retrieval and grounded LLM generation, CRIA transforms vague natural language intent into production-ready, governance-compliant records.



### 🛡️ Core Principles
* **Fully Offline:** Zero cloud API dependencies.
* **Data Sovereignty:** No external data transfer; runs entirely on local infrastructure.
* **Hallucination Control:** Grounded generation using historical change patterns.

---

## 🎥 Demo & Insights
* **Demo Video:** [Watch the Walkthrough](YOUR_LOOM_LINK)
* **Technical Deep Dive:** [Read the Blog Post](YOUR_MEDIUM_LINK)

---

## 🏗️ Architecture & Tech Stack

CRIA utilizes a modular pipeline to ensure high precision and low latency on commodity hardware:

* **Embedding Model:** `IBM Granite Embedding` (GGUF, Q4_K_M) for semantic representation.
* **Vector Store:** `FAISS (IndexFlatL2)` for high-performance similarity search.
* **Inference Engine:** `llama.cpp` supporting `IBM Granite Instruct` (Quantized).
* **UI Framework:** `Gradio` for an enterprise-styled web interface.
* **Execution:** CPU-only inference optimized for 16GB RAM environments.

---

## ⚙️ How It Works



1.  **Semantic Embedding:** Historical change records are vectorized into a dense multidimensional space.
2.  **Vector Retrieval:** When a user inputs an intent, FAISS identifies the **Top-K** most similar historical records based on L2 distance.
3.  **Pattern Aggregation:** The system extracts implementation steps, risk factors, and backout plans from retrieved metadata.
4.  **Grounded Generation:** The LLM receives the user intent + historical context. With a low temperature (**0.2**), it synthesizes a new, structured record based strictly on provided patterns.

---

## 📂 Project Structure

```text
cria-enterprise-rag/
├── architecture/           # System design diagrams
├── screenshots/            # UI/UX previews
├── embedder.py             # Vectorization logic
├── retriever.py            # FAISS search implementation
├── ui.py                   # Gradio interface
├── dataset_synthetic.json  # Sample enterprise data
└── requirements.txt        # Dependency manifest
```
## 🔐 Security & Governance

- **Local Inference:** All computations occur on-premise.  
- **Traceability:** Every generated record includes references to the source historical changes used for grounding.  
- **Zero Data Exfiltration:** No telemetry or external API calls.  

---

## 📬 Connect

If you're working on enterprise AI, RAG systems, or vector search infrastructure, feel free to reach out.

**MALLESH C N** | **[[LinkedIn Profile](https://www.linkedin.com/in/malleshcn/)]** | **[Portfolio[Here](https://malleshcn.netlify.app/)]**

---
