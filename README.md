# Enterprise AI Knowledge Assistant

An AI-powered knowledge assistant built with **Python**, **LangChain**, and **Large Language Models (LLMs)**. This project is developed during my AI Engineering internship and is evolving into a complete enterprise-ready AI assistant with Retrieval-Augmented Generation (RAG), Tool Calling, MCP, and AI Agent capabilities.

---

## Features

Current features:

- Chat with LLM (Groq - `llama-3.3-70b-versatile`)
- Retrieval-Augmented Generation (RAG) over Qdrant Cloud
- Semantic chunking pipeline (menu items, menu options, FAQ, promotions)
- Multilingual embedding via FastEmbed (no separate embedding API needed)
- Conversation Memory (with history trimming)
- Persona-driven System Prompt (refuses to answer outside the knowledge base)
- RAG Evaluation Framework (retriever accuracy, context precision, answer accuracy, hallucination rate, latency)
- Environment Configuration (`.env`)

Planned features:

- PDF Upload
- Multi-document Search
- Source Citation
- Tool Calling
- Model Context Protocol (MCP)
- AI Agent
- Spring AI Integration

---

## Tech Stack

- Python
- LangChain
- Groq API
- Qdrant Cloud (vector database)
- FastEmbed (local multilingual embeddings)
- python-dotenv

Future Stack:

- FastAPI / Spring Boot
- Spring AI
- LiteLLM
- MariaDB

---

## Project Structure

```
ai-knowledge-assistant/
│
├── chatbot.py               # Chatbot chính: Persona + RAG + Conversation Memory (ask())
├── rag.py                   # Truy vấn (retrieve) từ Qdrant
├── ingest.py                # Chunk dữ liệu + nạp vào Qdrant
├── evaluate.py               # Đánh giá tự động chất lượng RAG (retrieval + generation)
├── test_cases.json          # Bộ 20 câu hỏi mẫu để chạy evaluate.py
├── evaluation_report.csv    # Báo cáo chi tiết sinh ra sau mỗi lần evaluate.py chạy (không commit)
│
├── data/
│   ├── menu.json             # Dữ liệu món (dùng để ingest)
│   ├── menu.md                # Chỉ phần "Tùy chọn đường/đá/topping" được ingest
│   ├── faq.md                 # FAQ (dùng để ingest)
│   └── promotions.md          # Khuyến mãi/thành viên (dùng để ingest)
│
├── requirements.txt
├── .env.example               # GROQ_API_KEY, QDRANT_URL, QDRANT_API_KEY
├── .gitignore
└── README.md
```

---

## Installation

Clone the repository

```bash
git clone https://github.com/doanminhphuc2011-debug/ai-knowledge-assistant.git
```

Create virtual environment

```bash
python -m venv .venv
```

Activate environment

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Create a `.env` file (copy from `.env.example`)

```env
GROQ_API_KEY=your_groq_api_key_here
QDRANT_URL=https://your-cluster-url.qdrant.io
QDRANT_API_KEY=your_qdrant_api_key_here
```

Ingest data into Qdrant (run once, and again whenever `data/` changes)

```bash
python ingest.py
```

Run the chatbot

```bash
python chatbot.py
```

---

## Evaluation

The project includes a lightweight evaluation framework to measure RAG quality without any manual testing.

```bash
python evaluate.py
```

This runs the 20 test cases in `test_cases.json` through the same `retrieve()` and `ask()` functions used by the chatbot, then reports:

- **Retriever Accuracy** — does the retrieved chunk metadata match the expected source?
- **Context Precision** — ratio of relevant chunks among all retrieved chunks
- **Answer Accuracy** — ratio of expected keywords matched in the chatbot's answer (≥50% threshold)
- **Hallucination Rate** — retriever found the right context but the answer is wrong *and* isn't an honest "I don't know"
- **Average Response Time**

A detailed per-question breakdown is written to `evaluation_report.csv`.

---

## Roadmap

| Task | Start | End | Estimate | Status | Note |
|---|---|---|---|---|---|
| Tìm hiểu Prompt Engineering | 22/07/2026 | 24/07/2026 | 2 ngày | ✅ Hoàn thành | Tìm hiểu Prompt, Prompt Template |
| Tìm hiểu Context Management | 25/07/2026 | 26/07/2026 | 2 ngày | ✅ Hoàn thành | Quản lý hội thoại và Context Window |
| Xây dựng Chatbot bằng Python + LangChain | 27/07/2026 | 29/07/2026 | 3 ngày | ✅ Hoàn thành | Prototype chatbot cơ bản |
| Tìm hiểu RAG | 30/07/2026 | 01/08/2026 | 3 ngày | ✅ Hoàn Thành | Retrieval-Augmented Generation |
| Tìm hiểu Embedding & Vector Database (FAISS/Qdrant) | 03/08/2026 | 04/08/2026 | 2 ngày | ✅ Hoàn Thành | Embedding và lưu trữ vector |
| Tích hợp RAG vào Chatbot | 05/08/2026 | 07/08/2026 | 3 ngày | ✅ Hoàn Thành | Chat với tài liệu PDF/Markdown |
| Tìm hiểu GraphRAG | 08/08/2026 | 10/08/2026 | 2 ngày | ⬜ Chưa bắt đầu | Nghiên cứu kiến trúc và ứng dụng |
| Tìm hiểu MCP (Model Context Protocol) | 11/08/2026 | 12/08/2026 | 2 ngày | ⬜ Chưa bắt đầu | Hiểu cách LLM giao tiếp với công cụ |
| Tìm hiểu Function Calling & Tool Calling | 13/08/2026 | 14/08/2026 | 2 ngày | ⬜ Chưa bắt đầu | Tích hợp Tool Calling |
| Xây dựng AI Agent bằng LangChain | 15/08/2026 | 18/08/2026 | 3 ngày | ⬜ Chưa bắt đầu | Prototype AI Agent |
| Tìm hiểu Spring Boot | 19/08/2026 | 21/08/2026 | 3 ngày | ⬜ Chưa bắt đầu | REST API, Dependency Injection |
| Tìm hiểu Spring AI | 22/08/2026 | 24/08/2026 | 2 ngày | ⬜ Chưa bắt đầu | ChatClient, tích hợp LLM |
| Chuyển Prototype sang Java | 25/08/2026 | 27/08/2026 | 3 ngày | ⬜ Chưa bắt đầu | Spring Boot + Spring AI |
| Hoàn thiện tài liệu và báo cáo | 28/08/2026 | 28/08/2026 | 1 ngày | ⬜ Chưa bắt đầu | Demo và báo cáo tổng kết |

> Lưu ý: bảng trên phản ánh kế hoạch làm việc theo tuần (cập nhật thủ công). Trên thực tế RAG + Vector Database + đánh giá chất lượng đã được triển khai xong ở mức prototype (xem mục Features/Evaluation phía trên) — bảng sẽ được đồng bộ lại vào lần cập nhật kế tiếp.

---

## Demo

Coming soon.

---

## License

This project is developed for educational purposes and AI Engineering internship.
