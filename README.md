# Enterprise AI Knowledge Assistant

An AI-powered knowledge assistant built with **Python**, **LangChain**, and **Large Language Models (LLMs)**. This project is developed during my AI Engineering internship and will evolve into a complete enterprise-ready AI assistant with Retrieval-Augmented Generation (RAG), Tool Calling, MCP, and AI Agent capabilities.

---

## Features

Current features:

- Chat with LLM (Groq)
- Conversation Memory
- System Prompt
- Environment Configuration (.env)

Planned features:

- PDF Upload
- Retrieval-Augmented Generation (RAG)
- Vector Database (Qdrant)
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
- python-dotenv

Future Stack:

- Qdrant
- FastAPI / Spring Boot
- Spring AI
- LiteLLM
- MariaDB

---

## Project Structure

```
ai-knowledge-assistant/
│
├── chatbot.py
├── requirements.txt
├── .env.example
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

Create a `.env` file

```env
GROQ_API_KEY=your_groq_api_key_here
```

Run the chatbot

```bash
python chatbot.py
```

---

## Roadmap

- [x] Prompt Engineering
- [x] Context Management
- [x] Basic Chatbot
- [ ] RAG
- [ ] Embedding
- [ ] Vector Database
- [ ] Citation
- [ ] Multi-document Support
- [ ] Tool Calling
- [ ] MCP
- [ ] AI Agent
- [ ] Spring AI
- [ ] Enterprise Deployment

---

## Demo

Coming soon.

---

## License

This project is developed for educational purposes and AI Engineering internship.