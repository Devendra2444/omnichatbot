# OmniRAG Multi-File Assistant

A RAG chatbot that answers questions across multiple file types — PDF, TXT, CSV, DOCX — using LangChain, Chroma, and Groq.

## Features
- Loads all documents from the `data/` folder automatically
- Supports PDF, text, CSV, and Word files
- Uses Chroma vector DB with persistent caching (fast reload)
- Chainlit chat UI with streaming responses

## Usage
```bash
chainlit run MultiAgentChatBot.py
```

## Setup
1. Add your `GROQ_API_KEY` to `.env`
2. Place files in the `data/` folder
3. Run the app and ask questions in plain English
