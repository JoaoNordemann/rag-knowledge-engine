# RAG Knowledge Engine: Assistente Especialista de Documentos

Este repositorio contem a implementacao de um sistema de Retrieval-Augmented Generation (RAG) projetado para indexar textos complexos e fornecer respostas precisas utilizando Modelos de Linguagem de Grande Escala (LLMs) locais. O objetivo principal da arquitetura e garantir respostas baseadas exclusivamente nos documentos fornecidos, reduzindo a zero o risco de alucinacao.

## Arquitetura e Tecnologias

* Banco de Dados Vetorial: ChromaDB (Local/In-Memory)
* Modelo de Embeddings: Sentence-Transformers (all-MiniLM-L6-v2)
* LLM de Inferencia: Mistral-7B-Instruct-v0.2 (Quantizado em 4-bits com bitsandbytes) - [Em desenvolvimento]
* API/Backend: FastAPI - [Planejado]

## Roadmap de Implementacao

O projeto esta estruturado em modulos para demonstrar a construcao de ponta a ponta do pipeline de dados e inferencia:

- [x] Fase 1: Bancos Vetoriais e Embeddings (Script `1_vector_database_chromadb.ipynb`). Indexacao de textos e busca semantica com ChromaDB.
- [ ] Fase 2: Pipeline RAG. Integracao do banco de dados vetorial com o LLM para leitura do contexto e geracao da resposta em linguagem natural.
- [ ] Fase 3: Disponibilizacao (API). Empacotamento do motor RAG em uma API (FastAPI) para simular um ambiente de producao.

## Como Executar (Fase 1 - Busca Semantica)

Para rodar o modulo de indexacao e busca no banco de dados vetorial:

1. Instale as dependencias base:
   ```bash
   pip install chromadb sentence-transformers
