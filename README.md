# Vector Database com ChromaDB (RAG retrieval em pt-BR)

Camada de retrieval semântico com ChromaDB persistente, embeddings multilíngues e métrica cosseno.

## Decisões técnicas
| Tema | Decisão | Motivo |
|---|---|---|
| Embedding | `paraphrase-multilingual-MiniLM-L12-v2` (384d) | `all-MiniLM-L6-v2` é centrado em inglês e recupera mal em pt-BR |
| Armazenamento | `PersistentClient("./chroma_db")` | Índice sobrevive entre execuções |
| Métrica | `hnsw:space = cosine` | Padrão do Chroma é L2; cosseno é o usual para embeddings de texto |
| Escrita | `upsert` | Idempotente |
| Similaridade | `1 - distância` | Valor mais legível para threshold |

> A métrica é definida na criação da coleção. Se já existir `chroma_db/` com L2, apague a pasta.

## Uso
```bash
pip install -r requirements.txt
python vector_database_chromadb.py
```
Como biblioteca:
```python
from vector_database_chromadb import get_collection, index_documents, search
col = get_collection()
index_documents(col, [{"id": "d1", "text": "...", "metadata": {"dominio": "delivery"}}])
search(col, "meu pedido atrasou", k=3, where={"dominio": "delivery"})
```

## Testes
```bash
pytest -q
```

## Roadmap
- [ ] Golden set de perguntas pt-BR + métricas hit@k e MRR
- [ ] Comparar MiniLM-multilingual vs multilingual-e5-small vs bge-m3
- [ ] Chunking com overlap e dataset de delivery publicado no Hugging Face
- [ ] Reranking com cross-encoder
