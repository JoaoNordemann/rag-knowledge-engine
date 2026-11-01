"""Banco vetorial persistente (ChromaDB) com embeddings multilíngues e métrica cosseno."""
from __future__ import annotations

import torch
import chromadb
from chromadb.utils import embedding_functions

# Multilíngue (inclui pt-BR), 384 dimensões, leve. Alternativas: intfloat/multilingual-e5-small
# (exige prefixos "query: " / "passage: ") ou BAAI/bge-m3 (mais pesado e mais preciso).
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"
PERSIST_DIR = "./chroma_db"
COLLECTION_NAME = "base_conhecimento"


def get_collection(persist_dir: str = PERSIST_DIR, name: str = COLLECTION_NAME):
    client = chromadb.PersistentClient(path=persist_dir)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    # A métrica é fixada na criação da coleção; mudar depois exige recriar a coleção.
    return client.get_or_create_collection(
        name=name,
        embedding_function=emb_fn,
        metadata={"hnsw:space": "cosine"},
    )


def index_documents(collection, docs: list[dict]) -> None:
    """docs: [{'id': str, 'text': str, 'metadata': dict}]. upsert = idempotente."""
    collection.upsert(
        ids=[d["id"] for d in docs],
        documents=[d["text"] for d in docs],
        metadatas=[d.get("metadata") or None for d in docs],
    )


def search(collection, query: str, k: int = 2, where: dict | None = None) -> list[dict]:
    res = collection.query(query_texts=[query], n_results=k, where=where)
    return [
        {
            "id": res["ids"][0][i],
            "text": res["documents"][0][i],
            "metadata": res["metadatas"][0][i],
            "distance": res["distances"][0][i],
            "similarity": 1 - res["distances"][0][i],  # cosseno: distância = 1 - similaridade
        }
        for i in range(len(res["ids"][0]))
    ]


DOCUMENTOS = [
    {"id": "doc_rpg_combate", "metadata": {"dominio": "rpg"},
     "text": "Ataque Corpo a Corpo: Para realizar um ataque com espada ou machado, role 1d20 e some seu bônus de Força e perícia Luta contra a Defesa do alvo."},
    {"id": "doc_rpg_terror", "metadata": {"dominio": "rpg"},
     "text": "Mecânica de Sanidade: Ao presenciar uma criatura do Outro Lado, o investigador deve fazer um teste de Vontade. Falhar resulta na perda de pontos de Sanidade baseados na Presença Aterradora do monstro."},
    {"id": "doc_eletronica_and", "metadata": {"dominio": "eletronica"},
     "text": "Portas Lógicas: A porta AND (E) é um circuito digital que retorna saída verdadeira (1) única e exclusivamente se todas as suas entradas forem verdadeiras (1)."},
    {"id": "doc_culinaria_massa", "metadata": {"dominio": "culinaria"},
     "text": "Preparo de Focaccia: A massa de alta hidratação (90%) precisa descansar em dobras a cada 30 minutos nas primeiras duas horas para desenvolver a rede de glúten antes de assar."},
    # Domínio delivery (alinhado ao ecossistema iFood)
    {"id": "doc_delivery_atraso", "metadata": {"dominio": "delivery"},
     "text": "Política de Atraso: Se o pedido exceder o tempo estimado em mais de 30 minutos, o cliente pode solicitar cancelamento sem custo e receber reembolso integral em até 48 horas."},
    {"id": "doc_delivery_cupom", "metadata": {"dominio": "delivery"},
     "text": "Cupons: Cada cupom é válido para um único pedido, não é cumulativo com outras promoções e expira na data indicada, mesmo que não tenha sido utilizado."},
    {"id": "doc_delivery_entregador", "metadata": {"dominio": "delivery"},
     "text": "Roteirização: O entregador recebe a rota otimizada considerando distância, trânsito em tempo real e o tempo de preparo do restaurante para minimizar a espera na retirada."},
]


def main() -> None:
    colecao = get_collection()
    index_documents(colecao, DOCUMENTOS)
    print(f"Coleção pronta | {colecao.count()} documentos | modelo: {EMBEDDING_MODEL}")

    for pergunta in [
        "Como faço para não enlouquecer quando vejo um monstro?",
        "Meu pedido demorou demais, posso pedir meu dinheiro de volta?",
    ]:
        print(f"\nPergunta: {pergunta}")
        for i, r in enumerate(search(colecao, pergunta, k=2), 1):
            print(f"{i}º (sim: {r['similarity']:.3f} | dist: {r['distance']:.3f}) "
                  f"[{r['metadata'].get('dominio')}] {r['text']}")


if __name__ == "__main__":
    main()
