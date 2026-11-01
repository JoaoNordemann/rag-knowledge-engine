import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import vector_database_chromadb as v


class FakeCollection:
    def query(self, query_texts, n_results, where=None):
        return {"ids": [["a", "b"]], "documents": [["doc a", "doc b"]],
                "metadatas": [[{"dominio": "x"}, {"dominio": "y"}]],
                "distances": [[0.1, 0.6]]}


def test_search_converts_distance_to_similarity():
    hits = v.search(FakeCollection(), "q", k=2)
    assert [h["id"] for h in hits] == ["a", "b"]
    assert abs(hits[0]["similarity"] - 0.9) < 1e-9


def test_document_ids_are_unique():
    ids = [d["id"] for d in v.DOCUMENTOS]
    assert len(ids) == len(set(ids))
