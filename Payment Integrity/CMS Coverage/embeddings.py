import json
from pathlib import Path
import numpy as np
from access_token import get_access_token_sync

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

try:
    import hnswlib  # type: ignore
    _HAS_HNSW = True
except Exception:
    _HAS_HNSW = False

from langchain_openai import AzureOpenAIEmbeddings

def _l2_normalize(X: np.ndarray) -> np.ndarray:
    X = X.astype("float32", copy=False)
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / norms

class VectorStore:
    """
    Backend order of preference: FAISS -> HNSWLIB -> NumPy (brute-force).
    All backends expose the same add/save/load/search interface.
    """
    def __init__(
        self,
        azure_endpoint: str,
        api_key: str,
        azure_deployment: str,
        api_version: str,
        access_token=None,
        index_path: Path = None,
        backend: str = None
    ):
        if access_token is None:
            access_token = get_access_token_sync()
        self.embeddings = AzureOpenAIEmbeddings(
            azure_deployment=azure_deployment,
            model="text-embedding-3-small",
            api_version=api_version,
            azure_endpoint=azure_endpoint,
            openai_api_type="azure_ad",
            validate_base_url=False,
            azure_ad_token=access_token,
            default_headers={
                "projectId": "5a82e295-85f9-4458-835e-4db272326dca"
            },
            chunk_size=5,
            max_retries=15,
            retry_min_seconds=50,
            show_progress_bar=True,
            skip_empty=True,
        )
        self.dim = 1536  # Default for text-embedding-ada-002
        self.index_path = index_path
        self.ids: list[str] = []
        self.backend = backend or ("faiss" if _HAS_FAISS else ("hnsw" if _HAS_HNSW else "numpy"))
        if self.backend == "faiss":
            self.index = faiss.IndexFlatIP(self.dim)
        elif self.backend == "hnsw":
            self.index = hnswlib.Index(space="cosine", dim=self.dim)
            self._hnsw_initialized = False
        else:
            self.index = None
            self._matrix = None

    def _encode_norm(self, texts: list[str]) -> np.ndarray:
        emb = self.embeddings.embed_documents(texts)
        return _l2_normalize(np.array(emb))

    def add(self, id_text_pairs: list[tuple[str, str]]):
        texts = [t for _, t in id_text_pairs]
        ids = [i for i, _ in id_text_pairs]
        emb = self._encode_norm(texts)
        if self.backend == "faiss":
            self.index.add(emb)
        elif self.backend == "hnsw":
            if not getattr(self, "_hnsw_initialized", False):
                self.index.init_index(max_elements=len(ids), ef_construction=200, M=32)
                self._hnsw_initialized = True
            self.index.add_items(emb, np.arange(len(self.ids), len(self.ids) + len(ids)))
        else:
            self._matrix = emb if self._matrix is None else np.vstack([self._matrix, emb])
        self.ids.extend(ids)

    def save(self):
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        meta = {"ids": self.ids, "dim": self.dim, "backend": self.backend, "model_dim": self.dim}
        (self.index_path.parent / "faiss.meta.json").write_text(json.dumps(meta))
        if self.backend == "faiss":
            faiss.write_index(self.index, str(self.index_path))
        elif self.backend == "hnsw":
            self.index.save_index(str(self.index_path))
        else:
            np.save(str(self.index_path) + ".npy", self._matrix)

    @classmethod
    def load(cls, azure_endpoint: str, api_key: str, azure_deployment: str, api_version: str, index_path: Path):
        meta = json.loads((index_path.parent / "faiss.meta.json").read_text())
        backend = meta.get("backend", "faiss")
        inst = cls(azure_endpoint, api_key, azure_deployment, api_version, index_path=index_path, backend=backend)
        inst.ids = meta["ids"]
        if backend == "faiss":
            inst.index = faiss.read_index(str(index_path))
        elif backend == "hnsw":
            inst.index = hnswlib.Index(space="cosine", dim=inst.dim)
            inst.index.load_index(str(index_path))
            inst._hnsw_initialized = True
        else:
            inst._matrix = np.load(str(index_path) + ".npy")
        return inst

    def search(self, query_text: str, top_k: int = 20):
        q = self._encode_norm([query_text]).astype("float32")
        if self.backend == "faiss":
            D, I = self.index.search(q, top_k)
            idxs = I[0]
            sims = D[0]
        elif self.backend == "hnsw":
            labels, dists = self.index.knn_query(q, k=top_k)
            idxs = labels[0]
            sims = 1.0 - dists[0]
        else:
            sims = (self._matrix @ q[0])
            idxs = np.argpartition(-sims, range(min(top_k, sims.shape[0])))[:top_k]
            idxs = idxs[np.argsort(-sims[idxs])]
        results = []
        for rank_pos, idx in enumerate(idxs):
            if idx == -1 or idx >= len(self.ids):
                continue
            results.append((self.ids[int(idx)], float(sims[rank_pos])))
        return results