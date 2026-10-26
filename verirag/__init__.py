from .config import Settings

__all__ = ["Settings", "VeriRAG"]


def __getattr__(name):
    # Lazy so that importing lightweight modules doesn't pull in torch/faiss.
    if name == "VeriRAG":
        from .pipeline import VeriRAG

        return VeriRAG
    raise AttributeError(name)
