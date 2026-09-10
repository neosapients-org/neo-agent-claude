"""Runtime patches for mem0 integrations."""

from __future__ import annotations

import logging


logger = logging.getLogger(__name__)


def patch_mem0_milvus_update() -> bool:
    """
    Patch Mem0's MilvusDB.update to avoid None vector upserts.

    Mem0 sometimes calls update(vector=None, payload=...) when handling NONE actions.
    Milvus rejects None vectors, so we fetch the existing vector and reuse it.
    """
    try:
        from mem0.vector_stores.milvus import MilvusDB
    except Exception:
        return False

    if getattr(MilvusDB, "_neo_memory_hub_patch_update", False):
        return True

    original_update = MilvusDB.update

    def update(self, vector_id=None, vector=None, payload=None):  # type: ignore[override]
        if vector is None and vector_id:
            try:
                result = self.client.query(
                    collection_name=self.collection_name,
                    filter=f'id == \"{vector_id}\"',
                    output_fields=["vectors", "metadata"],
                    limit=1,
                )
            except Exception as exc:
                logger.warning(
                    "Milvus update fallback failed; using original update. Error: %s",
                    exc,
                )
                return original_update(self, vector_id=vector_id, vector=vector, payload=payload)

            if result:
                row = result[0]
                vector = row.get("vectors")
                if payload is None:
                    payload = row.get("metadata")

        if vector is None:
            logger.warning("Skipping Milvus update without vector for id %s", vector_id)
            return None

        schema = {"id": vector_id, "vectors": vector, "metadata": payload or {}}
        return self.client.upsert(collection_name=self.collection_name, data=schema)

    MilvusDB.update = update  # type: ignore[assignment]
    MilvusDB._neo_memory_hub_patch_update = True
    return True


def patch_mem0_qdrant_update() -> bool:
    """
    Patch Mem0's Qdrant.update to avoid PointStruct(vector=None) validation errors.

    Mem0 calls ``update(vector=None, payload=...)`` for NONE actions (dedup) when
    it only needs to update payload metadata (e.g., session IDs).  Qdrant's
    ``PointStruct`` model rejects ``vector=None`` with a Pydantic validation error::

        6 validation errors for PointStruct
        vector.list[float]
          Input should be a valid list ...

    The fix: when ``vector is None``, use ``client.set_payload`` to update only the
    payload without touching the vector, which is both correct and more efficient.
    """
    try:
        from mem0.vector_stores.qdrant import Qdrant  # type: ignore[import-untyped]
    except Exception:
        return False

    if getattr(Qdrant, "_neo_memory_hub_patch_update", False):
        return True

    original_update = Qdrant.update

    def update(self, vector_id=None, vector=None, payload=None):  # type: ignore[override]
        if vector is None:
            if payload is not None and vector_id is not None:
                # Payload-only update — no need to touch the vector at all.
                from qdrant_client.models import PointIdsList  # noqa: local import

                self.client.set_payload(
                    collection_name=self.collection_name,
                    payload=payload,
                    points=[vector_id],
                )
                logger.debug(
                    "Qdrant payload-only update for id %s (vector=None skipped)",
                    vector_id,
                )
                return None
            # Nothing useful to do — skip silently.
            logger.debug("Skipping Qdrant update without vector or payload for id %s", vector_id)
            return None

        return original_update(self, vector_id=vector_id, vector=vector, payload=payload)

    Qdrant.update = update  # type: ignore[assignment]
    Qdrant._neo_memory_hub_patch_update = True
    return True
