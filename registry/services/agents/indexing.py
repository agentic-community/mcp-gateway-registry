from __future__ import annotations

import logging

from ...schemas.agent_models import (
    AgentCard,
)

logger = logging.getLogger(__name__)


async def _index_agent_in_faiss(
    *,
    agent_card: AgentCard,
    is_enabled: bool,
) -> None:
    try:
        from ...search.service import faiss_service

        agent_data = agent_card.model_dump(mode="json")

        await faiss_service.add_or_update_entity(
            entity_path=agent_card.path,
            entity_info=agent_data,
            entity_type="a2a_agent",
            is_enabled=is_enabled,
        )

        logger.info("Indexed agent '%s' in FAISS", agent_card.name)

    except Exception as exc:
        logger.error("Failed to index agent in FAISS: %s", exc, exc_info=True)

