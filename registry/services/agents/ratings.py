from __future__ import annotations

import logging
from typing import (
    Any,
    Dict,
)

from ...schemas.agent_models import (
    AgentCard,
)

logger = logging.getLogger(__name__)

_MAX_RATINGS: int = 100


def _apply_rating_update(
    *,
    existing_agent: AgentCard,
    username: str,
    rating: int,
) -> Dict[str, Any]:
    """Return an updated agent dict with rating changes applied."""
    if not isinstance(rating, int):
        logger.error("Invalid rating type: %s (type=%s)", rating, type(rating))
        raise ValueError("Rating must be an integer")
    if rating < 1 or rating > 5:
        logger.error("Invalid rating value: %s. Must be between 1 and 5.", rating)
        raise ValueError("Rating must be between 1 and 5 (inclusive)")

    agent_dict = existing_agent.model_dump()

    if "rating_details" not in agent_dict or agent_dict["rating_details"] is None:
        agent_dict["rating_details"] = []

    user_found = False
    for entry in agent_dict["rating_details"]:
        if entry.get("user") == username:
            entry["rating"] = rating
            user_found = True
            break

    if not user_found:
        agent_dict["rating_details"].append(
            {
                "user": username,
                "rating": rating,
            }
        )

        if len(agent_dict["rating_details"]) > _MAX_RATINGS:
            agent_dict["rating_details"].pop(0)
            logger.info(
                "Removed oldest rating to maintain %s entries limit for agent at '%s'",
                _MAX_RATINGS,
                agent_dict.get("path"),
            )

    all_ratings = [entry["rating"] for entry in agent_dict["rating_details"]]
    agent_dict["num_stars"] = float(sum(all_ratings) / len(all_ratings))
    return agent_dict

