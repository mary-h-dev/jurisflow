from __future__ import annotations

import logging
from typing import Any

from django.conf import settings
from neo4j import GraphDatabase, Driver, Session

logger = logging.getLogger(__name__)


class Neo4jClient:
    """
    Lazy Neo4j connection for Django apps.
    One instance per app — never instantiate directly, use the singleton below.
    """

    def __init__(self) -> None:
        self._driver: Driver | None = None

    @property
    def driver(self) -> Driver:
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                settings.NEO4J_URI,
                auth=(settings.NEO4J_USERNAME, settings.NEO4J_PASSWORD),
            )
            logger.debug("Neo4j driver created")
        return self._driver

    def session(self) -> Session:
        return self.driver.session()

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None
            logger.debug("Neo4j driver closed")

    def verify(self) -> bool:
        """Health check — returns True if connection is alive."""
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as e:
            logger.error(f"Neo4j connectivity check failed: {e}")
            return False


neo4j_client = Neo4jClient()