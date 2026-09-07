from neo4j import GraphDatabase


class Neo4jConnection:
    def __init__(self, uri: str, username: str, password: str):
        self.driver = GraphDatabase.driver(uri, auth=(username, password))

    def close(self):
        self.driver.close()

    def session(self):
        return self.driver.session()

    def clear_database(self):
        """Wipe the entire database — development use only."""
        with self.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        print("🗑️ Database cleared.")