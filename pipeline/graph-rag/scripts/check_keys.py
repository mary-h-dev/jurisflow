
import os
from dotenv import load_dotenv
from database.connection import Neo4jConnection

load_dotenv()

NEO4J_URI  = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

def check_rel_keys():
    query = """
    MATCH (r:Ruling)-[rel]->(f)
    WHERE type(rel) IN ['HAS_CONCEPT','HAS_ROLE','HAS_ACTION','HAS_OBJECT','HAS_FACT']
    RETURN type(rel) AS rel_type, keys(rel) AS properties
    LIMIT 10
    """
    connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    try:
        with connection.driver.session() as session:
            result = session.run(query)
            for record in result:
                print(f"Rel Type: {record['rel_type']} -> Keys: {record['properties']}")
    finally:
        connection.close()

if __name__ == "__main__":
    check_rel_keys()




# import os
# from dotenv import load_dotenv
# from database.connection import Neo4jConnection

# load_dotenv()

# NEO4J_URI  = os.getenv("NEO4J_URI")
# NEO4J_USER = os.getenv("NEO4J_USERNAME")
# NEO4J_PASS = os.getenv("NEO4J_PASSWORD")

# def check_feature_keys():
#     query = """
#     MATCH (f) 
#     WHERE any(lbl IN labels(f) WHERE lbl IN ['LegalConcept', 'LegalRole', 'LegalAction', 'LegalObject', 'LegalFact'])
#     RETURN labels(f) AS node_types, keys(f) AS properties 
#     LIMIT 10
#     """
#     connection = Neo4jConnection(NEO4J_URI, NEO4J_USER, NEO4J_PASS)
    
#     try:
#         # اگر کلاس Neo4jConnection شما متد execute یا driver دارد
#         with connection.driver.session() as session:
#             result = session.run(query)
#             for record in result:
#                 print(f"Node Labels: {record['node_types']} -> Keys: {record['properties']}")
#     finally:
#         connection.close()

# if __name__ == "__main__":
#     check_feature_keys()