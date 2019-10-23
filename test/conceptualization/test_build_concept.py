import os
from aser.concept.conceptualize import  ASERConceptDB
from aser.database.kg_connection import KGConnection

if __name__ == "__main__":
    aser_kg_dir = "/data/hjpan/ASER/nyt_test_filtered"
    print("Loading ASER db...")
    aser_kg_conn = KGConnection(os.path.join(aser_kg_dir, "KG.db"), mode="memory")
    print("Loading ASER db done..")
    aser_concept_db = ASERConceptDB(
        source="probase",
        probase_path="/data/hjpan/probase/data-concept-instance-relations-yq.txt",
        probase_topk=3)
    aser_concept_db.build_concept_db_from_aser_kg(aser_kg_conn)
    aser_concept_db.save(os.path.join(aser_kg_dir, "concept.pkl"))
    aser_kg_conn.close()
    print()