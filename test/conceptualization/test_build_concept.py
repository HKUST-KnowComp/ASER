import os
from tqdm import tqdm
from aser.concept.conceptualize import  ASERConceptExtractor, ASERConceptDB
from aser.database.kg_connection import KGConnection


def build_concept_instance_table_from_aser_kg(aser_concept_db, aser_concept_extractor, aser_kg_conn):
    for i, eid in tqdm(enumerate(aser_kg_conn.eids)):
        event = aser_kg_conn.get_exact_match_event(eid)
        concepts = aser_concept_extractor.conceptualize(event)
        for concept, score in concepts:
            aser_concept_db.insert_instance(event, concept, score)


def build_concept_relation_table_from_aser_kg(aser_concept_db, aser_concept_extractor, aser_kg_conn):
    for h_concept_id in tqdm(aser_concept_db.id2concepts):
        instances = aser_concept_db.concept_to_instances[h_concept_id]
        for h_eid, _, instance_score in instances:
            relations = aser_kg_conn.get_relations_by_keys(
                bys=["hid"], keys=[h_eid])
            for rel in relations:
                t_event = aser_kg_conn.get_exact_match_event(rel.tid)
                # First check if this event is in db
                t_concepts = aser_concept_db.get_concepts_of_eventuality(t_event)
                # If this event not in db, then extract the concept and check whether
                # the extracted concept is in db
                if not t_concepts:
                    tmp_t_concepts = aser_concept_extractor.conceptualize(t_event)
                    t_concepts = list()
                    for i, (t_concept, prob) in enumerate(tmp_t_concepts):
                        tmp = aser_concept_db.get_exact_match_concept(t_concept.cid)
                        if tmp:
                            t_concepts.append((tmp, prob))
                        else:
                            t_concepts.append((t_concept, prob))
                for rel_sense, count in rel.relations.items():
                    for t_concept, prob in t_concepts:
                        aser_concept_db.insert_relation(
                            h_concept_id, t_concept.cid, rel_sense, count * prob * instance_score)
    aser_concept_db.build_concept_to_related_concepts()


if __name__ == "__main__":
    aser_kg_dir = "/data/hjpan/ASER/nyt_test_filtered"
    print(aser_kg_dir)
    print("Loading ASER db...")
    aser_kg_conn = KGConnection(os.path.join(aser_kg_dir, "KG.db"), mode="memory")
    print("Loading ASER db done..")

    aser_concept_extractor = ASERConceptExtractor(
        source="probase",
        probase_path="/data/hjpan/probase/data-concept-instance-relations-yq.txt",
        probase_topk=3)
    aser_concept_db = ASERConceptDB(
        db_path=os.path.join(aser_kg_dir, "concept.db"),
        overwrite=True)

    print("Building Concepts")
    build_concept_instance_table_from_aser_kg(
        aser_concept_db, aser_concept_extractor, aser_kg_conn)
    print("[Statistics] Overall unique eventualities: %d" % len(aser_concept_db.instance_to_concept))
    print("[Statistics] Overall unique concepts: %d" % len(aser_concept_db.concept_to_instances))

    print("Building Concepts relations")
    build_concept_relation_table_from_aser_kg(
        aser_concept_db, aser_concept_extractor, aser_kg_conn)
    print("[Statistics] Overall concept-by-concept relations: %d" % len(aser_concept_db.concept_relations))
    aser_concept_db.save()
    aser_kg_conn.close()