import os
from tqdm import tqdm
from aser.relation import Relation
from aser.concept.concept_extractor import ASERConceptExtractor
from aser.concept.concept_connection import ASERConceptConnection
from aser.database.kg_connection import ASERKGConnection
from copy import copy


def build_concept_instance_table_from_aser_kg(aser_concept_conn, aser_concept_extractor, aser_kg_conn):
    for i, eid in tqdm(enumerate(aser_kg_conn.eids)):
        event = aser_kg_conn.get_exact_match_event(eid)
        results = aser_concept_extractor.conceptualize(event)
        concepts = [x[0] for x in results]
        scores = [x[1] for x in results]
        aser_concept_conn.insert_concept(concepts, [event]*len(results), scores)


def build_concept_relation_table_from_aser_kg(aser_concept_conn, aser_concept_extractor, aser_kg_conn):
    relation_mapping = dict()
    for cid in tqdm(aser_concept_conn.cids):
        instances = aser_concept_conn.get_eventualities_given_concept(cid)
        for eid, _, instance_score in instances:
            results = aser_kg_conn.get_related_eventualities(eid)
            t_events = aser_kg_conn.get_exact_match_events([x[0] for x in results])
            relations = [x[1] for x in results]

            for t_event, relation in zip(t_events, relations):
                t_concepts = aser_concept_conn.get_concepts_given_eventuality(t_event)
                if len(t_concepts) == 0:
                    for idx, (t_concept, prob) in enumerate(aser_concept_extractor.conceptualize(t_event)):
                        tmp = aser_concept_conn.get_exact_match_concept(t_concept.cid)
                        if tmp:
                            rid = Relation.generate_rid(cid, tmp.cid)
                            r = copy(relation.relations)
                            for k in r:
                                r[k] = r[k] * instance_score * prob

                            if rid not in relation_mapping:
                                relation_mapping[rid] = Relation(cid, tmp.cid, relations=r)
                            else:
                                relation_mapping[rid].update_relations(r)
    aser_concept_conn.insert_relations(relation_mapping.values())


if __name__ == "__main__":
    aser_kg_dir = "/data/hjpan/ASER/nyt_test_filtered"
    print(aser_kg_dir)
    print("Loading ASER db...")
    aser_kg_conn = ASERKGConnection(os.path.join(aser_kg_dir, "KG.db"), mode="memory")
    print("Loading ASER db done..")

    aser_concept_extractor = ASERConceptExtractor(
        source="probase",
        probase_path="/data/hjpan/probase/data-concept-instance-relations-yq.txt",
        probase_topk=3)
    aser_concept_conn = ASERConceptConnection(
        db_path=os.path.join(aser_kg_dir, "concept.db"),
        overwrite=True)

    print("Building Concepts")
    build_concept_instance_table_from_aser_kg(
        aser_concept_conn, aser_concept_extractor, aser_kg_conn)
    print("[Statistics] Overall unique eventualities: %d" % len(aser_concept_conn.instance_to_concept))
    print("[Statistics] Overall unique concepts: %d" % len(aser_concept_conn.concept_to_instances))

    print("Building Concepts relations")
    build_concept_relation_table_from_aser_kg(
        aser_concept_conn, aser_concept_extractor, aser_kg_conn)
    print("[Statistics] Overall concept-by-concept relations: %d" % len(aser_concept_conn.concept_relations))
    aser_concept_conn.close()
    aser_kg_conn.close()