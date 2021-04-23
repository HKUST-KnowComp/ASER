import os
import sys
import time
from tqdm import tqdm
from aser.relation import Relation
from aser.conceptualize.aser_conceptualizer import ASERSeedConceptualizer, ASERProbaseConceptualizer
from aser.database.kg_connection import ASERKGConnection, ASERConceptConnection
from copy import copy


def build_concept_instance_table_from_aser_kg(aser_concept_conn, aser_conceptualizer, aser_kg_conn):
    cid2concept = dict()
    concept_instance_pairs = []
    cid_to_filter_score = dict()
    for eid in tqdm(aser_kg_conn.eids):
        eventuality = aser_kg_conn.get_exact_match_eventuality(eid)
        results = aser_conceptualizer.conceptualize(eventuality)
        for concept, score in results:
            if concept.cid not in cid2concept:
                cid2concept[concept.cid] = copy(concept)
            concept = cid2concept[concept.cid]
            if (eventuality.eid, eventuality.pattern, score) not in concept.instances:
                concept.instances.append(((eventuality.eid, eventuality.pattern, score)))
                if concept.cid not in cid_to_filter_score:
                    cid_to_filter_score[concept.cid] = 0.0
                cid_to_filter_score[concept.cid] += score * eventuality.frequency
            concept_instance_pairs.append((concept, eventuality, score))
    with open(os.path.join(aser_concept_dir, "concept_cids.txt"), "w") as f:
        for cid, filter_score in cid_to_filter_score.items():
            f.write(cid + "\t" + "{:.2f}".format(filter_score) + "\n")
    aser_concept_conn.insert_concepts(list(cid2concept.values()))
    aser_concept_conn.insert_concept_instance_pairs(concept_instance_pairs)


def build_concept_relation_table_from_aser_kg(aser_concept_conn, aser_kg_conn):
    rid2relation = dict()
    for h_cid in tqdm(aser_concept_conn.cids):
        instances = aser_concept_conn.get_eventualities_given_concept(h_cid)
        for h_eid, pattern, instance_score in instances:
            # eid -> event -> related eids -> related eventualities, relations -> related concepts, relations
            related_eventualities = aser_kg_conn.get_related_eventualities(h_eid)
            for t_eid, relation in related_eventualities:
                concept_score_pairs = aser_concept_conn.get_concepts_given_eventuality(t_eid)
                for t_concept, score in concept_score_pairs:
                    t_cid = t_concept.cid
                    if h_cid == t_cid:
                        continue
                    rid = Relation.generate_rid(h_cid, t_cid)
                    if rid not in rid2relation:
                        rid2relation[rid] = Relation(h_cid, t_cid)
                    rid2relation[rid].update(
                        {k: v * instance_score * score for k, v in relation.relations.items()})

    rid_to_filter_score = {k: sum(v.relations.values()) for k, v in rid2relation.items()}
    with open(os.path.join(aser_concept_dir, "concept_rids.txt"), "w") as f:
        for rid, filter_score in rid_to_filter_score.items():
            f.write(rid + "\t" + "{:.2f}".format(filter_score) + "\n")
    aser_concept_conn.insert_relations(rid2relation.values())


if __name__ == "__main__":
    st = time.time()
    aser_kg_dir = sys.argv[1]
    aser_concept_dir = sys.argv[2]
    probase_path = sys.argv[3]
    print("Loading ASER db from {} ...".format(aser_kg_dir))
    aser_kg_conn = ASERKGConnection(os.path.join(aser_kg_dir, "KG.db"), mode="memory")
    print("Loading ASER db done..")

    aser_conceptualizer = ASERProbaseConceptualizer(probase_path=probase_path, probase_topk=5)
    aser_concept_conn = ASERConceptConnection(db_path=os.path.join(aser_concept_dir, "conceptualize.db"), mode="memory")

    print("Building Concepts")
    build_concept_instance_table_from_aser_kg(
        aser_concept_conn, aser_conceptualizer, aser_kg_conn)
    print("[Statistics] Overall unique eventualities: %d" % len(aser_kg_conn.eids))
    print("[Statistics] Overall unique concepts: %d" % len(aser_concept_conn.cids))

    print("Building Concepts relations")
    build_concept_relation_table_from_aser_kg(
        aser_concept_conn, aser_kg_conn)
    print("[Statistics] Overall event-by-event relations: %d" % len(aser_kg_conn.rids))
    print("[Statistics] Overall conceptualize-by-conceptualize relations: %d" % len(aser_concept_conn.rids))
    aser_concept_conn.close()
    aser_kg_conn.close()
    print("Building Concept db finished in {:.2f}s".format(time.time() - st))