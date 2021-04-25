import argparse
import copy
import gc
import os
import time
from collections import defaultdict, Counter
from tqdm import tqdm
from aser.database.db_connection import SqliteDBConnection, MongoDBConnection
from aser.database.kg_connection import ASERConceptConnection
from aser.database.kg_connection import CHUNKSIZE
from aser.database.kg_connection import EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS, EVENTUALITY_COLUMN_TYPES
from aser.database.kg_connection import RELATION_TABLE_NAME, RELATION_COLUMNS, RELATION_COLUMN_TYPES
from aser.conceptualize.aser_conceptualizer import ProbaseASERConceptualizer
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.utils.logging import init_logger, close_logger


def convert_row_to_eventuality(row):
    eventuality = Eventuality().decode(row["info"])
    eventuality.eid = row["_id"]
    eventuality.frequency = row["frequency"]
    eventuality.pattern = row["pattern"]
    return eventuality


def convert_row_to_relation(row):
    return Relation(row["hid"], row["tid"], {r: cnt for r, cnt in row.items() if isinstance(cnt, float) and cnt > 0.0})


def build_concept_instance_table(aser_conceptualizer, erows):
    cid2concept = dict()
    concept_instance_pairs = []
    cid_to_filter_score = dict()
    for erow in tqdm(erows):
        event = convert_row_to_eventuality(erow)
        results = aser_conceptualizer.conceptualize(event)
        for concept, score in results:
            if concept.cid not in cid2concept:
                cid2concept[concept.cid] = copy.copy(concept)
            concept = cid2concept[concept.cid]
            if (event.eid, event.pattern, score) not in concept.instances:
                concept.instances.append(((event.eid, event.pattern, score)))
                if concept.cid not in cid_to_filter_score:
                    cid_to_filter_score[concept.cid] = 0.0
                cid_to_filter_score[concept.cid] += score * event.frequency
            concept_instance_pairs.append((concept, event, score))
    return cid2concept, concept_instance_pairs, cid_to_filter_score


def build_concept_relation_table(aser_concept_conn, rrows):
    rid2relation = dict()
    hid2related_events = defaultdict(list)
    for rrow in rrows:
        relation = convert_row_to_relation(rrow)
        hid2related_events[rrow["hid"]].append((rrow["tid"], relation))

    for h_cid in tqdm(aser_concept_conn.cids):
        instances = aser_concept_conn.get_eventualities_given_concept(h_cid)
        for h_eid, pattern, instance_score in instances:
            # eid -> event -> related eids -> related events, relations -> related concepts, relations
            related_events = hid2related_events[h_eid]
            for t_eid, relation in related_events:
                concept_score_pairs = aser_concept_conn.get_concepts_given_eventuality(t_eid)
                for t_concept, score in concept_score_pairs:
                    t_cid = t_concept.cid
                    if h_cid == t_cid:
                        continue
                    rid = Relation.generate_rid(h_cid, t_cid)
                    if rid not in rid2relation:
                        rid2relation[rid] = Relation(h_cid, t_cid)
                    rid2relation[rid].update({k: v * instance_score * score for k, v in relation.relations.items()})
    return rid2relation

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-db", type=str, default="sqlite", choices=["sqlite"])
    parser.add_argument("-kg_path", type=str)
    parser.add_argument("-filtered_kg_dir", type=str)
    parser.add_argument("-log_path", type=str, default="filter_kg.log")

    args = parser.parse_args()

    logger = init_logger(log_file=args.log_path)

    if not os.path.exists(args.filtered_kg_dir):
        os.mkdir(args.filtered_kg_dir)

    if args.db == "sqlite":
        kg_conn = SqliteDBConnection(args.kg_path, CHUNKSIZE)
    else:
        raise NotImplementedError

    erows = []
    rrows = []

    efreqs = dict()
    for erow in kg_conn.get_columns(EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS):
        efreqs[erow["_id"]] = erow["frequency"]
        erows.append(erow)
    logger.info("%d unique eventualities" % (len(erows)))

    rfreqs = dict()
    for rrow in kg_conn.get_columns(RELATION_TABLE_NAME, RELATION_COLUMNS):
        rfreqs[rrow["_id"]] = sum([rrow.get(r, 0.0) for r in relation_senses])
        rrows.append(rrow)
    logger.info("%d unique relations" % (len(rrows)))

    kg_conn.close()

    aser_conceptualizer = ProbaseASERConceptualizer(
        probase_path="/home/xliucr/probase/data-concept-instance-relations-demo.txt", probase_topk=5
    )

    for threadshold in [50, 30, 20, 10, 5, 3]:
        st = time.time()
        logger.info("threadshold", threadshold)
        new_erows = list(filter(lambda erow: erow["frequency"] >= threadshold, erows))
        new_eids = set([erow["_id"] for erow in new_erows])
        new_rrows = list(filter(lambda rrow: rrow["hid"] in new_eids and rrow["tid"] in new_eids, rrows))
        logger.info("\t# eventualities", sum([erow["frequency"] for erow in new_erows]))
        logger.info("\t# unique eventualities", len(new_erows))
        logger.info("\t# relations", sum([rfreqs[rrow["_id"]] for rrow in new_rrows]))
        logger.info("\t# unique relations", len(new_rrows))

        if not os.path.exists(os.path.join(args.filtered_kg_dir, str(threadshold), "KG.db")):
            new_kg_conn = SqliteDBConnection(os.path.join(args.filtered_kg_dir, str(threadshold), "KG.db"), CHUNKSIZE)
            for table_name, columns, column_types in zip(
                [EVENTUALITY_TABLE_NAME, RELATION_TABLE_NAME], [EVENTUALITY_COLUMNS, RELATION_COLUMNS],
                [EVENTUALITY_COLUMN_TYPES, RELATION_COLUMN_TYPES]
            ):
                if len(columns) == 0 or len(column_types) == 0:
                    raise NotImplementedError(
                        "Error: %s_columns and %s_column_types must be defined" % (table_name, table_name)
                    )
                try:
                    new_kg_conn.create_table(table_name, columns, column_types)
                except BaseException as e:
                    print(e)
            new_kg_conn.insert_rows(EVENTUALITY_TABLE_NAME, new_erows)
            new_kg_conn.insert_rows(RELATION_TABLE_NAME, new_rrows)
            new_kg_conn.close()

        cid2concept, concept_instance_pairs, cid_to_filter_score = \
            build_concept_instance_table(aser_conceptualizer, new_erows)
        logger.info("\t# unique concepts", len(cid2concept))
        logger.info("\t# unique concept-event relations", len(concept_instance_pairs))

        concept_conn = ASERConceptConnection(
            os.path.join(args.filtered_kg_dir, str(threadshold), "concept.db"), mode="memory"
        )

        with open(os.path.join(args.filtered_kg_dir, str(threadshold), "concept_cids.txt"), "w") as f:
            for cid, filter_score in cid_to_filter_score.items():
                f.write(cid + "\t" + "{:.2f}".format(filter_score) + "\n")
        concept_conn.insert_concepts(list(cid2concept.values()))
        concept_conn.insert_concept_instance_pairs(concept_instance_pairs)

        rid2relation = build_concept_relation_table(concept_conn, new_rrows)
        logger.info("\t# unique concept-concept relations", len(rid2relation))

        with open(os.path.join(args.filtered_kg_dir, str(threadshold), "concept_rids.txt"), "w") as f:
            for rid, relation in rid2relation.items():
                filter_score = sum(relation.relations.values())
                f.write(rid + "\t" + "{:.2f}".format(filter_score) + "\n")
        concept_conn.insert_relations(rid2relation.values())
        concept_conn.close()

        logger.info("\t", time.time() - st)
        del new_erows
        del new_rrows
        del new_eids
        del cid2concept
        del concept_instance_pairs
        del cid_to_filter_score
        del rid2relation
        gc.collect()
