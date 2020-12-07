import os
import pickle
import shutil
import gc
from copy import copy, deepcopy
from collections import defaultdict, Counter
from aser.database.base import SqliteConnection, MongoDBConnection
from aser.database.kg_connection import CHUNKSIZE
from aser.database.kg_connection import EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS, EVENTUALITY_COLUMN_TYPES
from aser.database.kg_connection import RELATION_TABLE_NAME, RELATION_COLUMNS, RELATION_COLUMN_TYPES
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.utils.logging import init_logger, close_logger

db = "sqlite"
log_path = "./.merge_kg.log"
eventuality_frequency_lower_cnt_threshold = 2
eventuality_frequency_upper_percent_threshold = 1.0
relation_frequency_lower_cnt_threshold = 2
relation_frequency_upper_percent_threshold = 1.0

if __name__ == "__main__":
    logger = init_logger(log_file=log_path)
    datasets = ["yelp", "nyt", "wikipedia", "reddit", "subtitles", "gutenberg"]
    kg_paths = ["/home/data/corpora/aser/database/0.3/%s_core_0.3" % (dataset) for dataset in datasets]
    prefixes_to_be_added = [os.path.join(dataset, "parsed_para")+os.sep for dataset in datasets]

    # merged_kg_path = "/home/data/corpora/aser/database/0.3/reddit_full_0.3"
    merged_kg_path = "/home/data/corpora/aser/database/0.3/core"
    if not os.path.exists(merged_kg_path):
        os.mkdir(merged_kg_path)

    eid2sids, rid2sids = defaultdict(list), defaultdict(list)
    for kg_path, prefix_to_be_added in zip(kg_paths, prefixes_to_be_added):
        logger.info("Connecting %s" % (os.path.join(kg_path, "eid2sids.pkl")))
        with open(os.path.join(kg_path, "eid2sids.pkl"), "rb") as f:
            for eid, sids in pickle.load(f).items():
                eid2sids[eid].extend([prefix_to_be_added+sid for sid in sids])

        logger.info("Connecting %s" % (os.path.join(kg_path, "rid2sids.pkl")))
        with open(os.path.join(kg_path, "rid2sids.pkl"), "rb") as f:
            for rid, sids in pickle.load(f).items():
                rid2sids[rid].extend([tuple([prefix_to_be_added+x for x in sid]) for sid in sids])
    logger.info("Storing inverted tables")
    with open(os.path.join(merged_kg_path, "eid2sids.pkl"), "wb") as f:
        pickle.dump(eid2sids, f)
    with open(os.path.join(merged_kg_path, "rid2sids.pkl"), "wb") as f:
        pickle.dump(rid2sids, f)
    del eid2sids
    del rid2sids
    # gc.collect()


    if db == "sqlite":
        merged_conn = SqliteConnection(os.path.join(merged_kg_path, "KG.db"), CHUNKSIZE)
    elif db == "mongoDB":
        merged_conn = MongoDBConnection(os.path.join(merged_kg_path, "KG.db"), CHUNKSIZE)
    else:
        raise NotImplementedError

    logger.info("Creating tables...")
    # create tables
    for table_name, columns, column_types in zip(
        [EVENTUALITY_TABLE_NAME, RELATION_TABLE_NAME],
        [EVENTUALITY_COLUMNS, RELATION_COLUMNS],
        [EVENTUALITY_COLUMN_TYPES, RELATION_COLUMN_TYPES]):
        if len(columns) == 0 or len(column_types) == 0:
            raise NotImplementedError("Error: %s_columns and %s_column_types must be defined" % (table_name, table_name))
        try:
            merged_conn.create_table(table_name, columns, column_types)
        except BaseException as e:
            print(e)
    
    eid2row, rid2row = dict(), dict()
    eventuality_counter, relation_counter = Counter(), Counter()
    for kg_path in kg_paths:
        logger.info("Connecting %s" % (os.path.join(kg_path, "KG.db")))
        if db == "sqlite":
            conn = SqliteConnection(os.path.join(kg_path, "KG.db"), CHUNKSIZE)
        elif db == "mongoDB":
            conn = MongoDBConnection(os.path.join(kg_path, "KG.db"), CHUNKSIZE)
        else:
            raise NotImplementedError

        logger.info("Retrieving rows from %s.%s..." % (kg_path, EVENTUALITY_TABLE_NAME))
        for row in conn.get_columns(EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS):
            eventuality_counter[row["_id"]] += row["frequency"]
            if row["_id"] not in eid2row:
                eid2row[row["_id"]] = deepcopy(row)
            else:
                eid2row[row["_id"]]["frequency"] += row["frequency"]

        logger.info("Retrieving rows from %s.%s..." % (kg_path, RELATION_TABLE_NAME))
        for row in conn.get_columns(RELATION_TABLE_NAME, RELATION_COLUMNS):
            relation_counter[row["_id"]] += sum([row.get(r, 0.0) for r in relation_senses])
            if row["_id"] not in rid2row:
                rid2row[row["_id"]] = deepcopy(row)
            else:
                for r in relation_senses:
                    rid2row[row["_id"]][r] += row.get(r, 0.0)
        conn.close()
    total_eventuality, total_relation = sum(eventuality_counter.values()), sum(relation_counter.values())
    logger.info("%d eventualities (%d unique) have been extracted." % (total_eventuality, len(eid2row)))
    logger.info("%d relations (%d unique) have been extracted." % (total_relation, len(rid2row)))

    # filter high-frequency and low-frequency eventualities
    logger.info("Filtering high-frequency and low-frequency eventualities.")
    eventuality_frequency_lower_cnt_threshold = eventuality_frequency_lower_cnt_threshold
    eventuality_frequency_upper_cnt_threshold = eventuality_frequency_upper_percent_threshold * total_eventuality
    for eid, freq in eventuality_counter.items():
        if freq < eventuality_frequency_lower_cnt_threshold or freq > eventuality_frequency_upper_cnt_threshold:
            filtered_eids.append(eid)
    logger.info("%d eventualities (%d unique) will be inserted into the core KG." % (
        total_eventuality-sum([eventuality_counter[eid] for eid in filtered_eids]), len(eventuality_counter)-len(filtered_eids)))
    del eventuality_counter
    if len(filtered_eids) == 0:
        shutil.copyfile(os.path.join(merged_kg_path, "eid2sids_full.pkl"), os.path.join(merged_kg_path, "eid2sids_core.pkl"))
        eid2row_core = eid2row
        del eid2sids 
    else:
        filtered_eids = set(filtered_eids)
        eid2sids_core = defaultdict(list)
        for eid, sids in eid2sids.items():
            if eid not in filtered_eids:
                eid2sids_core[eid] = sids
        with open(os.path.join(merged_kg_path, "eid2sids_core.pkl"), "wb") as f:
            pickle.dump(eid2sids_core, f)
        del eid2sids_core
        del eid2sids
        eid2row_core = dict()
        for eid, row in eid2row.items():
            if eid not in filtered_eids:
                eid2row_core[eid] = row
    del filtered_eids
    del eid2row
    merged_conn.insert_rows(EVENTUALITY_TABLE_NAME, eid2row_core.values())
    # gc.collect()

    # filter high-frequency and low-frequency relations
    logger.info("Filtering high-frequency and low-frequency relations.")
    relation_frequency_lower_cnt_threshold = relation_frequency_lower_cnt_threshold
    relation_frequency_upper_cnt_threshold = relation_frequency_upper_percent_threshold * total_relation
    for rid, freq in relation_counter.items():
        if freq < relation_frequency_lower_cnt_threshold or freq > relation_frequency_upper_cnt_threshold:
            filtered_rids.append(rid)
        else:
            row = rid2row[rid]
            if row["hid"] not in eid2row_core or row["tid"] not in eid2row_core:
                filtered_rids.append(rid)
    logger.info("%d relations (%d unique) will be inserted into the core KG." % (
        total_relation-sum([relation_counter[rid] for rid in filtered_rids]), len(relation_counter)-len(filtered_rids)))
    del relation_counter
    if len(filtered_rids) == 0:
        shutil.copyfile(os.path.join(merged_kg_path, "rid2sids_full.pkl"), os.path.join(merged_kg_path, "rid2sids_core.pkl"))
        rid2row_core = rid2row
        del rid2sids 
    else:
        filtered_rids = set(filtered_rids)
        rid2sids_core = defaultdict(list)
        for rid, sids in rid2sids.items():
            if rid not in filtered_rids:
                rid2sids_core[rid] = sids
        with open(os.path.join(merged_kg_path, "rid2sids_core.pkl"), "wb") as f:
            pickle.dump(rid2sids_core, f)
        del rid2sids_core
        del rid2sids
        rid2row_core = dict()
        for rid, row in rid2row.items():
            if rid not in filtered_rids:
                rid2row_core[rid] = row
    del filtered_rids
    del rid2row
    merged_conn.insert_rows(RELATION_TABLE_NAME, rid2row_core.values())
    # gc.collect()

    merged_conn.close()

    logger.info("Done.")
    close_logger(logger)

