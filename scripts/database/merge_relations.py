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
log_path = "./merge_kg_gutenberg.log"
eventuality_frequency_lower_cnt_threshold = 2
eventuality_frequency_upper_percent_threshold = 1.0
relation_frequency_lower_cnt_threshold = 2
relation_frequency_upper_percent_threshold = 1.0

if __name__ == "__main__":
    logger = init_logger(log_file=log_path)
    # datasets = ["gutenberg", "nyt", "reddit", "subtitles", "wikipedia" , "yelp"]
    # datasets = ["nyt_test"]
    # kg_paths = [os.path.join("/home/data/corpora/aser/database/0.3/gutenberg_full_0.3", dataset) for dataset in datasets]
    # kg_paths = [os.path.join(r"D:\Workspace\ASER-core\data\database", dataset) for dataset in datasets]
    # prefixes_to_be_added = [os.path.join(dataset, "parsed")+os.sep for dataset in datasets]
    # prefixes_to_be_added = [""] * len(kg_paths)
    prefix_to_be_added = ""
    kg_path = "/home/data/corpora/aser/database/0.3/gutenberg_full_0.3"

    merged_kg_path = "/home/data/corpora/aser/database/0.3/gutenberg_full_0.3"
    # merged_kg_path = r"D:\Workspace\ASER-core\data\database\all"
    if not os.path.exists(merged_kg_path):
        os.mkdir(merged_kg_path)

    eid2sids, rid2sids = defaultdict(list), defaultdict(list)
    # for i in range(1, 5):
    #     logger.info("Connecting %s" % (os.path.join(kg_path, "rid2sids_%d.pkl" % (i))))
    #     with open(os.path.join(kg_path, "rid2sids_%d.pkl" % (i)), "rb") as f:
    #         for rid, sids in pickle.load(f).items():
    #             rid2sids[rid].extend([tuple([prefix_to_be_added+x for x in sid]) for sid in sids])
    # logger.info("Storing inverted tables")
    # with open(os.path.join(merged_kg_path, "rid2sids_full.pkl"), "wb") as f:
    #     pickle.dump(rid2sids, f)
    # gc.collect()

    with open(os.path.join(merged_kg_path, "eid2sids_core.pkl"), "rb") as f:
        eid2sids = pickle.load(f)
    with open(os.path.join(merged_kg_path, "rid2sids_full.pkl"), "rb") as f:
        rid2sids = pickle.load(f)

    logger.info("Connecting %s" % (os.path.join(merged_kg_path, "KG.db")))
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
    
    rid2row = dict()
    relation_counter = Counter()
    filtered_rids = list()
    for row in merged_conn.get_columns(RELATION_TABLE_NAME, RELATION_COLUMNS):
        relation_counter[row["_id"]] += sum([row.get(r, 0.0) for r in relation_senses])
        rid2row[row["_id"]] = row

    for i in range(1, 5):
        logger.info("Connecting %s" % (os.path.join(kg_path, "KG_%d.db" % (i))))
        if db == "sqlite":
            conn = SqliteConnection(os.path.join(kg_path, "KG_%d.db" % (i)), CHUNKSIZE)
        elif db == "mongoDB":
            conn = MongoDBConnection(os.path.join(kg_path, "KG_%d.db" % (i)), CHUNKSIZE)
        else:
            raise NotImplementedError

        logger.info("Retrieving rows from %s.%s..." % (kg_path, RELATION_TABLE_NAME))
        for row in conn.get_columns(RELATION_TABLE_NAME, RELATION_COLUMNS):
            if row["_id"] not in rid2sids:
                continue
            freq = sum([row.get(r, 0.0) for r in relation_senses])
            relation_counter[row["_id"]] += freq
            if row["_id"] not in rid2row:
                rid2row[row["_id"]] = row
            else:
                for r in relation_senses:
                    rid2row[row["_id"]][r] += row.get(r, 0.0)

    total_relation = sum(relation_counter.values())
    logger.info("%d relations (%d unique) have been extracted." % (total_relation, len(relation_counter)))

    # filter high-frequency and low-frequency relations
    logger.info("Filtering high-frequency and low-frequency relations.")
    relation_frequency_lower_cnt_threshold = relation_frequency_lower_cnt_threshold
    relation_frequency_upper_cnt_threshold = relation_frequency_upper_percent_threshold * total_relation
    for rid, freq in relation_counter.items():
        if freq < relation_frequency_lower_cnt_threshold or freq > relation_frequency_upper_cnt_threshold:
            filtered_rids.append(rid)
        else:
            row = rid2row[rid]
            if row["hid"] not in eid2sids or row["tid"] not in eid2sids:
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

