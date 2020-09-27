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
log_path = "./merge_kg_reddit.log"
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
    kg_path = "/home/data/corpora/aser/database/0.3/reddit_full_0.3"

    merged_kg_path = "/home/data/corpora/aser/database/0.3/reddit_full_0.3"
    # merged_kg_path = r"D:\Workspace\ASER-core\data\database\all"
    if not os.path.exists(merged_kg_path):
        os.mkdir(merged_kg_path)

    eid2sids, rid2sids = defaultdict(list), defaultdict(list)
    for i in range(1, 5):
        logger.info("Connecting %s" % (os.path.join(kg_path, "eid2sids_%d.pkl" % (i))))
        with open(os.path.join(kg_path, "eid2sids_%d.pkl" % (i)), "rb") as f:
            for eid, sids in pickle.load(f).items():
                eid2sids[eid].extend([prefix_to_be_added+sid for sid in sids])
    logger.info("Storing inverted tables")
    with open(os.path.join(merged_kg_path, "eid2sids_full.pkl"), "wb") as f:
        pickle.dump(eid2sids, f)
    gc.collect()

    # with open(os.path.join(merged_kg_path, "eid2sids_full.pkl"), "rb") as f:
    #     eid2sids = pickle.load(f)
    
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
    
    eid2row = dict()
    eventuality_counter = Counter()
    filtered_eids = list()
    for row in merged_conn.get_columns(EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS):
        eventuality_counter[row["_id"]] += row["frequency"]
        eid2row[row["_id"]] = row

    for i in range(1, 5):
        logger.info("Connecting %s" % (os.path.join(kg_path, "KG_%d.db" % (i))))
        if db == "sqlite":
            conn = SqliteConnection(os.path.join(kg_path, "KG_%d.db" % (i)), CHUNKSIZE)
        elif db == "mongoDB":
            conn = MongoDBConnection(os.path.join(kg_path, "KG_%d.db" % (i)), CHUNKSIZE)
        else:
            raise NotImplementedError

        logger.info("Retrieving rows from %s.%s..." % (kg_path, EVENTUALITY_TABLE_NAME))
        for row in conn.get_columns(EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS):
            eventuality_counter[row["_id"]] += row["frequency"]
            if row["_id"] not in eid2row:
                eid2row[row["_id"]] = row
            else:
                eid2row[row["_id"]]["frequency"] += row["frequency"]

    total_eventuality = sum(eventuality_counter.values())
    logger.info("%d eventualities (%d unique) have been extracted." % (total_eventuality, len(eventuality_counter)))

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
    del eid2row_core
    gc.collect()

    merged_conn.close()

    logger.info("Done.")
    close_logger(logger)

