import os
import pickle
import shutil
import gc
import bisect
from copy import copy, deepcopy
from collections import defaultdict, Counter
from aser.database.base import SqliteConnection, MongoDBConnection
from aser.database.kg_connection import CHUNKSIZE
from aser.database.kg_connection import EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS, EVENTUALITY_COLUMN_TYPES
from aser.database.kg_connection import RELATION_TABLE_NAME, RELATION_COLUMNS, RELATION_COLUMN_TYPES
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.utils.logging import init_logger, close_logger

def find(a, x):
    i = bisect.bisect_left(a, x)
    return i != len(a) and a[i] == x

def index(a, x):
    'Locate the leftmost value exactly equal to x'
    i = bisect.bisect_left(a, x)
    if i != len(a) and a[i] == x:
        return i
    raise -1

db = "sqlite"
log_path = "./.split_kg.log"
MAX_EVENTUALITIES = CHUNKSIZE * 32 * 2
MAX_RELATIONS = CHUNKSIZE * 32

if __name__ == "__main__":
    logger = init_logger(log_file=log_path)
    
    slice_kg_path = "/home/data/corpora/aser/database/0.3/slice"
    kg_path = "/home/data/corpora/aser/database/0.3/core"
    if not os.path.exists(slice_kg_path):
        os.mkdir(slice_kg_path)

    logger.info("Connecting %s" % (os.path.join(kg_path, "KG.db")))
    if db == "sqlite":
        conn = SqliteConnection(os.path.join(kg_path, "KG.db"), CHUNKSIZE)
    elif db == "mongoDB":
        conn = MongoDBConnection(os.path.join(kg_path, "KG.db"), CHUNKSIZE)
    else:
        raise NotImplementedError

    eid2row, hid2rows, tids = dict(), defaultdict(list), list()
    logger.info("Retrieving rows from %s.%s..." % (kg_path, EVENTUALITY_TABLE_NAME))
    for row in conn.get_columns(EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS):
        eid2row[row["_id"]] = row

    logger.info("Retrieving rows from %s.%s..." % (kg_path, RELATION_TABLE_NAME))
    for row in conn.get_columns(RELATION_TABLE_NAME, RELATION_COLUMNS):
        hid2rows[row["hid"]].append(row)
        tids.append(row["tid"])
    conn.close()

    hids = sorted(hid2rows)
    tids = set(tids)
    singleton_eids = [eid for eid in eid2row if not (find(hids, eid) or eid in tids)]
    singleton_eids.sort()

    kg_idx = 0
    for idx in range(0, len(singleton_eids), MAX_EVENTUALITIES):
        logger.info("Creating KG_e_%d.db..." % (kg_idx))
        slice_conn = SqliteConnection(os.path.join(slice_kg_path, "KG_e_%d.db" % (kg_idx)), CHUNKSIZE)
        kg_idx += 1
        logger.info("Creating tables...")
        # create tables
        for table_name, columns, column_types in zip(
            [EVENTUALITY_TABLE_NAME, RELATION_TABLE_NAME],
            [EVENTUALITY_COLUMNS, RELATION_COLUMNS],
            [EVENTUALITY_COLUMN_TYPES, RELATION_COLUMN_TYPES]):
            if len(columns) == 0 or len(column_types) == 0:
                raise NotImplementedError("Error: %s_columns and %s_column_types must be defined" % (table_name, table_name))
            try:
                slice_conn.create_table(table_name, columns, column_types)
            except BaseException as e:
                print(e)
        logger.info("Slicing data...")
        e_rows = [eid2row[eid] for eid in singleton_eids[idx:idx+MAX_EVENTUALITIES]]
        logger.info("Inserting %d eventualities and %d relations..." % (len(e_rows), 0))
        slice_conn.insert_rows(EVENTUALITY_TABLE_NAME, e_rows)
        slice_conn.close()
    
    h_idx = 0
    while h_idx < len(hids):
        logger.info("Creating KG_r_%d.db..." % (kg_idx))
        slice_conn = SqliteConnection(os.path.join(slice_kg_path, "KG_r_%d.db" % (kg_idx)), CHUNKSIZE)
        kg_idx += 1
        logger.info("Creating tables...")
        # create tables
        for table_name, columns, column_types in zip(
            [EVENTUALITY_TABLE_NAME, RELATION_TABLE_NAME],
            [EVENTUALITY_COLUMNS, RELATION_COLUMNS],
            [EVENTUALITY_COLUMN_TYPES, RELATION_COLUMN_TYPES]):
            if len(columns) == 0 or len(column_types) == 0:
                raise NotImplementedError("Error: %s_columns and %s_column_types must be defined" % (table_name, table_name))
            try:
                slice_conn.create_table(table_name, columns, column_types)
            except BaseException as e:
                print(e)
        logger.info("Slicing data...")
        non_singleton_eids = set()
        r_rows = list()
        while h_idx < len(hids) and len(non_singleton_eids) < MAX_EVENTUALITIES and len(r_rows) < MAX_RELATIONS:
            rows = hid2rows[hids[h_idx]]
            rows.sort(key=lambda x: x["tid"])
            non_singleton_eids.add(hids[h_idx])
            non_singleton_eids.update([row["tid"] for row in rows])
            r_rows.extend(rows)
            h_idx += 1
        e_rows = [eid2row[eid] for eid in sorted(non_singleton_eids)]
        logger.info("Inserting %d eventualities and %d relations..." % (len(e_rows), len(r_rows)))
        slice_conn.insert_rows(EVENTUALITY_TABLE_NAME, e_rows)
        slice_conn.insert_rows(RELATION_TABLE_NAME, r_rows)
        slice_conn.close()

    logger.info("Done.")
    close_logger(logger)

