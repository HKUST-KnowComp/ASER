import argparse
import copy
import gc
import os
import time
from collections import defaultdict, Counter
from tqdm import tqdm
from aser.database.db_connection import SqliteDBConnection, MongoDBConnection
from aser.database.kg_connection import CHUNKSIZE
from aser.database.kg_connection import EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS, EVENTUALITY_COLUMN_TYPES
from aser.database.kg_connection import RELATION_TABLE_NAME, RELATION_COLUMNS, RELATION_COLUMN_TYPES
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

        logger.info("\t", time.time() - st)
        del new_erows
        del new_rrows
        del new_eids
        gc.collect()
