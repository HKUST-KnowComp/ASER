import os
from copy import copy, deepcopy
from aser.database.base import SqliteConnection, MongoDBConnection
from aser.database.kg_connection import CHUNKSIZE
from aser.database.kg_connection import EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS, EVENTUALITY_COLUMN_TYPES
from aser.database.kg_connection import RELATION_TABLE_NAME, RELATION_COLUMNS, RELATION_COLUMN_TYPES
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.utils.logging import init_logger, close_logger

db = "sqlite"
log_path = "./.tmp.log"

if __name__ == "__main__":
    logger = init_logger(log_file=log_path)
    kg_paths = [os.path.join("/metadata/data/corpora/aser/database", kg) for kg in ["gutenberg", "nyt", "reddit", "subtitles", "wikipedia" , "yelp"]]
    merged_kg_path = "/metadata/data/corpora/aser/database/all"
    if not os.path.exists(merged_kg_path):
        os.mkdir(merged_kg_path)
    if db == "sqlite":
        merged_conn = SqliteConnection(os.path.join(merged_kg_path, "KG.db"), CHUNKSIZE)
    elif db == "mongoDB":
        merged_conn = MongoDBConnection(os.path.join(merged_kg_path, "KG.db"), CHUNKSIZE)
    else:
        raise NotImplementedError

    eid2row = dict()
    rid2row = dict()

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
            raise e

    total_eventuality, total_relation = 0, 0
    for kg_path in kg_paths:
        if db == "sqlite":
            conn = SqliteConnection(os.path.join(kg_path, "KG.db"), CHUNKSIZE)
        elif db == "mongoDB":
            conn = MongoDBConnection(os.path.join(kg_path, "KG.db"), CHUNKSIZE)
        else:
            raise NotImplementedError

        logger.info("Retrieving rows from %s.%s..." % (kg_path, EVENTUALITY_TABLE_NAME))
        for row in conn.get_columns(EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS):
            total_eventuality += row["frequency"]
            if row["_id"] not in eid2row:
                eid2row[row["_id"]] = copy(row)
            else:
                eid2row[row["_id"]]["frequency"] += row["frequency"]

        logger.info("Retrieving rows from %s.%s..." % (kg_path, RELATION_TABLE_NAME))
        for row in conn.get_columns(RELATION_TABLE_NAME, RELATION_COLUMNS):
            if row["_id"] not in rid2row:
                total_relation += sum([row.get(r, 0.0) for r in relation_senses])
                rid2row[row["_id"]] = deepcopy(row)
            else:
                for r in relation_senses:
                    freq = row.get(r, 0.0)
                    total_relation += freq
                    rid2row[row["_id"]][r] += freq
    logger.info("%d eventualities (%d unique) have been extracted." % (total_eventuality, len(eid2row)))
    logger.info("%d relations (%d unique) have been extracted." % (total_relation, len(rid2row)))
    
    logger.info("Storing inverted tables and building the KG.")
    merged_conn.insert_rows(EVENTUALITY_TABLE_NAME, eid2row.values())
    merged_conn.insert_rows(RELATION_TABLE_NAME, rid2row.values())
    merged_conn.close()
    logger.info("Done.")
    close_logger(logger)

