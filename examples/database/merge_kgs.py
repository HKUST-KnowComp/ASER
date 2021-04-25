import argparse
import gc
import os
import pickle
import shutil
from collections import defaultdict, Counter
from aser.database.db_connection import SqliteDBConnection, MongoDBConnection
from aser.database.kg_connection import CHUNKSIZE
from aser.database.kg_connection import EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS, EVENTUALITY_COLUMN_TYPES
from aser.database.kg_connection import RELATION_TABLE_NAME, RELATION_COLUMNS, RELATION_COLUMN_TYPES
from aser.relation import relation_senses
from aser.utils.logging import init_logger, close_logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-db", type=str, default="sqlite", choices=["sqlite"])
    parser.add_argument("-kg_path", type=str)
    parser.add_argument("-merged_kg_path", type=str)
    parser.add_argument("-fix_prefix", action="store_true")
    parser.add_argument("-eventuality_frequency_threshold", type=float, default=2.0)
    parser.add_argument("-relation_frequency_threshold", type=float, default=1.0001)
    parser.add_argument("-log_path", type=str, default="merge_kgs.log")

    args = parser.parse_args()

    logger = init_logger(log_file=args.log_path)

    if not os.path.exists(args.merged_kg_path):
        os.mkdir(args.merged_kg_path)

    eid2sids, rid2sids = defaultdict(list), defaultdict(list)

    if os.path.exists(os.path.join(args.kg_path, "eid2sids_full.pkl")):
        logger.info("eid2sids_full.pkl is found in %s. We would copy it directly." % (args.kg_path))
        # shutil.copyfile(os.path.join(args.kg_path, "eid2sids_full.pkl"), os.path.join(args.merged_kg_path, "eid2sids_full.pkl"))
        os.symlink(
            os.path.join(args.kg_path, "eid2sids_full.pkl"), os.path.join(args.merged_kg_path, "eid2sids_full.pkl")
        )
        logger.info("Openning inverted tables")
        with open(os.path.join(args.merged_kg_path, "eid2sids_full.pkl"), "rb") as f:
            eid2sids = pickle.load(f)
    else:
        logger.info("Generating inverted tables")
        for filename in os.listdir(args.kg_path):
            if not (filename.startswith("eid2sids") and filename.endswith(".pkl")):
                continue
            if filename in ["eid2sids_full.pkl", "eid2sids_core.pkl"]:
                continue
            filename = os.path.join(args.kg_path, filename)
            logger.info("Connecting %s" % (filename))
            if args.fix_prefix:
                prefix = ""
                for d in ["nyt", "yelp", "wikipedia", "reddit", "subtitles", "gutenberg"]:
                    if d in filename:
                        prefix = d + os.sep + "parsed_para" + os.sep
                        break
                if prefix == "":
                    logger.warning("Warning: %s is not matched for 6 datasets." % (filename))
                    continue
                with open(filename, "rb") as f:
                    for eid, sids in pickle.load(f).items():
                        eid2sids[eid].extend([prefix + sid for sid in sids])
            else:
                with open(filename, "rb") as f:
                    for eid, sids in pickle.load(f).items():
                        eid2sids[eid].extend(sids)
        logger.info("Storing inverted tables")
        with open(os.path.join(args.merged_kg_path, "eid2sids_full.pkl"), "wb") as f:
            pickle.dump(eid2sids, f)
    gc.collect()

    if os.path.exists(os.path.join(args.kg_path, "rid2sids_full.pkl")):
        logger.info("rid2sids_full.pkl is found in %s. We would copy it directly." % (args.kg_path))
        # shutil.copyfile(os.path.join(args.kg_path, "rid2sids_full.pkl"), os.path.join(args.merged_kg_path, "rid2sids_full.pkl"))
        os.symlink(
            os.path.join(args.kg_path, "rid2sids_full.pkl"), os.path.join(args.merged_kg_path, "rid2sids_full.pkl")
        )
        logger.info("Openning inverted tables")
        with open(os.path.join(args.merged_kg_path, "rid2sids_full.pkl"), "rb") as f:
            rid2sids = pickle.load(f)
    else:
        logger.info("Generating inverted tables")
        for filename in os.listdir(args.kg_path):
            if not (filename.startswith("rid2sids") and filename.endswith(".pkl")):
                continue
            if filename in ["rid2sids_full.pkl", "rid2sids_core.pkl"]:
                continue
            filename = os.path.join(args.kg_path, filename)
            logger.info("Connecting %s" % (filename))
            if args.fix_prefix:
                prefix = ""
                for d in ["nyt", "yelp", "wikipedia", "reddit", "subtitles", "gutenberg"]:
                    if d in filename:
                        prefix = d + os.sep + "parsed_para" + os.sep
                        break
                if prefix == "":
                    logger.warning("Warning: %s is not matched for 6 datasets." % (filename))
                    continue
                with open(filename, "rb") as f:
                    for rid, sids in pickle.load(f).items():
                        rid2sids[rid].extend([tuple([prefix + x for x in sid]) for sid in sids])
            else:
                with open(filename, "rb") as f:
                    for rid, sids in pickle.load(f).items():
                        rid2sids[rid].extend(sids)
        logger.info("Storing inverted tables")
        with open(os.path.join(args.merged_kg_path, "rid2sids_full.pkl"), "wb") as f:
            pickle.dump(rid2sids, f)
    gc.collect()

    logger.info("Connecting %s" % (os.path.join(args.merged_kg_path, "KG.db")))
    if args.db == "sqlite":
        merged_conn = SqliteDBConnection(os.path.join(args.merged_kg_path, "KG.db"), CHUNKSIZE)
    else:
        raise NotImplementedError

    logger.info("Creating tables...")
    # create tables
    for table_name, columns, column_types in zip(
        [EVENTUALITY_TABLE_NAME, RELATION_TABLE_NAME], [EVENTUALITY_COLUMNS, RELATION_COLUMNS],
        [EVENTUALITY_COLUMN_TYPES, RELATION_COLUMN_TYPES]
    ):
        if len(columns) == 0 or len(column_types) == 0:
            raise NotImplementedError(
                "Error: %s_columns and %s_column_types must be defined" % (table_name, table_name)
            )
        try:
            merged_conn.create_table(table_name, columns, column_types)
        except BaseException as e:
            print(e)

    for table_name, columns, column_types in zip(
        [EVENTUALITY_TABLE_NAME, RELATION_TABLE_NAME], [EVENTUALITY_COLUMNS, RELATION_COLUMNS],
        [EVENTUALITY_COLUMN_TYPES, RELATION_COLUMN_TYPES]
    ):
        if len(columns) == 0 or len(column_types) == 0:
            raise NotImplementedError(
                "Error: %s_columns and %s_column_types must be defined" % (table_name, table_name)
            )
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

    rid2row = dict()
    relation_counter = Counter()
    filtered_rids = list()
    for row in merged_conn.get_columns(RELATION_TABLE_NAME, RELATION_COLUMNS):
        relation_counter[row["_id"]] += sum([row.get(r, 0.0) for r in relation_senses])
        rid2row[row["_id"]] = row

    for filename in os.listdir(args.kg_path):
        if not (filename.startswith("KG") and filename.endswith(".db")):
            continue
        filename = os.path.join(args.kg_path, filename)
        logger.info("Connecting %s" % (filename))
        if args.db == "sqlite":
            conn = SqliteDBConnection(filename, CHUNKSIZE)
        else:
            raise NotImplementedError

        logger.info("Retrieving rows from %s.%s..." % (filename, EVENTUALITY_TABLE_NAME))
        for row in conn.get_columns(EVENTUALITY_TABLE_NAME, EVENTUALITY_COLUMNS):
            eventuality_counter[row["_id"]] += row["frequency"]
            if row["_id"] not in eid2row:
                eid2row[row["_id"]] = row
            else:
                eid2row[row["_id"]]["frequency"] += row["frequency"]

        logger.info("Retrieving rows from %s.%s..." % (filename, RELATION_TABLE_NAME))
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
        conn.close()

    total_eventuality = sum(eventuality_counter.values())
    logger.info("%d eventualities (%d unique) have been extracted." % (total_eventuality, len(eventuality_counter)))
    total_relation = sum(relation_counter.values())
    logger.info("%d relations (%d unique) have been extracted." % (total_relation, len(relation_counter)))

    # filter low-frequency eventualities
    logger.info("Filtering low-frequency eventualities.")
    eventuality_frequency_threshold = args.eventuality_frequency_threshold
    for eid, freq in eventuality_counter.items():
        if freq < eventuality_frequency_threshold:
            filtered_eids.append(eid)
    logger.info(
        "%d eventualities (%d unique) will be inserted into the core KG." % (
            total_eventuality - sum([eventuality_counter[eid]
                                     for eid in filtered_eids]), len(eventuality_counter) - len(filtered_eids)
        )
    )
    del eventuality_counter

    if len(filtered_eids) == 0:
        # shutil.copyfile(os.path.join(args.merged_kg_path, "eid2sids_full.pkl"), os.path.join(args.merged_kg_path, "eid2sids_core.pkl"))
        os.symlink(
            os.path.join(args.merged_kg_path, "eid2sids_full.pkl"),
            os.path.join(args.merged_kg_path, "eid2sids_core.pkl")
        )
        eid2row_core = eid2row
        del eid2sids
    else:
        filtered_eids = set(filtered_eids)
        eid2sids_core = defaultdict(list)
        for eid, sids in eid2sids.items():
            if eid not in filtered_eids:
                eid2sids_core[eid] = sids
        with open(os.path.join(args.merged_kg_path, "eid2sids_core.pkl"), "wb") as f:
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
    gc.collect()

    # filter low-frequency relations
    logger.info("Filtering low-frequency relations.")
    relation_frequency_threshold = args.relation_frequency_threshold
    for rid, freq in relation_counter.items():
        if freq < relation_frequency_threshold:
            filtered_rids.append(rid)
        else:
            row = rid2row[rid]
            if row["hid"] not in eid2row_core or row["tid"] not in eid2row_core:
                filtered_rids.append(rid)
    logger.info(
        "%d relations (%d unique) will be inserted into the core KG." % (
            total_relation - sum([relation_counter[rid]
                                  for rid in filtered_rids]), len(relation_counter) - len(filtered_rids)
        )
    )
    del relation_counter
    if len(filtered_rids) == 0:
        # shutil.copyfile(os.path.join(args.merged_kg_path, "rid2sids_full.pkl"), os.path.join(args.merged_kg_path, "rid2sids_core.pkl"))
        os.symlink(
            os.path.join(args.merged_kg_path, "rid2sids_full.pkl"),
            os.path.join(args.merged_kg_path, "rid2sids_core.pkl")
        )
        rid2row_core = rid2row
        del rid2sids
    else:
        filtered_rids = set(filtered_rids)
        rid2sids_core = defaultdict(list)
        for rid, sids in rid2sids.items():
            if rid not in filtered_rids:
                rid2sids_core[rid] = sids
        with open(os.path.join(args.merged_kg_path, "rid2sids_core.pkl"), "wb") as f:
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
    gc.collect()

    merged_conn.close()

    logger.info("Done.")
    close_logger(logger)
