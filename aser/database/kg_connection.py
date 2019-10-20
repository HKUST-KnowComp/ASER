try:
    import ujson as json
except:
    import json
import os
import pandas as pd
import numpy as np
import functools
import pickle
import csv
import random
import heapq
import operator
from copy import copy
from functools import partial
from collections import defaultdict, OrderedDict
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from .util import *

CHUNKSIZE = 32768
EVENT_TABLE_NAME = "Eventualities"
EVENT_COLUMNS = ["_id", "frequency", "pattern", "verbs", "skeleton_words", "words", "info"]
EVENT_COLUMN_TYPES = ["PRIMARY KEY", "REAL", "TEXT", "TEXT", "TEXT", "TEXT", "BLOB"]
RELATION_TABLE_NAME = "Relations"
RELATION_COLUMNS = ["_id", "heid", "teid"] + relation_senses
RELATION_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"] + ["REAL"] * len(relation_senses)

class _BaseConnection(object):
    def __init__(self, db_path, chunksize):
        self._conn = None
        self.chunksize = chunksize

    def close(self):
        if self._conn:
            self._conn.close()

    def __del__(self):
        self.close()

    def create_table(self, table_name, columns):
        raise NotImplementedError

    def get_columns(self, table_name, columns):
        raise NotImplementedError

    def select_row(self, table_name, _id, columns):
        raise NotImplementedError

    def select_rows(self, table_name, _ids, columns):
        raise NotImplementedError

    def insert_row(self, table_name, row):
        raise NotImplementedError

    def insert_rows(self, table_name, rows):
        raise NotImplementedError

    def update_row(self, table_name, row, update_op, update_columns):
        raise NotImplementedError

    def update_rows(self, table_name, rows, update_ops, update_columns):
        raise NotImplementedError

    def get_update_op(self, update_columns, operator):
        raise NotImplementedError

    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        raise NotImplementedError


class _SqliteConnection(_BaseConnection):
    def __init__(self, db_path, chunksize):
        import sqlite3
        super(_SqliteConnection, self).__init__(db_path, chunksize)
        self._conn = sqlite3.connect(db_path)

    def create_table(self, table_name, columns, columns_types):
        create_table = "CREATE TABLE %s (%s);" % (table_name, ",".join(
            [' '.join(x) for x in zip(columns, columns_types)]))
        self._conn.execute(create_table)
        self._conn.commit()

    def get_columns(self, table_name, columns):
        select_table = "SELECT %s FROM %s;" % (",".join(columns), table_name)
        result = list(map(lambda x: OrderedDict(zip(columns, x)), self._conn.execute(select_table)))
        return result

    def select_row(self, table_name, _id, columns):
        select_table = "SELECT %s FROM %s WHERE _id=?;" % (",".join(columns), table_name)
        result = list(self._conn.execute(select_table, [_id]))
        if len(result) == 0:
            return None
        else:
            return OrderedDict(zip(columns, result[0]))

    def select_rows(self, table_name, _ids, columns):
        if len(_ids) > 0:
            row_cache = dict()
            result = []
            for idx in range(0, len(_ids), self.chunksize):
                select_table = "SELECT %s FROM %s WHERE _id IN ('%s');" % (
                    ",".join(columns), table_name, "','".join(_ids[idx:idx+self.chunksize]))
                result.extend(list(self._conn.execute(select_table)))
            for x in result:
                exact_match_row = OrderedDict(zip(columns, x))
                row_cache[exact_match_row["_id"]] = exact_match_row
            exact_match_rows = []
            for _id in _ids:
                exact_match_rows.append(row_cache.get(_id, None))
            return exact_match_rows
        else:
            return []

    def insert_row(self, table_name, row):
        insert_table = "INSERT INTO %s VALUES (%s)" % (table_name, ",".join(['?'] * (len(row))))
        self._conn.execute(insert_table, list(row.values()))
        self._conn.commit()

    def insert_rows(self, table_name, rows):
        if len(rows) > 0:
            insert_table = "INSERT INTO %s VALUES (%s)" % (table_name, ",".join(['?'] * (len(rows[0]))))
            self._conn.executemany(insert_table, [list(row.values()) for row in rows])
            self._conn.commit()

    def _update_update_op(self, row, update_op, update_columns):
        update_op_sp = update_op.split('?')
        while len(update_op_sp) >= 0 and update_op_sp[-1] == '':
            update_op_sp.pop()
        assert len(update_op_sp) == len(update_columns)
        new_update_op = []
        for i in range(len(update_op_sp)):
            new_update_op.append(update_op_sp[i])
            if isinstance(row[update_columns[i]], str):
                new_update_op.append("'" + row[update_columns[i]].replace("'", "''") + "'")
            else:
                new_update_op.append(str(row[update_columns[i]]))
        return ''.join(new_update_op)

    def update_row(self, table_name, row, update_op, update_columns):
        update_table = "UPDATE %s SET %s WHERE _id=?" % (table_name, update_op)
        self._conn.execute(update_table, [row[k] for k in update_columns] + [row["_id"]])
        self._conn.commit()

    def update_rows(self, table_name, rows, update_ops, update_columns):
        if len(rows) > 0:
            if isinstance(update_ops, (tuple, list)): # +-*/
                assert len(rows) == len(update_ops)
                # group rows by op to speed up
                update_op_collections = defaultdict(list)  # key: _update_update_op
                for i, row in enumerate(rows):
                    # self.update_row(row, table_name, update_ops[i], update_columns)
                    new_update_op = self._update_update_op(row, update_ops[i], update_columns)
                    update_op_collections[new_update_op].append(row)
                for new_update_op, op_rows in update_op_collections.items():
                    _ids = [row["_id"] for row in op_rows]
                    for idx in range(0, len(_ids), self.chunksize):
                        update_table = "UPDATE %s SET %s WHERE _id IN ('%s');" % (
                            table_name, new_update_op, "','".join(_ids[idx:idx+self.chunksize]))
                        self._conn.execute(update_table)
            else: # =
                update_op = update_ops
                # group rows by new values to speed up
                value_collections = defaultdict(list) # key: values of new values
                for row in rows:
                    # self.update_row(row, table_name, update_op, update_columns)
                    value_collections[json.dumps([row[k] for k in update_columns])].append(row)
                for new_update_op, op_rows in value_collections.items():
                    new_update_op = self._update_update_op(op_rows[0], update_op, update_columns)
                    _ids = [row["_id"] for row in op_rows]
                    for idx in range(0, len(_ids), self.chunksize):
                        update_table = "UPDATE %s SET %s WHERE _id IN ('%s');" % (
                            table_name, new_update_op, "','".join(_ids[idx:idx+self.chunksize]))
                        self._conn.execute(update_table)
            self._conn.commit()
            """
            if isinstance(update_ops, list) or isinstance(update_ops, tuple):
                assert len(rows) == len(update_ops)
                for i, row in enumerate(rows):
                    self.update_row(row, table_name, update_ops[i], update_columns)
            else:
                update_op = update_ops
                update_table = "UPDATE %s SET %s WHERE _id=?" % (
                    table_name, update_op)
                self._conn.executemany(
                    update_table, [[row[k] for k in update_columns] + [row["_id"]] for row in rows])
            self._conn.commit()
            """

    def get_update_op(self, update_columns, operator):
        if operator in "+-*/":
            update_ops = []
            for update_column in update_columns:
                update_ops.append(update_column + "=" + update_column + operator + "?")
            return ",".join(update_ops)
        elif operator == "=":
            update_ops = []
            for update_column in update_columns:
                update_ops.append(update_column + "=?")
            return ",".join(update_ops)
        else:
            raise NotImplementedError

    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        key_match_events = []
        select_table = "SELECT %s FROM %s WHERE %s" % (
            ",".join(columns), table_name, " AND ".join(["%s=?" % (by) for by in bys]))
        if order_bys:
            select_table += " ORDER BY %s %s" % (",".join(order_bys), "DESC" if reverse else "ASC")
        if top_n:
            select_table += " LIMIT %d" % (top_n)
        select_table += ";"
        for x in self._conn.execute(select_table, keys):
            key_match_event = OrderedDict(zip(columns, x))
            key_match_events.append(key_match_event)
        return key_match_events


class _MongoDBConnection(_BaseConnection):
    def __init__(self, db_path, chunksize):
        import pymongo
        super(_MongoDBConnection, self).__init__(db_path, chunksize)
        self._client = pymongo.MongoClient("mongodb://localhost:27017/", document_class=OrderedDict)
        self._conn = self._client[os.path.splitext(os.path.basename(db_path))[0]]

    def close(self):
        self._client.close()

    def create_table(self, table_name, columns, columns_types):
        self._conn[table_name]

    def __get_projection(self, columns):
        projection = {"_id": 0}
        for k in columns:
            projection[k] = 1
        return projection

    def get_columns(self, table_name, columns):
        projection = self.__get_projection(columns)
        results = list(self._conn[table_name].find({}, projection))
        return results

    def select_row(self, table_name, _id, columns):
        projection = self.__get_projection(columns)
        return self._conn[table_name].find_one({"_id": _id}, projection)

    def select_rows(self, table_name, _ids, columns):
        table = self._conn[table_name]
        exact_match_rows = []
        projection = self.__get_projection(columns)
        for idx in range(0, len(_ids), self.chunksize):
            query = {"_id": {'$in': _ids[idx:idx+self.chunksize]}}
            exact_match_rows.extend(table.find(query, projection))
        row_cache = {x["_id"]: x for x in exact_match_rows}
        exact_match_rows = [row_cache.get(_id, None) for _id in _ids]
        return exact_match_rows

    def insert_row(self, table_name, row):
        self._conn[table_name].insert_one(row)

    def insert_rows(self, table_name, rows):
        self._conn[table_name].insert_many(rows)

    def _update_update_op(self, row, update_op, update_columns):
        new_update_op = update_op.copy()
        for k, v in new_update_op.items():
            if k == "$inc":
                for update_column in update_columns:
                    if v[update_column] == 1:
                        v[update_column] = row[update_column]
                    else:
                        v[update_column] = -row[update_column]
            elif k == "$mul":
                for update_column in update_columns:
                    if v[update_column] == 2:
                        v[update_column] = row[update_column]
                    else:
                        v[update_column] = 1.0 / row[update_column]
            elif k == "$set":
                for update_column in update_columns:
                    v[update_column] = row[update_column]
        return new_update_op

    def update_row(self, table_name, row, update_op, update_columns):
        self._conn[table_name].update_one(
            {"_id": row["_id"]}, self._update_update_op(row, update_op, update_columns))

    def update_rows(self, table_name, rows, update_ops, update_columns):
        if len(rows) > 0:
            if isinstance(update_ops, (tuple, list)): # +-*/
                assert len(rows) == len(update_ops)
                update_op_collections = defaultdict(list)
                for i, row in enumerate(rows):
                    # self.update_row(row, table_name, update_ops[i], update_columns)
                    new_update_op = self._update_update_op(row, update_ops[i], update_columns)
                    update_op_collections[json.dumps(new_update_op)].append(row)
                for new_update_op, op_rows in update_op_collections.items():
                    new_update_op = json.loads(new_update_op)
                    _ids = [row["_id"] for row in op_rows]
                    for idx in range(0, len(_ids), self.chunksize):
                        query = {"_id": {'$in': _ids[idx:idx+self.chunksize]}}
                        self._conn[table_name].update_many(query, new_update_op)
            else: # =
                update_op = update_ops
                value_collections = defaultdict(list)
                for row in rows:
                    # self.update_row(row, table_name, update_op, update_columns)
                    value_collections[json.dumps([row[k] for k in update_columns])].append(row)
                for new_update_op, op_rows in value_collections.items():
                    new_update_op = self._update_update_op(op_rows[0], update_op, update_columns)
                    _ids = [row["_id"] for row in op_rows]
                    for idx in range(0, len(_ids), self.chunksize):
                        query = {"_id": {'$in': _ids[idx:idx+self.chunksize]}}
                        self._conn[table_name].update_many(query, new_update_op)

    def get_update_op(self, update_columns, operator):
        if operator == "+":
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 1  # placeholder
            return {"$inc": update_ops}
        elif operator == "-":
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = -1  # placeholder
            return {"$inc": update_ops}
        elif operator == "*":
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 2  # placeholder
            return {"$mul": update_ops}
        elif operator == "/":
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 0.5  # placeholder
            return {"$mul": update_ops}
        elif operator == "=":
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 1  # placeholder
            return {"$set": update_ops}
        else:
            raise NotImplementedError

    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        query = OrderedDict(zip(bys, keys))
        projection = self.__get_projection(columns)
        cursor = self._conn[table_name].find(query, projection)
        if order_bys:
            direction = -1 if reverse else 1
            cursor = cursor.sort([(k, direction) for k in order_bys])
        if top_n:
            result = []
            for x in cursor:
                result.append(x)
                if len(result) >= top_n:
                    break
            return result
        else:
            return list(cursor)


class KGConnection(object):

    def __init__(self, db_path, db="sqlite", mode='cache', grain=None, chunksize=-1):
        if db == 'sqlite':
            self._conn = _SqliteConnection(db_path, chunksize if chunksize > 0 else CHUNKSIZE)
        elif db == 'mongoDB':
            self._conn = _MongoDBConnection(db_path, chunksize if chunksize > 0 else CHUNKSIZE)
        else:
            raise NotImplementedError("%s database is not supported!" % (db))
        self.mode = mode
        if self.mode not in ['insert', 'cache', 'memory']:
            raise NotImplementedError(
                "only support event/relation querying only support insert/cache/memory modes.")

        if grain not in [None, "verbs", "skeleton_words", "words"]:
            raise NotImplementedError("only support event/relation querying only support None/verbs/skeleton_words/words grain.")
        self.grain = grain  # None, verbs, skeleton_words, words

        self.event_table_name = EVENT_TABLE_NAME
        self.event_columns = EVENT_COLUMNS
        self.event_column_types = EVENT_COLUMN_TYPES
        self.relation_table_name = RELATION_TABLE_NAME
        self.relation_columns = RELATION_COLUMNS
        self.relation_column_types = RELATION_COLUMN_TYPES

        self.eids = set()
        self.rids = set()
        self.eid2event_cache = dict()
        self.rid2relation_cache = dict()        
        if self.grain == "words":
            self.partial2eids_cache = {"verbs": dict(), "skeleton_words": dict(), "words": dict()}
        elif self.grain == "skeleton_words":
            self.partial2eids_cache = {"verbs": dict(), "skeleton_words": dict()}
        elif self.grain == "verbs":
            self.partial2eids_cache = {"verbs": dict()}
        else:
            self.partial2eids_cache = dict()
        self.partial2rids_cache = {"heid": dict()}

        self.init()

    def init(self):
        """
        create tables
        load id sets
        load cache
        """
        if len(self.event_columns) == 0 or len(self.event_column_types) == 0:
            raise NotImplementedError(
                "only support event/relation querying event_columns and event_column_types must be defined")
        try:
            self._conn.create_table(
                self.event_table_name, self.event_columns, self.event_column_types)
        except:
            pass
        try:
            self._conn.create_table(
                self.relation_table_name, self.relation_columns, self.relation_column_types)
        except:
            pass
        if self.mode == 'memory':
            for e in map(self._convert_row_to_event, self._conn.get_columns(self.event_table_name, self.event_columns)):
                self.eids.add(e["_id"])
                self.eid2event_cache[e["_id"]] = e
                # handle another cache
                for k, v in self.partial2eids_cache.items():
                    if e[k] not in v:
                        v[e[k]] = [e["_id"]]
                    else:
                        v[e[k]].append(e["_id"])
            for r in map(self._convert_row_to_relation, self._conn.get_columns(self.relation_table_name, self.relation_columns)):
                self.rids.add(r["_id"])
                self.rid2relation_cache[r["_id"]] = r
                # handle another cache
                for k, v in self.partial2rids_cache.items():
                    if r[k] not in v:
                        v[r[k]] = [r["_id"]]
                    else:
                        v[r[k]].append(r["_id"])
        else:
            for e in self._conn.get_columns(self.event_table_name, ["_id"]):
                self.eids.add(e["_id"])
            for r in self._conn.get_columns(self.relation_table_name, ["_id"]):
                self.rids.add(r["_id"])

    def close(self):
        self._conn.close()
        self.eids.clear()
        self.rids.clear()
        self.eid2event_cache.clear()
        self.rid2relation_cache.clear()
        # close another cache
        for k in self.partial2eids_cache:
            self.partial2eids_cache[k].clear()
        for k in self.partial2rids_cache:
            self.partial2rids_cache[k].clear()

    """
    KG (Eventualities)
    """
    def _convert_event_to_row(self, event):
        row = OrderedDict({"_id": event.eid})
        for c in self.event_columns[1:-1]:
            d = getattr(event, c)
            if isinstance(d, list):
                row[c] = " ".join(d)
            else:
                row[c] = d
        row["info"] = event.encode()
        return row
    
    def _convert_row_to_event(self, row):
        return Eventuality().decode(row["info"])

    def get_event_columns(self, columns):
        return self._conn.get_columns(self.event_table_name, columns)

    def _insert_event(self, event):
        row = self._convert_event_to_row(event)
        self._conn.insert_row(self.event_table_name, row)
        if self.mode == 'insert':
            self.eids.add(event.eid)
        elif self.mode == 'cache':
            self.eids.add(event.eid)
            self.eid2event_cache[event.eid] = event
            for k, v in self.partial2eids_cache.items():
                if event.get(k) in v:
                    v[event.get(k)].append(event.eid)
        elif self.mode == 'memory':
            self.eids.add(event.eid)
            self.eid2event_cache[event.eid] = event
            for k, v in self.partial2eids_cache.items():
                if event.get(k) not in v:
                    v[event.get(k)] = [event.eid]
                else:
                    v[event.get(k)].append(event.eid)
        return event

    def _insert_events(self, events):
        rows = list(map(self._convert_event_to_row, events))
        self._conn.insert_rows(self.event_table_name, rows)
        if self.mode == 'insert':
            for event in events:
                self.eids.add(event.eid)
        elif self.mode == 'cache':
            for event in events:
                self.eids.add(event.eid)
                self.eid2event_cache[event.eid] = event
                for k, v in self.partial2eids_cache.items():
                    if event.get(k) in v:
                        v[event.get(k)].append(event.eid)
        elif self.mode == 'memory':
            for event in events:
                self.eids.add(event.eid)
                self.eid2event_cache[event.eid] = event
                for k, v in self.partial2eids_cache.items():
                    if event.get(k) not in v:
                        v[event.get(k)] = [event.eid]
                    else:
                        v[event.get(k)].append(event.eid)
        return events

    def _get_event_and_store_in_cache(self, eid):
        return self._get_events_and_store_in_cache([eid])[0]

    def _get_events_and_store_in_cache(self, eids):
        events = list(map(self._convert_row_to_event, self._conn.select_rows(self.event_table_name, eids, self.event_columns)))
        for event in events:
            if event:
                self.eid2event_cache[event.eid] = event
                # It seems not to need to append
                # if self.mode == 'cache':
                #     for k, v in self.partial2eids_cache.items():
                #         if event.get(k) in v:
                #             v[event.get(k)].append(event.eid)
                # elif self.mode == 'memory':
                #     for k, v in self.partial2eids_cache.items():
                #         if event.get(k) not in v:
                #             v[event.get(k)] = [event.eid]
                #         else:
                #             v[event.get(k)].append(event.eid)
        return events

    def _update_event(self, event):
        # update db
        update_op = self._conn.get_update_op(['frequency'], "+")
        row = self._convert_event_to_row(event)
        self._conn.update_row(self.event_table_name, row, update_op, ['frequency'])

        # updata cache
        if self.mode == 'insert':
            return None  # don't care
        updated_event = self.eid2event_cache.get(event.eid, None)
        if updated_event:  # self.mode == 'memory' or hit in cache
            updated_event.frequency += event.frequency
        else:  # self.mode == 'cache' and miss in cache
            updated_event = self._get_event_and_store_in_cache(event.eid)
        return updated_event

    def _update_events(self, events):
        # update db
        update_op = self._conn.get_update_op(['frequency'], "+")
        rows = list(map(self._convert_event_to_row, events))
        self._conn.update_rows(self.event_table_name, rows, update_op, ['frequency'])

        # update cache
        if self.mode == 'insert':
            return [None] * len(events)  # don't care
        updated_events = []
        missed_indices = []
        missed_eids = []
        for idx, event in enumerate(events):
            if event.eid not in self.eids:
                updated_events.append(None)
            updated_event = self.eid2event_cache.get(event.eid, None)
            updated_events.append(updated_event)
            if updated_event:
                updated_event.frequency += event.frequency
            else:
                missed_indices.append(idx)
                missed_eids.append(event.eid)
        for idx, updated_event in enumerate(self._get_events_and_store_in_cache(missed_eids)):
            updated_events[missed_indices[idx]] = updated_event
        return updated_events

    def insert_event(self, event):
        if event.eid not in self.eids:
            return self._insert_event(event)
        else:
            return self._update_event(event)

    def insert_events(self, events):
        results = []
        new_events = []
        existing_indices = []
        existing_events = []
        for idx, event in enumerate(events):
            if event.eid not in self.eids:
                new_events.append(event)
                results.append(event)
            else:
                existing_indices.append(idx)
                existing_events.append(event)
                results.append(None)
        if len(new_events):
            self._insert_events(new_events)
        if len(existing_events):
            for idx, updated_event in enumerate(self._update_events(existing_events)):
                results[existing_indices[idx]] = updated_event
        return results

    def get_exact_match_event(self, event):
        """
        event can be Eventuality, Dictionary, str
        """
        if isinstance(event, Eventuality):
            eid = event.eid
        elif isinstance(event, dict):
            eid = event["eid"]
        elif isinstance(event, str):
            eid = event
        else:
            raise ValueError("Error: event should be an instance of Eventuality, a dictionary, or a eid.")

        if eid not in self.eids:
            return None
        exact_match_event = self.eid2event_cache.get(eid, None)
        if not exact_match_event:
            exact_match_event = self._get_event_and_store_in_cache(eid)
        return exact_match_event

    def get_exact_match_events(self, events):
        """
        events can be Eventualities, Dictionaries, strs
        """
        exact_match_events = []
        if len(events):
            if isinstance(events[0], Eventuality):
                eids = [event.eid for event in events]
            elif isinstance(events[0], dict):
                eids = [event["eid"] for event in events]
            elif isinstance(events[0], str):
                eids = events
            else:
                raise ValueError("Error: events should instances of Eventuality, dictionaries, or eids.")
            
            missed_indices = []
            missed_eids = []
            for idx, eid in enumerate(eids):
                if eid not in self.eids:
                    exact_match_events.append(None)
                exact_match_event = self.eid2event_cache.get(eid, None)
                exact_match_events.append(exact_match_event)
                if not exact_match_event:
                    missed_indices.append(idx)
                    missed_eids.append(eid)
            for idx, exact_match_event in enumerate(self._get_events_and_store_in_cache(missed_eids)):
                exact_match_events[missed_indices[idx]] = exact_match_event
        return exact_match_events

    def get_events_by_keys(self, bys, keys, order_bys=None, reverse=False, top_n=None):
        assert len(bys) == len(keys)
        for i in range(len(bys)-1, -1, -1):
            if bys[i] not in self.event_columns:
                bys.pop(i)
                keys.pop(i)
        if len(bys) == 0:
            return []
        cache = None
        by_index = -1
        for k in ["words", "skeleton_words", "verbs"]:
            if k in bys and k in self.partial2eids_cache:
                cache = self.partial2eids_cache[k]
                by_index = bys.index(k)
                break
        if cache:
            if keys[by_index] in cache:
                key_match_events = [self.eid2event_cache[eid] for eid in cache[keys[by_index]]]
            else:
                if self.mode == 'memory':
                    return []
                key_cache = []
                key_match_events = list(map(self._convert_row_to_event, 
                    self._conn.get_rows_by_keys(self.event_table_name, [bys[by_index]], [keys[by_index]], self.event_columns)))
                for key_match_event in key_match_events:
                    if key_match_event.eid not in self.eid2event_cache:
                        self.eid2event_cache[key_match_event.eid] = key_match_event
                    key_cache.append(key_match_event.eid)
                cache[keys[by_index]] = key_cache
            for i in range(len(bys)):
                if i == by_index:
                    continue
                key_match_events = list(filter(lambda x: x[bys[i]] == keys[i], key_match_events))
            if order_bys:
                key_match_events.sort(key=operator.itemgetter(*order_bys), reverse=reverse)
            if top_n:
                key_match_events = key_match_events[:top_n]
            return key_match_events
        return list(map(self._convert_row_to_event, 
            self._conn.get_rows_by_keys(self.event_table_name, bys, keys, self.event_columns, order_bys=order_bys, reverse=reverse, top_n=top_n)))

    def get_partial_match_events(self, event, bys, top_n=None, threshold=0.1, sort=True):
        """
        try to use skeleton_words to match exactly, and compute similarity between words
        if failed, try to use skeleton_words_clean to match exactly, and compute similarity between words
        if failed, try to use verbs to match exactly, and compute similarity between words
        """
        # exact match by skeleton_words, skeleton_words_clean or verbs, and compute similarity according type
        for by in bys:
            key_match_events = self.get_events_by_keys([by], [event[by]])
            if len(key_match_events) == 0:
                continue
            if not sort:
                if top_n and len(key_match_events) > top_n:
                    return random.sample(key_match_events, top_n)
                else:
                    return key_match_events
            # sort by (similarity, frequency, idx)
            queue = []
            queue_len = 0
            for idx, key_match_event in enumerate(key_match_events):
                similarity = compute_overlap(event.get(self.grain), key_match_event.get(self.grain))
                if similarity >= threshold:
                    if not top_n or queue_len < top_n:
                        heapq.heappush(queue, (similarity, key_match_event.get('frequency'), idx, key_match_event))
                        queue_len += 1
                    else:
                        heapq.heappushpop(queue, (similarity, key_match_event.get('frequency'), idx, key_match_event))
            key_match_results = []
            while len(queue) > 0:
                x = heapq.heappop(queue)
                key_match_results.append((x[0], x[-1]))
            return reversed(key_match_results)
        return []

    """
    KG (Relations)
    """
    def _convert_relation_to_row(self, relation):
        row = OrderedDict({"_id": relation.rid})
        for c in self.relation_columns[1:-len(relation_senses)]:
            row[c] = getattr(relation, c)
        for r in relation_senses:
            row[r] = relation.relations.get(r, 0.0)
        return row

    def _convert_row_to_relation(self, row):
        return Relation(row["heid"], row["teid"], {r: cnt for r, cnt in row.items() if isinstance(cnt, float) and cnt > 0.0})

    def get_relation_columns(self, columns):
        return self._conn.get_columns(self.relation_table_name, columns)

    def _insert_relation(self, relation):
        row = self._convert_relation_to_row(relation)
        self._conn.insert_row(self.relation_table_name, row)
        if self.mode == 'insert':
            self.rids.add(relation.rid)
        elif self.mode == 'cache':
            self.rids.add(relation.rid)
            self.rid2relation_cache[relation.rid] = relation
            for k, v in self.partial2rids_cache.items():
                if relation.get(k) in v:
                    v[relation.get(k)].append(relation.rid)
        elif self.mode == 'memory':
            self.rids.add(relation.rid)
            self.rid2relation_cache[relation.rid] = relation
            for k, v in self.partial2rids_cache.items():
                if relation.get(k) not in v:
                    v[relation.get(k)] = [relation.rid]
                else:
                    v[relation.get(k)].append(relation.rid)
        return relation

    def _insert_relations(self, relations):
        rows = list(map(self._convert_relation_to_row, relations))
        self._conn.insert_rows(self.relation_table_name, rows)
        if self.mode == 'insert':
            for relation in relations:
                self.rids.add(relation.rid)
        elif self.mode == 'cache':
            for relation in relations:
                self.rids.add(relation.rid)
                self.rid2relation_cache[relation.rid] = relation
                for k, v in self.partial2rids_cache.items():
                    if relation.get(k) in v:
                        v[relation.get(k)].append(relation.rid)
        elif self.mode == 'memory':
            for relation in relations:
                self.rids.add(relation.rid)
                self.rid2relation_cache[relation.rid] = relation
                for k, v in self.partial2rids_cache.items():
                    if relation.get(k) not in v:
                        v[relation.get(k)] = [relation.rid]
                    else:
                        v[relation.get(k)].append(relation.rid)
        return relations

    def _get_relation_and_store_in_cache(self, rid):
        return self._get_relations_and_store_in_cache([rid])

    def _get_relations_and_store_in_cache(self, rids):
        relations = list(map(self._convert_row_to_relation, self._conn.select_rows(self.relation_table_name, rids, self.relation_columns)))
        for relation in relations:
            if relation:
                self.rid2relation_cache[relation.rid] = relation
        return relations

    def _update_relation(self, relation):
        # find new relation frequencies
        update_columns = []
        for r in relation_senses:
            if relation.relations.get(r, 0.0) > 0.0:
                update_columns.append(r)
        
        # update db
        update_op = self._conn.get_update_op(update_columns, "+")
        row = self._convert_relation_to_row(relation)
        self._conn.update_row(self.relation_table_name, row, update_op, update_columns)
        
        # update cache
        updated_relation = self.rid2relation_cache.get(relation.rid, None)
        if updated_relation:
            for r in update_columns:
                updated_relation.relation[r] += relation.relation[r]
        else:
            updated_relation = self._get_relation_and_store_in_cache(relation.rid)
        return updated_relation

    def _update_relations(self, relations):
        # update db
        update_op = self._conn.get_update_op(relation_senses, "+")
        rows = list(map(self._convert_relation_to_row, relations))
        self._conn.update_rows(self.relation_table_name, rows, update_op, relation_senses)

        # update cache
        updated_relations = []
        missed_indices = []
        missed_rids = []
        for idx, relation in enumerate(relations):
            if relation.rid not in self.rids:
                updated_relations.append(None)
            updated_relation = self.rid2relation_cache.get(relation.rid, None)
            updated_relations.append(updated_relations)
            if updated_relation:
                for r in relation_senses:
                    if updated_relation.relations.get(r, 0.0) > 0.0:
                        updated_relation.relations[r] += relation.relations[r]
            else:
                missed_indices.append(idx)
                missed_rids.append(relation.rid)
        for idx, updated_relation in enumerate(self._get_relations_and_store_in_cache(missed_rids)):
            updated_relations[missed_indices[idx]] = updated_relation
        return updated_relations

    def insert_relation(self, relation):
        if relation.rid not in self.rid2relation_cache:
            return self._insert_relation(relation)
        else:
            return self._update_relation(relation)

    def insert_relations(self, relations):
        results = []
        new_relations = []
        existing_indices = []
        existing_relations = []
        for idx, relation in enumerate(relations):
            if relation.rid not in self.rids:
                new_relations.append(relation)
                results.append(relation)
            else:
                existing_indices.append(idx)
                existing_relations.append(relation)
                results.append(None)
        if len(new_relations):
            self._insert_relations(new_relations)
        if len(existing_relations):
            for idx, updated_relation in enumerate(self._update_relations(existing_relations)):
                results[existing_indices[idx]] = updated_relation
        return results

    def get_exact_match_relation(self, relation):
        """
        relation can be Relation, Dictionary, str, (Eventuality, Eventuality), (str, str)
        """
        if isinstance(relation, Relation):
            rid = relation.rid
        elif isinstance(relation, dict):
            rid = relation["rid"]
        elif isinstance(relation, str):
            rid = relation
        elif isinstance(relation, (tuple, list)) and len(relation) == 2:
            if isinstance(relation[0], Eventuality) and isinstance(relation[1], Eventuality):
                rid = Relation.generate_rid(relation[0].eid, relation[1].eid)
            elif isinstance(relation[0], str) and isinstance(relation[1], str):
                rid = Relation.generate_rid(relation[0], relation[1])
            else:
                raise ValueError("Error: relation should be (an instance of Eventuality, an instance of Eventuality) or (heid, teid).")
        else:
            raise ValueError("Error: relation should be an instance of Relation, a dictionary, rid, (an instance of Eventuality, an instance of Eventuality), or (heid, teid).")
        
        if rid not in self.rids:
            return None
        exact_match_relation = self.rid2relation_cache.get(rid, None)
        if not exact_match_relation:
            exact_match_relation = self._get_relation_and_store_in_cache(rid)
        return exact_match_relation

    def get_exact_match_relations(self, relations):
        """
        relations can be Relations, Dictionaries, strs, [(Eventuality, Eventuality), ...], [(str, str), ...]
        """
        exact_match_relations = []
        if len(relations):
            if isinstance(relations[0], Relation):
                rids = [relation.rid for relation in relations]
            elif isinstance(relations[0], dict):
                rids = [relation["rid"] for relation in relations]
            elif isinstance(relations[0], str):
                rids = relations
            elif isinstance(relations[0], (tuple, list)) and len(relation) == 2:
                if isinstance(relations[0][0], Eventuality) and isinstance(relations[0][1], Eventuality):
                    rids = [Relation.generate_rid(relation[0].eid, relation[1].eid) for relation in relations]
                elif isinstance(relations[0][0], str) and isinstance(relations[0][1], str):
                    rids = [Relation.generate_rid(relation[0], relation[1]) for relation in relations]
                else:
                    raise ValueError("Error: relations should be [(an instance of Eventuality, an instance of Eventuality), ...] or [(heid, teid), ...].")
            else:
                raise ValueError("Error: relations should be instances of Relation, dictionaries, rids, [(an instance of Eventuality, an instance of Eventuality), ...], or [(heid, teid), ...].")

            missed_indices = []
            missed_rids = []
            for idx, rid in enumerate(rids):
                if rid not in self.rids:
                    exact_match_relations.append(None)
                exact_match_relation = self.rid2relation_cache(rid, None)
                exact_match_relations.append(exact_match_relation)
                if not exact_match_relation:
                    missed_indices.append(idx)
                    missed_rids.append(rid)
            for idx, exact_match_relation in enumerate(self._get_relations_and_store_in_cache(missed_rids)):
                exact_match_relations[missed_indices[idx]] = exact_match_relation
        return exact_match_relations

    def get_relations_by_keys(self, bys, keys, order_bys=None, reverse=False, top_n=None):
        assert len(bys) == len(keys)
        for i in range(len(bys)-1, -1, -1):
            if bys[i] not in self.relation_columns:
                bys.pop(i)
                keys.pop(i)
        if len(bys) == 0:
            return []
        cache = None
        by_index = -1
        for k in ["heid", "teid"]:
            if k in bys and k in self.partial2rids_cache:
                cache = self.partial2rids_cache[k]
                by_index = bys.index(k)
                break
        if cache:
            if keys[by_index] in cache:
                key_match_relations = [self.rid2relation_cache[rid] for rid in cache[keys[by_index]]]
            else:
                if self.mode == 'memory':
                    return []
                key_cache = []
                key_match_relations = list(map(self._convert_row_to_relation, 
                    self._conn.get_rows_by_keys(self.relation_table_name, [bys[by_index]], [keys[by_index]], self.relation_columns)))
                for key_match_relation in key_match_relations:
                    if key_match_relation.rid not in self.rid2relation_cache:
                        self.rid2relation_cache[key_match_relation.rid] = key_match_relation
                    key_cache.append(key_match_relation.rid)
                cache[keys[by_index]] = key_cache
            for i in range(len(bys)):
                if i == by_index:
                    continue
                key_match_relations = list(filter(lambda x: x[bys[i]] == keys[i], key_match_relations))
            if order_bys:
                key_match_relations.sort(key=operator.itemgetter(*order_bys), reverse=reverse)
            if top_n:
                key_match_relations = key_match_relations[:top_n]
            return key_match_relations
        return list(map(self._convert_row_to_relation, 
            self._conn.get_rows_by_keys(self.relation_table_name, bys, keys, self.relation_columns, order_bys=order_bys, reverse=reverse, top_n=top_n)))