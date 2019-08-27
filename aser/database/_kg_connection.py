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
from collections import defaultdict, OrderedDict
from abc import ABCMeta, abstractclassmethod
from aser.database.util import *
from multiprocessing import Pool
from tqdm import tqdm

BATCH_SIZE = 32768
DB = 'sqlite'

_event_table_name = "Eventualities"
_event_columns = []  # need to be defined
_event_column_types = []  # need to be defined

relation_senses = [
    'Precedence', 'Succession', 'Synchronous',
    'Reason', 'Result',
    'Condition', 'Contrast', 'Concession',
    'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception',
    'Co_Occurrence']

_relation_table_name = "Relations"
_relation_columns = ['_id', 'event1_id', 'event2_id'] + relation_senses
_relation_column_types = ['PRIMARY KEY', 'TEXT',
                           'TEXT'] + ['REAL'] * len(relation_senses)

_example_table_name = "Examples"
_example_columns = ['location', 'corpus', 'event_pair_id' 'event1_id',
                     'event2_id', 'relations', 'is_double', 'flag']
_example_column_types = ['PRIMARY KEY', 'TEXT', 'TEXT',
                          'TEXT', 'TEXT', 'TEXT', 'INTEGER', 'INTEGER']


class __Connection(metaclass=ABCMeta):
    def __init__(self, db_path):
        self._conn = None

    def close(self):
        if self._conn:
            self._conn.close()

    def __del__(self):
        self.close()

    @abstractclassmethod
    def create_table(self, table_name, columns):
        raise NotImplementedError

    @abstractclassmethod
    def get_columns(self, table_name, columns):
        raise NotImplementedError

    @abstractclassmethod
    def select_row(self, _id, table_name, columns):
        raise NotImplementedError

    @abstractclassmethod
    def select_rows(self, id_list, table_name, columns):
        raise NotImplementedError

    @abstractclassmethod
    def insert_row(self, row, table_name, columns):
        raise NotImplementedError

    @abstractclassmethod
    def insert_rows(self, rows, table_name, columns):
        raise NotImplementedError

    @abstractclassmethod
    def update_row(self, row, table_name, update_op, update_columns):
        raise NotImplementedError

    @abstractclassmethod
    def update_rows(self, rows, table_name, update_ops, update_columns):
        raise NotImplementedError

    @abstractclassmethod
    def get_update_op(self, update_columns, operator):
        raise NotImplementedError

    @abstractclassmethod
    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        raise NotImplementedError


class _Sqlite_Connection(__Connection):
    def __init__(self, db_path):
        import sqlite3
        super(_Sqlite_Connection, self).__init__(db_path)
        self._conn = sqlite3.connect(db_path)

    def create_table(self, table_name, columns, columns_types):
        create_table = "CREATE TABLE %s (%s);" % (table_name, ','.join(
            [' '.join(x) for x in zip(columns, columns_types)]))
        self._conn.execute(create_table)
        self._conn.commit()

    def get_columns(self, table_name, columns):
        select_table = "SELECT %s FROM %s;" % (','.join(columns), table_name)
        result = []
        for x in self._conn.execute(select_table):
            result.append(OrderedDict(zip(columns, x)))
        return result

    def select_row(self, _id, table_name, columns):
        select_table = "SELECT %s FROM %s WHERE _id=?;" % (
            ','.join(columns), table_name)
        result = list(self._conn.execute(select_table, [_id]))
        if len(result) == 0:
            return None
        else:
            return OrderedDict(zip(columns, result[0]))

    def select_rows(self, id_list, table_name, columns):
        row_cache = dict()
        result = []
        for i in range(0, len(id_list), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(id_list))
            select_table = "SELECT %s FROM %s WHERE _id IN ('%s');" % (
                ','.join(columns), table_name, "','".join(id_list[i:j]))
            result.extend(list(self._conn.execute(select_table)))
        for x in result:
            exact_match_row = OrderedDict(zip(columns, x))
            row_cache[exact_match_row['_id']] = exact_match_row
        exact_match_rows = []
        for _id in id_list:
            exact_match_rows.append(row_cache.get(_id, None))
        return exact_match_rows

    def insert_row(self, row, table_name, columns):
        insert_table = "INSERT INTO %s VALUES (%s)" % (
            table_name, ','.join(['?'] * (len(columns))))
        self._conn.execute(insert_table, [row[k] for k in columns])
        self._conn.commit()

    def insert_rows(self, rows, table_name, columns):
        insert_table = "INSERT INTO %s VALUES (%s)" % (
            table_name, ','.join(['?'] * (len(columns))))
        self._conn.executemany(
            insert_table, [[row[k] for k in columns] for row in rows])
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
                new_update_op.append(
                    "'" + row[update_columns[i]].replace("'", "''") + "'")
            else:
                new_update_op.append(str(row[update_columns[i]]))
        return ''.join(new_update_op)

    def update_row(self, row, table_name, update_op, update_columns):
        update_table = "UPDATE %s SET %s WHERE _id=?" % (table_name, update_op)
        self._conn.execute(update_table, [row[k]
                                          for k in update_columns] + [row['_id']])
        self._conn.commit()

    def update_rows(self, rows, table_name, update_ops, update_columns):
        if isinstance(update_ops, list) or isinstance(update_ops, tuple):
            assert len(rows) == len(update_ops)
            update_op_collections = defaultdict(list)  # key: _update_update_op
            for i, row in enumerate(rows):
                # self.update_row(row, table_name, update_ops[i], update_columns)
                new_update_op = self._update_update_op(
                    row, update_ops[i], update_columns)
                update_op_collections[new_update_op].append(row)
            for k, v in update_op_collections.items():
                new_update_op = k
                id_list = [row['_id'] for row in v]
                for i in range(0, len(id_list), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(id_list))
                    update_table = "UPDATE %s SET %s WHERE _id IN ('%s');" % (
                        table_name, new_update_op, "','".join(id_list[i:j]))
                    self._conn.execute(update_table)
        else:
            update_op = update_ops
            # key: values of update_columns
            columns_collections = defaultdict(list)
            for row in rows:
                # self.update_row(row, table_name, update_op, update_columns)
                columns_collections[json.dumps(
                    [row[k] for k in update_columns])].append(row)
            for k, v in columns_collections.items():
                new_update_op = self._update_update_op(
                    v[0], update_op, update_columns)
                id_list = [row['_id'] for row in v]
                for i in range(0, len(id_list), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(id_list))
                    update_table = "UPDATE %s SET %s WHERE _id IN ('%s');" % (
                        table_name, new_update_op, "','".join(id_list[i:j]))
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
                update_table, [[row[k] for k in update_columns] + [row['_id']] for row in rows])
        self._conn.commit()
        """

    def get_update_op(self, update_columns, operator):
        if operator in '+-*/':
            update_ops = []
            for update_column in update_columns:
                update_ops.append(update_column + '=' +
                                  update_column + operator + '?')
            return ','.join(update_ops)
        elif operator == '=':
            update_ops = []
            for update_column in update_columns:
                update_ops.append(update_column + '=?')
            return ','.join(update_ops)
        else:
            raise NotImplementedError

    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        key_match_events = []
        select_table = "SELECT %s FROM %s WHERE %s" % (
            ','.join(columns), table_name, ' AND '.join(['%s=?' % (by) for by in bys]))
        if order_bys:
            select_table += ' ORDER BY %s %s' % (
                ','.join(order_bys), 'DESC' if reverse else 'ASC')
        if top_n:
            select_table += ' LIMIT %d' % (top_n)
        select_table += ';'
        for x in self._conn.execute(select_table, keys):
            key_match_event = OrderedDict(zip(columns, x))
            key_match_events.append(key_match_event)
        return key_match_events


class _MongoDB_Connection(__Connection):
    def __init__(self, db_path):
        import pymongo
        super(_MongoDB_Connection, self).__init__(db_path)
        self._client = pymongo.MongoClient("mongodb://localhost:27017/")
        self._conn = self._client[os.path.splitext(
            os.path.basename(db_path))[0]]

    def close(self):
        self._client.close()

    def create_table(self, table_name, columns, columns_types):
        self._conn[table_name]

    def get_columns(self, table_name, columns):
        projection = {'_id': 0}
        for k in columns:
            projection[k] = 1
        return list(self._conn[table_name].find({}, projection))

    def select_row(self, _id, table_name, columns):
        projection = {'_id': 0}
        for k in columns:
            projection[k] = 1
        return self._conn[table_name].find_one({'_id': _id}, projection)

    def select_rows(self, id_list, table_name, columns):
        table = self._conn[table_name]
        row_cache = dict()
        exact_match_rows = []
        projection = {'_id': 0}
        for k in columns:
            projection[k] = 1
        for i in range(0, len(id_list), BATCH_SIZE):
            j = min(i + BATCH_SIZE, len(id_list))
            query = {'_id': {'$in': id_list[i:j]}}
            exact_match_rows.extend(table.find(query, projection))
        for exact_match_row in exact_match_rows:
            row_cache[exact_match_row['_id']] = exact_match_row
        exact_match_rows = []
        for _id in id_list:
            exact_match_rows.append(row_cache.get(_id, None))
        return exact_match_rows

    def insert_row(self, row, table_name, columns):
        self._conn[table_name].insert_one(row)

    def insert_rows(self, rows, table_name, columns):
        self._conn[table_name].insert_many(rows)

    def _update_update_op(self, row, update_op, update_columns):
        new_update_op = update_op.copy()
        for k, v in new_update_op.items():
            if k == '$inc':
                for update_column in update_columns:
                    if v[update_column] == 1:
                        v[update_column] = row[update_column]
                    else:
                        v[update_column] = -row[update_column]
            elif k == '$mul':
                for update_column in update_columns:
                    if v[update_column] == 2:
                        v[update_column] = row[update_column]
                    else:
                        v[update_column] = 1.0 / row[update_column]
            elif k == '$set':
                for update_column in update_columns:
                    v[update_column] = row[update_column]
        return new_update_op

    def update_row(self, row, table_name, update_op, update_columns):
        self._conn[table_name].update_one(
            {'_id': row['_id']}, self._update_update_op(row, update_op, update_columns))

    def update_rows(self, rows, table_name, update_ops, update_columns):
        if isinstance(update_ops, list) or isinstance(update_ops, tuple):
            assert len(rows) == len(update_ops)
            update_op_collections = defaultdict(list)
            for i, row in enumerate(rows):
                # self.update_row(row, table_name, update_ops[i], update_columns)
                new_update_op = self._update_update_op(
                    row, update_ops[i], update_columns)
                update_op_collections[json.dumps(new_update_op)].append(row)
            for k, v in update_op_collections.items():
                new_update_op = json.loads(k)
                id_list = [row['_id'] for row in v]
                for i in range(0, len(id_list), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(id_list))
                    query = {'_id': {'$in': id_list[i:j]}}
                    self._conn[table_name].update_many(query, new_update_op)
        else:
            update_op = update_ops
            columns_collections = defaultdict(list)
            for row in rows:
                # self.update_row(row, table_name, update_op, update_columns)
                columns_collections[json.dumps(
                    [row[k] for k in update_columns])].append(row)
            for k, v in columns_collections.items():
                new_update_op = self._update_update_op(
                    v[0], update_op, update_columns)
                id_list = [row['_id'] for row in v]
                for i in range(0, len(id_list), BATCH_SIZE):
                    j = min(i + BATCH_SIZE, len(id_list))
                    query = {'_id': {'$in': id_list[i:j]}}
                    self._conn[table_name].update_many(query, new_update_op)

    def get_update_op(self, update_columns, operator):
        if operator == '+':
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 1  # placeholder
            return {'$inc': update_ops}
        elif operator == '-':
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = -1  # placeholder
            return {'$inc': update_ops}
        elif operator == '*':
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 2  # placeholder
            return {'$mul': update_ops}
        elif operator == '/':
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 0.5  # placeholder
            return {'$mul': update_ops}
        elif operator == '=':
            update_ops = {}
            for update_column in update_columns:
                update_ops[update_column] = 1  # placeholder
            return {'$set': update_ops}
        else:
            raise NotImplementedError

    def get_rows_by_keys(self, table_name, bys, keys, columns, order_bys=None, reverse=False, top_n=None):
        query = OrderedDict(zip(bys, keys))
        projection = dict()
        for k in columns:
            projection[k] = 1
        cursor = self._conn[table_name].find(query, projection)
        if order_bys:
            direction = -1 if reverse else 1
            cursor = cursor.sort([(k, direction) for k in order_bys])
        if top_n:
            cnt = 0
            result = []
            for x in cursor:
                result.append(x)
                cnt += 1
                if cnt >= top_n:
                    break
            return result
        else:
            return list(cursor)


class _KG_Connection(object):
    def __init__(self, db_path, mode='cache'):
        if DB == 'sqlite':
            self._conn = _Sqlite_Connection(db_path)
        elif DB == 'mongoDB':
            self._conn = _MongoDB_Connection(db_path)
        else:
            raise NotImplementedError("%s database is not supported!" % (DB))
        self.mode = mode
        if self.mode not in ['insert', 'cache', 'memory']:
            raise NotImplementedError(
                "Error: only support insert/cache/memory modes")

        self.type = None  # verbs, skeleton_words_clean, skeleton_words, words

        self.event_table_name = _event_table_name
        self.event_columns = []  # need to be defined
        self.event_column_types = []  # need to be defined
        self.relation_table_name = _relation_table_name
        self.relation_columns = _relation_columns
        self.relation_column_types = _relation_column_types

        self.event_id_set = set()
        self.relation_id_set = set()
        self.event_cache = dict()
        self.relation_cache = dict()
        # and another cache which stores event_ids
        # key: verbs, skeleton_words_clean, skeleton_words
        # value: dict which stores event_ids
        self.event_partial_cache = dict()

        # call init()
        # self.init()

    def init(self):
        """
        create tables
        load id sets
        load cache
        """
        if not self.type:
            raise NotImplementedError("Error: type must be defined")
        if len(self.event_columns) == 0 or len(self.event_column_types) == 0:
            raise NotImplementedError(
                "Error: event_columns and event_column_types must be defined")
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
            for x in self._conn.get_columns(self.event_table_name, self.event_columns):
                self.event_id_set.add(x['_id'])
                self.event_cache[x['_id']] = x
                # handle another cache
                for k, v in self.event_partial_cache.items():
                    if x[k] not in v:
                        v[x[k]] = []
                    v[x[k]].append(x['_id'])
            for x in self._conn.get_columns(self.relation_table_name, self.relation_columns):
                self.relation_id_set.add(x['_id'])
                self.relation_cache[x['_id']] = x
        else:
            for x in self._conn.get_columns(self.event_table_name, ['_id']):
                self.event_id_set.add(x['_id'])
            for x in self._conn.get_columns(self.relation_table_name, ['_id']):
                self.relation_id_set.add(x['_id'])

    def close(self):
        self._conn.close()
        self.event_id_set.clear()
        self.relation_id_set.clear()
        self.event_cache.clear()
        self.relation_cache.clear()
        # close another cache
        for k in self.event_partial_cache:
            self.event_partial_cache[k].clear()

    def get_event_columns(self, columns):
        return self._conn.get_columns(self.event_table_name, columns)

    def _insert_event(self, event):
        self._conn.insert_row(event, self.event_table_name, self.event_columns)
        if self.mode == 'insert':
            self.event_id_set.add(event['_id'])
        elif self.mode == 'cache':
            self.event_id_set.add(event['_id'])
            self.event_cache[event['_id']] = event
            for k, v in self.event_partial_cache.items():
                if event[k] in v:
                    v[event[k]].append(event['_id'])
        elif self.mode == 'memory':
            self.event_id_set.add(event['_id'])
            self.event_cache[event['_id']] = event
            for k, v in self.event_partial_cache.items():
                if event[k] not in v:
                    v[event[k]] = []
                v[event[k]].append(event['_id'])
        return event

    def _insert_events(self, events):
        self._conn.insert_rows(
            events, self.event_table_name, self.event_columns)
        if self.mode == 'insert':
            for event in events:
                self.event_id_set.add(event['_id'])
        elif self.mode == 'cache':
            for event in events:
                self.event_id_set.add(event['_id'])
                self.event_cache[event['_id']] = event
                for k, v in self.event_partial_cache.items():
                    if event[k] in v:
                        v[event[k]].append(event['_id'])
        elif self.mode == 'memory':
            for event in events:
                self.event_id_set.add(event['_id'])
                self.event_cache[event['_id']] = event
                for k, v in self.event_partial_cache.items():
                    if event[k] not in v:
                        v[event[k]] = []
                    v[event[k]].append(event['_id'])
        return events

    def __get_event_and_store_in_cache(self, event_id):
        self.__get_events_and_store_in_cache([event_id])

    def __get_events_and_store_in_cache(self, event_id_list):
        missed_id_list = []
        for _id in event_id_list:
            if _id not in self.event_id_set:
                continue
            if _id not in self.event_cache:
                missed_id_list.append(_id)
        missed_events = self._conn.select_rows(
            missed_id_list, self.event_table_name, self.event_columns)
        for missed_event in missed_events:
            if missed_event:
                self.event_cache[missed_event['_id']] = missed_event
                if self.mode == 'cache':
                    for k, v in self.event_partial_cache.items():
                        if missed_event[k] in v:
                            v[missed_event[k]].append(missed_event['_id'])
                elif self.mode == 'memory':
                    for k, v in self.event_partial_cache.items():
                        if missed_event[k] not in v:
                            v[missed_event[k]] = []
                        v[missed_event[k]].append(missed_event['_id'])

    def _update_event(self, event):
        update_op = self._conn.get_update_op(['frequency'], '+')
        self._conn.update_row(event, self.event_table_name,
                              update_op, ['frequency'])
        if self.mode == 'insert':
            return None  # don't care
        new_event = self.event_cache.get(event['_id'], None)
        if new_event:  # self.mode == 'memory' or hit in cache
            new_event['frequency'] += event['frequency']
        else:  # self.mode == 'cache' and miss in cache
            self.__get_event_and_store_in_cache(event['_id'])
            new_event = self.event_cache[event['_id']]
        return event

    def _update_events(self, events):
        update_op = self._conn.get_update_op(['frequency'], '+')
        self._conn.update_rows(events, self.event_table_name,
                               update_op, ['frequency'])
        if self.mode == 'insert':
            return []  # don't care
        missed_id_list = []
        for event in events:
            if event['_id'] not in self.event_id_set:
                continue
            new_event = self.event_cache.get(event['_id'], None)
            if new_event:
                new_event['frequency'] += event['frequency']
            else:
                missed_id_list.append(event['_id'])
        self.__get_events_and_store_in_cache(missed_id_list)
        new_events = [self.event_cache.get(
            event['_id'], None) for event in events]
        return new_events

    def insert_event(self, event):
        if event['_id'] not in self.event_id_set:
            return self._insert_event(event)
        else:
            return self._update_event(event)

    def insert_events(self, events):
        events_insert = []
        events_update = []
        for event in events:
            if event['_id'] not in self.event_id_set:
                events_insert.append(event)
            else:
                events_update.append(event)
        events = []
        if len(events_insert):
            events.extend(self._insert_events(events_insert))
        if len(events_update):
            events.extend(self._update_events(events_update))
        return events

    def get_exact_match_event(self, event):
        try:
            _id = event['_id']
        except:
            _id = event
        if _id not in self.event_id_set:
            return None
        if _id not in self.event_cache:
            self.__get_event_and_store_in_cache(_id)
        exact_match_event = self.event_cache.get(_id, None)
        return exact_match_event

    def get_exact_match_events(self, events):
        try:
            id_list = [event['_id'] for event in events]
        except:
            id_list = events
        missed_id_list = []
        for _id in id_list:
            if _id not in self.event_id_set:
                continue
            if _id not in self.event_cache:
                missed_id_list.append(_id)
        self.__get_events_and_store_in_cache(missed_id_list)
        exact_match_events = [self.event_cache.get(
            _id, None) for _id in id_list]
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
        for k in ['skeleton_words', 'skeleton_words_clean', 'verbs']:
            if k in bys and k in self.event_partial_cache:
                cache = self.event_partial_cache[k]
                by_index = bys.index(k)
                break
        if cache:
            if keys[by_index] in cache:
                result = [self.event_cache[_id]
                          for _id in cache[keys[by_index]]]
            else:
                if self.mode == 'memory':
                    return []
                key_cache = []
                result = self._conn.get_rows_by_keys(
                    self.event_table_name, [bys[by_index]], [keys[by_index]], self.event_columns)
                for key_match_event in result:
                    if key_match_event['_id'] not in self.event_cache:
                        self.event_cache[key_match_event['_id']
                                         ] = key_match_event
                    key_cache.append(key_match_event['_id'])
                cache[keys[by_index]] = key_cache
            for i in range(len(bys)):
                if i == by_index:
                    continue
                result = filter(lambda x: x[bys[i] == keys[i]], result)
            if isinstance(result, filter):
                result = list(result)
            if order_bys:
                result.sort(key=operator.itemgetter(
                    *order_bys), reverse=reverse)
            if top_n:
                result = result[:top_n]
            return result
        return self._conn.get_rows_by_keys(self.event_table_name, bys, keys, self.event_columns, order_bys=order_bys, reverse=reverse, top_n=top_n)

    def get_partial_match_events(self, event, bys=['skeleton_words', 'skeleton_words_clean', 'verbs'],
                                 top_n=None, threshold=0.1, sort=True):
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
            queue = []
            queue_len = 0
            for index, key_match_event in enumerate(key_match_events):
                similarity = compute_overlap(
                    event[self.type], key_match_event[self.type])
                if similarity >= threshold:
                    if not top_n or queue_len < top_n:
                        heapq.heappush(
                            queue, (similarity, key_match_event['frequency'], index, key_match_event))
                        queue_len += 1
                    else:
                        heapq.heappushpop(
                            queue, (similarity, key_match_event['frequency'], index, key_match_event))
            key_match_results = []
            while len(queue) > 0:
                x = heapq.heappop(queue)
                key_match_results.append((x[0], x[-1]))
            key_match_results.reverse()
            return key_match_results
        return []

    """
    KG (relations)
    """

    def get_relation_columns(self, columns):
        return self._conn.get_columns(self.relation_table_name, columns)

    def _insert_relation(self, relation):
        self._conn.insert_row(
            relation, self.relation_table_name, self.relation_columns)
        if self.mode == 'insert':
            self.relation_id_set.add(relation['_id'])
        else:
            self.relation_id_set.add(relation['_id'])
            self.relation_cache[relation['_id']] = relation
        return relation

    def _insert_relations(self, relations):
        self._conn.insert_rows(
            relations, self.relation_table_name, self.relation_columns)
        if self.mode == 'insert':
            for relation in relations:
                self.relation_id_set.add(relation['_id'])
        else:
            for relation in relations:
                self.relation_id_set.add(relation['_id'])
                self.relation_cache[relation['_id']] = relation
        return relations

    def __get_relation_and_store_in_cache(self, relation_id):
        self.__get_relations_and_store_in_cache([relation_id])

    def __get_relations_and_store_in_cache(self, relation_id_list):
        missed_id_list = []
        for _id in relation_id_list:
            if _id not in self.relation_id_set:
                continue
            if _id not in self.relation_cache:
                missed_id_list.append(_id)
        missed_relations = self._conn.select_rows(
            missed_id_list, self.relation_table_name, self.relation_columns)
        for missed_relation in missed_relations:
            if missed_relation:
                self.relation_cache[missed_relation['_id']] = missed_relation

    def _update_relation(self, relation):
        update_columns = []
        for k in relation_senses:
            if relation[k] > 0.0:
                update_columns.append(k)
        update_op = self._conn.get_update_op(update_columns, '+')
        self._conn.update_row(relation, self.relation_table_name,
                              update_op, update_columns)
        new_relation = self.relation_cache.get(relation['_id'], None)
        if new_relation:
            for k in update_columns:
                new_relation[k] += relation[k]
        else:
            self.__get_relation_and_store_in_cache(relation['_id'])
            new_relation = self.relation_cache.get(relation['_id'], None)
        return relation

    def _update_relations(self, relations):
        update_op = self._conn.get_update_op(relation_senses, '+')
        self._conn.update_rows(
            relations, self.relation_table_name, update_op, relation_senses)
        missed_id_list = []
        for relation in relations:
            if relation['_id'] not in self.relation_id_set:
                continue
            new_relation = self.relation_cache.get(relation['_id'], None)
            if new_relation:
                for k in relation_senses:
                    if relation[k] > 0.0:
                        new_relation[k] += relation[k]
            else:
                missed_id_list.append(relation['_id'])
        self.__get_relations_and_store_in_cache(missed_id_list)
        new_relations = [self.relation_cache.get(
            relation['_id'], None) for relation in relations]
        return new_relations

    def insert_relation(self, relation):
        if relation['_id'] not in self.relation_cache:
            return self._insert_relation(relation)
        else:
            return self._update_relation(relation)

    def insert_relations(self, relations):
        relations_insert = []
        relations_update = []
        for relation in relations:
            if relation['_id'] in self.relation_id_set:
                relations_update.append(relation)
            else:
                relations_insert.append(relation)
        relations = []
        # insert directly
        if len(relations_insert):
            relations.extend(self._insert_relations(relations_insert))
        if len(relations_update):
            relations.extend(self._update_relations(relations_update))
        return relations

    def get_exact_match_relation(self, relation):
        """
        relation can be (e1, e2), relation_id, or relation
        """
        if (isinstance(relation, tuple) or isinstance(relation, list)) and len(relation) == 2:
            try:
                event1, event2 = relation
                _id = generate_id(event1['_id'] + '$' + event2['_id'])
            except:
                event1_id, event2_id = relation
                _id = generate_id(event1_id + '$' + event2_id)
        else:
            try:
                _id = relation['_id']
            except:
                _id = relation
        if _id not in self.relation_id_set:
            return None
        if _id not in self.relation_cache:
            self.__get_relation_and_store_in_cache(_id)
        exact_match_relation = self.relation_cache.get(_id, None)
        return exact_match_relation

    def get_exact_match_relations(self, relations):
        """
        relations can be [(e1, e2), ...], [relation_id, ...], or [relation, ...]
        """
        if len(relations) == 0:
            return []
        if (isinstance(relations[0], tuple) or isinstance(relations[0], list)) and len(relations[0]) == 2:
            id_list = []
            try:
                for relation in relations:
                    event1, event2 = relation
                    id_list.append(generate_id(
                        event1['_id'] + '$' + event2['_id']))
            except:
                for relation in relations:
                    event1_id, event2_id = relation
                    id_list.append(generate_id(event1_id + '$' + event2_id))
        else:
            try:
                id_list = [relation['_id'] for relation in relations]
            except:
                id_list = relations
        missed_id_list = []
        for _id in id_list:
            if _id not in self.relation_id_set:
                continue
            if _id not in self.relation_cache:
                missed_id_list.append(_id)
        self.__get_relations_and_store_in_cache(missed_id_list)
        exact_match_relations = [
            self.relation_cache.get(_id, None) for _id in id_list]
        return exact_match_relations

    def get_relations_by_keys(self, bys, keys, order_bys=None, reverse=False, top_n=None):
        assert len(bys) == len(keys)
        return self._conn.get_rows_by_keys(self.relation_table_name, bys, keys, self.relation_columns, order_bys=order_bys, reverse=reverse, top_n=top_n)