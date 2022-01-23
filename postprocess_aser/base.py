try:
    import ujson as json
except:
    import json
from collections import defaultdict, OrderedDict

class BaseConnection(object):
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


class SqliteConnection(BaseConnection):
    def __init__(self, db_path, chunksize):
        import sqlite3
        super(SqliteConnection, self).__init__(db_path, chunksize)
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
            insert_table = "INSERT INTO %s VALUES (%s)" % (table_name, ",".join(['?'] * (len(next(iter(rows))))))
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


class MongoDBConnection(BaseConnection):
    def __init__(self, db_path, chunksize):
        import pymongo
        super(MongoDBConnection, self).__init__(db_path, chunksize)
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