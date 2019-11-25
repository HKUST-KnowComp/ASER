try:
    import ujson as json
except:
    import json
import os
import random
import heapq
import operator
from copy import copy
from functools import partial
from collections import defaultdict, OrderedDict
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.database.base import SqliteConnection, MongoDBConnection
from aser.database.util import *

CHUNKSIZE = 32768
EVENTUALITY_TABLE_NAME = "Eventualities"
EVENTUALITY_COLUMNS = ["_id", "frequency", "pattern", "verbs", "skeleton_words", "words", "info"]
EVENTUALITY_COLUMN_TYPES = ["PRIMARY KEY", "REAL", "TEXT", "TEXT", "TEXT", "TEXT", "BLOB"]
RELATION_TABLE_NAME = "Relations"
RELATION_COLUMNS = ["_id", "hid", "tid"] + relation_senses
RELATION_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"] + ["REAL"] * len(relation_senses)

class ASERKGConnection(object):
    # TODO: use iterator to retrieve eventualities and relations in case of out of memory
    def __init__(self, db_path, db="sqlite", mode="cache", grain=None, chunksize=-1):
        if db == "sqlite":
            self._conn = SqliteConnection(db_path, chunksize if chunksize > 0 else CHUNKSIZE)
        elif db == "mongoDB":
            self._conn = MongoDBConnection(db_path, chunksize if chunksize > 0 else CHUNKSIZE)
        else:
            raise NotImplementedError("Error: %s database is not supported!" % (db))
        self.mode = mode
        if self.mode not in ["insert", "cache", "memory"]:
            raise NotImplementedError(
                "only support insert/cache/memory modes.")

        if grain not in [None, "verbs", "skeleton_words", "words"]:
            raise NotImplementedError("Error: only support None/verbs/skeleton_words/words grain.")
        self.grain = grain  # None, verbs, skeleton_words, words

        self.eventuality_table_name = EVENTUALITY_TABLE_NAME
        self.eventuality_columns = EVENTUALITY_COLUMNS
        self.eventuality_column_types = EVENTUALITY_COLUMN_TYPES
        self.relation_table_name = RELATION_TABLE_NAME
        self.relation_columns = RELATION_COLUMNS
        self.relation_column_types = RELATION_COLUMN_TYPES

        self.eids = set()
        self.rids = set()
        self.eid2eventuality_cache = dict()
        self.rid2relation_cache = dict()        
        if self.grain == "words":
            self.partial2eids_cache = {"verbs": dict(), "skeleton_words": dict(), "words": dict()}
        elif self.grain == "skeleton_words":
            self.partial2eids_cache = {"verbs": dict(), "skeleton_words": dict()}
        elif self.grain == "verbs":
            self.partial2eids_cache = {"verbs": dict()}
        else:
            self.partial2eids_cache = dict()
        self.partial2rids_cache = {"hid": dict()}

        self.init()

    def init(self):
        """
        create tables
        load id sets
        load cache
        """
        for table_name, columns, column_types in zip(
            [self.eventuality_table_name, self.relation_table_name], 
            [self.eventuality_columns, self.relation_columns],
            [self.eventuality_column_types, self.relation_column_types]):
            if len(columns) == 0 or len(column_types) == 0:
                raise NotImplementedError("Error: %s_columns and %s_column_types must be defined" % (table_name, table_name))
            try:
                self._conn.create_table(table_name, columns, column_types)
            except:
                pass

        if self.mode == "memory":
            for e in map(self._convert_row_to_eventuality, self._conn.get_columns(self.eventuality_table_name, self.eventuality_columns)):
                self.eids.add(e.eid)
                self.eid2eventuality_cache[e.eid] = e
                # handle another cache
                for k, v in self.partial2eids_cache.items():
                    if getattr(e, k) not in v:
                        v[getattr(e, k)] = [e.eid]
                    else:
                        v[getattr(e, k)].append(e.eid)
            for r in map(self._convert_row_to_relation, self._conn.get_columns(self.relation_table_name, self.relation_columns)):
                self.rids.add(r.rid)
                self.rid2relation_cache[r.rid] = r
                # handle another cache
                for k, v in self.partial2rids_cache.items():
                    if getattr(r, k) not in v:
                        v[getattr(r, k)] = [r.rid]
                    else:
                        v[getattr(r, k)].append(r.rid)
        else:
            for e in self._conn.get_columns(self.eventuality_table_name, ["_id"]):
                self.eids.add(e["_id"])
            for r in self._conn.get_columns(self.relation_table_name, ["_id"]):
                self.rids.add(r["_id"])

    def close(self):
        self._conn.close()
        self.eids.clear()
        self.rids.clear()
        self.eid2eventuality_cache.clear()
        self.rid2relation_cache.clear()
        # close another cache
        for k in self.partial2eids_cache:
            self.partial2eids_cache[k].clear()
        for k in self.partial2rids_cache:
            self.partial2rids_cache[k].clear()

    """
    KG (Eventualities)
    """
    def _convert_eventuality_to_row(self, eventuality):
        row = OrderedDict({"_id": eventuality.eid})
        for c in self.eventuality_columns[1:-1]:
            d = getattr(eventuality, c)
            if isinstance(d, list):
                row[c] = " ".join(d)
            else:
                row[c] = d
        row["info"] = json.dumps(eventuality.to_dict(minimum=True)).encode("utf-8")
        return row
    
    def _convert_row_to_eventuality(self, row):
        eventuality = Eventuality().from_dict(json.loads(row["info"].decode("utf-8")))
        eventuality.eid = row["_id"]
        eventuality.frequency = row["frequency"]
        eventuality.pattern = row["pattern"]
        return eventuality

    def get_eventuality_columns(self, columns):
        return self._conn.get_columns(self.eventuality_table_name, columns)

    def _insert_eventuality(self, eventuality):
        row = self._convert_eventuality_to_row(eventuality)
        self._conn.insert_row(self.eventuality_table_name, row)
        if self.mode == "insert":
            self.eids.add(eventuality.eid)
        elif self.mode == "cache":
            self.eids.add(eventuality.eid)
            self.eid2eventuality_cache[eventuality.eid] = eventuality
            for k, v in self.partial2eids_cache.items():
                if eventuality.get(k) in v:
                    v[eventuality.get(k)].append(eventuality.eid)
        elif self.mode == "memory":
            self.eids.add(eventuality.eid)
            self.eid2eventuality_cache[eventuality.eid] = eventuality
            for k, v in self.partial2eids_cache.items():
                if eventuality.get(k) not in v:
                    v[eventuality.get(k)] = [eventuality.eid]
                else:
                    v[eventuality.get(k)].append(eventuality.eid)
        return eventuality

    def _insert_eventualities(self, eventualities):
        rows = list(map(self._convert_eventuality_to_row, eventualities))
        self._conn.insert_rows(self.eventuality_table_name, rows)
        if self.mode == "insert":
            for eventuality in eventualities:
                self.eids.add(eventuality.eid)
        elif self.mode == "cache":
            for eventuality in eventualities:
                self.eids.add(eventuality.eid)
                self.eid2eventuality_cache[eventuality.eid] = eventuality
                for k, v in self.partial2eids_cache.items():
                    if eventuality.get(k) in v:
                        v[eventuality.get(k)].append(eventuality.eid)
        elif self.mode == "memory":
            for eventuality in eventualities:
                self.eids.add(eventuality.eid)
                self.eid2eventuality_cache[eventuality.eid] = eventuality
                for k, v in self.partial2eids_cache.items():
                    if eventuality.get(k) not in v:
                        v[eventuality.get(k)] = [eventuality.eid]
                    else:
                        v[eventuality.get(k)].append(eventuality.eid)
        return eventualities

    def _get_eventuality_and_store_in_cache(self, eid):
        return self._get_eventualities_and_store_in_cache([eid])[0]

    def _get_eventualities_and_store_in_cache(self, eids):
        eventualities = list(map(self._convert_row_to_eventuality, self._conn.select_rows(self.eventuality_table_name, eids, self.eventuality_columns)))
        for eventuality in eventualities:
            if eventuality:
                self.eid2eventuality_cache[eventuality.eid] = eventuality
                # It seems not to need to append
                # if self.mode == "cache":
                #     for k, v in self.partial2eids_cache.items():
                #         if eventuality.get(k) in v:
                #             v[eventuality.get(k)].append(eventuality.eid)
                # elif self.mode == "memory":
                #     for k, v in self.partial2eids_cache.items():
                #         if eventuality.get(k) not in v:
                #             v[eventuality.get(k)] = [eventuality.eid]
                #         else:
                #             v[eventuality.get(k)].append(eventuality.eid)
        return eventualities

    def _update_eventuality(self, eventuality):
        # update db
        update_op = self._conn.get_update_op(["frequency"], "+")
        row = self._convert_eventuality_to_row(eventuality)
        self._conn.update_row(self.eventuality_table_name, row, update_op, ["frequency"])

        # updata cache
        if self.mode == "insert":
            return None  # don"t care
        updated_eventuality = self.eid2eventuality_cache.get(eventuality.eid, None)
        if updated_eventuality:  # self.mode == "memory" or hit in cache
            updated_eventuality.frequency += eventuality.frequency
        else:  # self.mode == "cache" and miss in cache
            updated_eventuality = self._get_eventuality_and_store_in_cache(eventuality.eid)
        return updated_eventuality

    def _update_eventualities(self, eventualities):
        # update db
        update_op = self._conn.get_update_op(["frequency"], "+")
        rows = list(map(self._convert_eventuality_to_row, eventualities))
        self._conn.update_rows(self.eventuality_table_name, rows, update_op, ["frequency"])

        # update cache
        if self.mode == "insert":
            return [None] * len(eventualities)  # don"t care
        updated_eventualities = []
        missed_indices = []
        missed_eids = []
        for idx, eventuality in enumerate(eventualities):
            if eventuality.eid not in self.eids:
                updated_eventualities.append(None)
            updated_eventuality = self.eid2eventuality_cache.get(eventuality.eid, None)
            updated_eventualities.append(updated_eventuality)
            if updated_eventuality:
                updated_eventuality.frequency += eventuality.frequency
            else:
                missed_indices.append(idx)
                missed_eids.append(eventuality.eid)
        for idx, updated_eventuality in enumerate(self._get_eventualities_and_store_in_cache(missed_eids)):
            updated_eventualities[missed_indices[idx]] = updated_eventuality
        return updated_eventualities

    def insert_eventuality(self, eventuality):
        if eventuality.eid not in self.eids:
            return self._insert_eventuality(eventuality)
        else:
            return self._update_eventuality(eventuality)

    def insert_eventualities(self, eventualities):
        results = []
        new_eventualities = []
        existing_indices = []
        existing_eventualities = []
        for idx, eventuality in enumerate(eventualities):
            if eventuality.eid not in self.eids:
                new_eventualities.append(eventuality)
                results.append(eventuality)
            else:
                existing_indices.append(idx)
                existing_eventualities.append(eventuality)
                results.append(None)
        if len(new_eventualities):
            self._insert_eventualities(new_eventualities)
        if len(existing_eventualities):
            for idx, updated_eventuality in enumerate(self._update_eventualities(existing_eventualities)):
                results[existing_indices[idx]] = updated_eventuality
        return results

    def get_exact_match_eventuality(self, eventuality):
        """
        eventuality can be Eventuality, Dictionary, str
        """
        if isinstance(eventuality, Eventuality):
            eid = eventuality.eid
        elif isinstance(eventuality, dict):
            eid = eventuality["eid"]
        elif isinstance(eventuality, str):
            eid = eventuality
        else:
            raise ValueError("Error: eventuality should be an instance of Eventuality, a dictionary, or a eid.")

        if eid not in self.eids:
            return None
        exact_match_eventuality = self.eid2eventuality_cache.get(eid, None)
        if not exact_match_eventuality:
            exact_match_eventuality = self._get_eventuality_and_store_in_cache(eid)
        return exact_match_eventuality

    def get_exact_match_eventualities(self, eventualities):
        """
        eventualities can be Eventualities, Dictionaries, strs
        """
        exact_match_eventualities = []
        if len(eventualities):
            if isinstance(eventualities[0], Eventuality):
                eids = [eventuality.eid for eventuality in eventualities]
            elif isinstance(eventualities[0], dict):
                eids = [eventuality["eid"] for eventuality in eventualities]
            elif isinstance(eventualities[0], str):
                eids = eventualities
            else:
                raise ValueError("Error: eventualities should instances of Eventuality, dictionaries, or eids.")
            
            missed_indices = []
            missed_eids = []
            for idx, eid in enumerate(eids):
                if eid not in self.eids:
                    exact_match_eventualities.append(None)
                exact_match_eventuality = self.eid2eventuality_cache.get(eid, None)
                exact_match_eventualities.append(exact_match_eventuality)
                if not exact_match_eventuality:
                    missed_indices.append(idx)
                    missed_eids.append(eid)
            for idx, exact_match_eventuality in enumerate(self._get_eventualities_and_store_in_cache(missed_eids)):
                exact_match_eventualities[missed_indices[idx]] = exact_match_eventuality
        return exact_match_eventualities

    def get_eventualities_by_keys(self, bys, keys, order_bys=None, reverse=False, top_n=None):
        assert len(bys) == len(keys)
        for i in range(len(bys)-1, -1, -1):
            if bys[i] not in self.eventuality_columns:
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
                key_match_eventualities = [self.eid2eventuality_cache[eid] for eid in cache[keys[by_index]]]
            else:
                if self.mode == "memory":
                    return []
                key_cache = []
                key_match_eventualities = list(map(self._convert_row_to_eventuality, 
                    self._conn.get_rows_by_keys(self.eventuality_table_name, [bys[by_index]], [keys[by_index]], self.eventuality_columns)))
                for key_match_eventuality in key_match_eventualities:
                    if key_match_eventuality.eid not in self.eid2eventuality_cache:
                        self.eid2eventuality_cache[key_match_eventuality.eid] = key_match_eventuality
                    key_cache.append(key_match_eventuality.eid)
                cache[keys[by_index]] = key_cache
            for i in range(len(bys)):
                if i == by_index:
                    continue
                key_match_eventualities = list(filter(lambda x: x[bys[i]] == keys[i], key_match_eventualities))
            if order_bys:
                key_match_eventualities.sort(key=operator.itemgetter(*order_bys), reverse=reverse)
            if top_n:
                key_match_eventualities = key_match_eventualities[:top_n]
            return key_match_eventualities
        return list(map(self._convert_row_to_eventuality, 
            self._conn.get_rows_by_keys(self.eventuality_table_name, bys, keys, self.eventuality_columns, order_bys=order_bys, reverse=reverse, top_n=top_n)))

    def get_partial_match_eventualities(self, eventuality, bys, top_n=None, threshold=0.1, sort=True):
        """
        try to use skeleton_words to match exactly, and compute similarity between words
        if failed, try to use skeleton_words_clean to match exactly, and compute similarity between words
        if failed, try to use verbs to match exactly, and compute similarity between words
        """
        # exact match by skeleton_words, skeleton_words_clean or verbs, and compute similarity according type
        for by in bys:
            key_match_eventualities = self.get_eventualities_by_keys([by], [eventuality[by]])
            if len(key_match_eventualities) == 0:
                continue
            if not sort:
                if top_n and len(key_match_eventualities) > top_n:
                    return random.sample(key_match_eventualities, top_n)
                else:
                    return key_match_eventualities
            # sort by (similarity, frequency, idx)
            queue = []
            queue_len = 0
            for idx, key_match_eventuality in enumerate(key_match_eventualities):
                similarity = compute_overlap(eventuality.get(self.grain), key_match_eventuality.get(self.grain))
                if similarity >= threshold:
                    if not top_n or queue_len < top_n:
                        heapq.heappush(queue, (similarity, key_match_eventuality.get("frequency"), idx, key_match_eventuality))
                        queue_len += 1
                    else:
                        heapq.heappushpop(queue, (similarity, key_match_eventuality.get("frequency"), idx, key_match_eventuality))
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
        return Relation(row["hid"], row["tid"], {r: cnt for r, cnt in row.items() if isinstance(cnt, float) and cnt > 0.0})

    def get_relation_columns(self, columns):
        return self._conn.get_columns(self.relation_table_name, columns)

    def _insert_relation(self, relation):
        row = self._convert_relation_to_row(relation)
        self._conn.insert_row(self.relation_table_name, row)
        if self.mode == "insert":
            self.rids.add(relation.rid)
        elif self.mode == "cache":
            self.rids.add(relation.rid)
            self.rid2relation_cache[relation.rid] = relation
            for k, v in self.partial2rids_cache.items():
                if relation.get(k) in v:
                    v[relation.get(k)].append(relation.rid)
        elif self.mode == "memory":
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
        if self.mode == "insert":
            for relation in relations:
                self.rids.add(relation.rid)
        elif self.mode == "cache":
            for relation in relations:
                self.rids.add(relation.rid)
                self.rid2relation_cache[relation.rid] = relation
                for k, v in self.partial2rids_cache.items():
                    if relation.get(k) in v:
                        v[relation.get(k)].append(relation.rid)
        elif self.mode == "memory":
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
                raise ValueError("Error: relation should be (an instance of Eventuality, an instance of Eventuality) or (hid, tid).")
        else:
            raise ValueError("Error: relation should be an instance of Relation, a dictionary, rid, (an instance of Eventuality, an instance of Eventuality), or (hid, tid).")
        
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
                    raise ValueError("Error: relations should be [(an instance of Eventuality, an instance of Eventuality), ...] or [(hid, tid), ...].")
            else:
                raise ValueError("Error: relations should be instances of Relation, dictionaries, rids, [(an instance of Eventuality, an instance of Eventuality), ...], or [(hid, tid), ...].")

            missed_indices = []
            missed_rids = []
            for idx, rid in enumerate(rids):
                if rid not in self.rids:
                    exact_match_relations.append(None)
                exact_match_relation = self.rid2relation_cache.get(rid, None)
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
        for k in ["hid", "tid"]:
            if k in bys and k in self.partial2rids_cache:
                cache = self.partial2rids_cache[k]
                by_index = bys.index(k)
                break
        if cache:
            if keys[by_index] in cache:
                key_match_relations = [self.rid2relation_cache[rid] for rid in cache[keys[by_index]]]
            else:
                if self.mode == "memory":
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

    """
    Addtional apis
    """
    def get_related_eventualities(self, eventuality):
        """
        eventuality can be Eventuality, Dictionary, str
        """
        if self.mode == "insert":
            return []
        if isinstance(eventuality, Eventuality):
            eid = eventuality.eid
        if isinstance(eventuality, dict):
            eid = eventuality["eid"]
        elif isinstance(eventuality, str):
            eid = eventuality
        else:
            raise ValueError("Error: eventuality should a instance of Eventuality, or eid.")
        
        # eid == hid
        if self.mode == "memory":
            if "hid" in self.partial2rids_cache:
                related_rids = self.partial2rids_cache["hid"].get(eid, list())
                related_relations = self.get_exact_match_relations(related_rids)
            else:
                related_relations = self.get_relations_by_keys(bys=["hid"], keys=[eid])
            tids = [x.tid for x in related_relations]
            t_eventualities = self.get_exact_match_eventualities(tids)
        elif self.mode == "cache":
            if "hid" in self.partial2rids_cache:
                if eid in self.partial2rids_cache["hid"]: # hit
                    related_rids = self.partial2rids_cache["hid"].get(eid, list())
                    related_relations = self.get_exact_match_relations(related_rids)
                    tids = [x.tid for x in related_relations]
                    t_eventualities = self.get_exact_match_eventualities(tids)
                else: # miss
                    related_relations = self.get_relations_by_keys(bys=["hid"], keys=[eid])
                    tids = [x.tid for x in related_relations]
                    t_eventualities = self.get_exact_match_eventualities(tids)
                    # update cache
                    self.partial2rids_cache["hid"][eid] = [relation.rid for relation in related_relations]
            else:
                related_relations = self.get_relations_by_keys(bys=["hid"], keys=[eid])
                tids = [x.tid for x in related_relations]
                t_eventualities = self.get_exact_match_eventualities(tids)
        
        return sorted(zip(t_eventualities, related_relations), key=lambda x: sum(x[1].relations.values()))