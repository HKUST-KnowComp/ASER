import heapq
import operator
import random
from collections import OrderedDict
from ..concept import ASERConcept, ASERConceptInstancePair
from ..eventuality import Eventuality
from ..relation import Relation, relation_senses
from ..database.db_connection import SqliteDBConnection, MongoDBConnection
from ..database.utils import compute_overlap

CHUNKSIZE = 32768

EVENTUALITY_TABLE_NAME = "Eventualities"
EVENTUALITY_COLUMNS = ["_id", "frequency", "pattern", "verbs", "skeleton_words", "words", "info"]
EVENTUALITY_COLUMN_TYPES = ["PRIMARY KEY", "REAL", "TEXT", "TEXT", "TEXT", "TEXT", "BLOB"]

CONCEPT_TABLE_NAME = "Concepts"
CONCEPT_COLUMNS = ["_id", "pattern", "info"]
CONCEPT_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "BLOB"]

RELATION_TABLE_NAME = "Relations"
RELATION_COLUMNS = ["_id", "hid", "tid"] + relation_senses
RELATION_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"] + ["REAL"] * len(relation_senses)

CONCEPTINSTANCEPAIR_TABLE_NAME = "ConceptInstancePairs"
CONCEPTINSTANCEPAIR_COLUMNS = ["_id", "cid", "eid", "pattern", "score"]
CONCEPTINSTANCEPAIR_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT", "TEXT", "REAL"]


class ASERKGConnection(object):
    """ KG connection for ASER (including eventualities and relations)

    """
    def __init__(self, db_path, db="sqlite", mode="cache", grain=None, chunksize=CHUNKSIZE):
        """

        :param db_path: database path
        :type db_path: str
        :param db: the backend database, e.g., "sqlite" or "mongodb"
        :type db: str (default = "sqlite")
        :param mode: the mode to use the connection.
            "insert": this connection is only used to insert/update rows;
            "cache": this connection caches some contents that have been retrieved;
            "memory": this connection loads all contents in memory;
        :type mode: str (default = "cache")
        :param grain: the grain to build cache
            "words": cache is built on "verbs", "skeleton_words", and "words"
            "skeleton_words": cache is built on "verbs", and "skeleton_words"
            "verbs": cache is built on "verbs"
            None: no cache
        :type grain: Union[str, None] (default = None)
        :param chunksize: the chunksize to load/write database
        :type chunksize: int (default = 32768)
        """

        if db == "sqlite":
            self._conn = SqliteDBConnection(db_path, chunksize)
        elif db == "mongodb":
            self._conn = MongoDBConnection(db_path, chunksize)
        else:
            raise ValueError("Error: %s database is not supported!" % (db))
        self.mode = mode
        if self.mode not in ["insert", "cache", "memory"]:
            raise ValueError("only support insert/cache/memory modes.")

        if grain not in [None, "verbs", "skeleton_words", "words"]:
            raise ValueError("Error: only support None/verbs/skeleton_words/words grain.")
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
        """ Initialize the ASERKGConnection, including creating tables, loading eids and rids, and building cache

        """
        for table_name, columns, column_types in zip(
            [self.eventuality_table_name, self.relation_table_name], [self.eventuality_columns, self.relation_columns],
            [self.eventuality_column_types, self.relation_column_types]
        ):
            if len(columns) == 0 or len(column_types) == 0:
                raise ValueError("Error: %s_columns and %s_column_types must be defined" % (table_name, table_name))
            try:
                self._conn.create_table(table_name, columns, column_types)
            except:
                pass

        if self.mode == "memory":
            for e in map(
                self._convert_row_to_eventuality,
                self._conn.get_columns(self.eventuality_table_name, self.eventuality_columns)
            ):
                self.eids.add(e.eid)
                self.eid2eventuality_cache[e.eid] = e
                # handle another cache
                for k, v in self.partial2eids_cache.items():
                    if " ".join(getattr(e, k)) not in v:
                        v[" ".join(getattr(e, k))] = [e.eid]
                    else:
                        v[" ".join(getattr(e, k))].append(e.eid)
            for r in map(
                self._convert_row_to_relation, self._conn.get_columns(self.relation_table_name, self.relation_columns)
            ):
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
        """ Close the ASERKGConnection safely

        """

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
        row["info"] = eventuality.encode(minimum=True)
        return row

    def _convert_row_to_eventuality(self, row):
        eventuality = Eventuality().decode(row["info"])
        eventuality.eid = row["_id"]
        eventuality.frequency = row["frequency"]
        eventuality.pattern = row["pattern"]
        return eventuality

    def get_eventuality_columns(self, columns):
        """ Get column information from eventualities

        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
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
        eventualities = list(
            map(
                self._convert_row_to_eventuality,
                self._conn.select_rows(self.eventuality_table_name, eids, self.eventuality_columns)
            )
        )
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
            else:
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
        """ Insert/Update an eventuality into ASER
        (suggestion: consider to use `insert_eventualities` if you want to insert multiple eventualities)

        :param eventuality: an eventuality to insert/update
        :type eventuality: aser.eventuality.Eventuality
        :return: the inserted/updated eventuality
        :rtype: aser.eventuality.Eventuality
        """

        if eventuality.eid not in self.eids:
            return self._insert_eventuality(eventuality)
        else:
            return self._update_eventuality(eventuality)

    def insert_eventualities(self, eventualities):
        """ Insert/Update eventualities into ASER

        :param eventualities: eventualities to insert/update
        :type eventualities: List[aser.eventuality.Eventuality]
        :return: the inserted/updated eventualities
        :rtype: List[aser.eventuality.Eventuality]
        """

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
        """ Retrieve an exact matched eventuality from ASER
        (suggestion: consider to use `get_exact_match_eventualities` if you want to retrieve multiple eventualities)

        :param eventuality: an eventuality that contains the eid
        :type eventuality: Union[aser.eventuality.Eventuality, Dict[str, object], str]
        :return: the exact matched eventuality
        :rtype: aser.eventuality.Eventuality
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
        """ Retrieve multiple exact matched eventualities from ASER

        :param eventualities: eventualities
        :type eventualities: Union[List[aser.eventuality.Eventuality], List[Dict[str, object]], List[str]]
        :return: the exact matched eventualities
        :rtype: List[aser.eventuality.Eventuality]
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
        """ Retrieve multiple partial matched eventualities by keys and values from ASER

        :param bys: the given columns to match
        :type bys: List[str]
        :param keys: the given values to match
        :type keys: List[str]
        :param order_bys: the columns whose value are used to sort rows
        :type order_bys: List[str]
        :param reverse: whether to sort in a reversed order
        :type reverse: bool
        :param top_n: how many eventualities to return, default `None` for all eventualities
        :type top_n: int
        :return: the partial matched eventualities
        :rtype: List[aser.eventuality.Eventuality]
        """

        assert len(bys) == len(keys)
        for i in range(len(bys) - 1, -1, -1):
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
                key_match_eventualities = list(
                    map(
                        self._convert_row_to_eventuality,
                        self._conn.get_rows_by_keys(
                            self.eventuality_table_name, [bys[by_index]], [keys[by_index]], self.eventuality_columns
                        )
                    )
                )
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
        return list(
            map(
                self._convert_row_to_eventuality,
                self._conn.get_rows_by_keys(
                    self.eventuality_table_name,
                    bys,
                    keys,
                    self.eventuality_columns,
                    order_bys=order_bys,
                    reverse=reverse,
                    top_n=top_n
                )
            )
        )

    def get_partial_match_eventualities(self, eventuality, bys, top_n=None, threshold=0.8, sort=True):
        """ Retrieve multiple partial matched eventualities by a given eventuality and properties from ASER

        :param eventuality: the given eventuality to match
        :type eventuality: aser.eventuality.Eventuality
        :param bys: the given properties to match
        :type bys: List[str]
        :param top_n: how many rows to return, default `None` for all rows
        :type top_n: int
        :param threshold: the minimum similarity
        :type threshold: float (default = 0.8)
        :param sort: whether to sort
        :type sort: bool (default = True)
        :return: the partial matched eventualities
        :rtype: List[aser.eventuality.Eventuality]
        """

        assert self.grain is not None
        # exact match by skeleton_words, skeleton_words_clean or verbs, and compute similarity according type
        for by in bys:
            key_match_eventualities = self.get_eventualities_by_keys([by], [" ".join(getattr(eventuality, by))])
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
                similarity = compute_overlap(
                    getattr(eventuality, self.grain), getattr(key_match_eventuality, self.grain)
                )
                if similarity >= threshold:
                    if not top_n or queue_len < top_n:
                        heapq.heappush(queue, (similarity, key_match_eventuality.frequency, idx, key_match_eventuality))
                        queue_len += 1
                    else:
                        heapq.heappushpop(
                            queue, (similarity, key_match_eventuality.frequency, idx, key_match_eventuality)
                        )
            key_match_results = []
            while len(queue) > 0:
                x = heapq.heappop(queue)
                key_match_results.append((x[0], x[-1]))
            key_match_results.reverse()
            return key_match_results
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
        return Relation(
            row["hid"], row["tid"], {r: cnt
                                     for r, cnt in row.items() if isinstance(cnt, float) and cnt > 0.0}
        )

    def get_relation_columns(self, columns):
        """ Get column information from relations

        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
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
        return self._get_relations_and_store_in_cache([rid])[0]

    def _get_relations_and_store_in_cache(self, rids):
        relations = list(
            map(
                self._convert_row_to_relation,
                self._conn.select_rows(self.relation_table_name, rids, self.relation_columns)
            )
        )
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
            else:
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
        """ Insert/Update a relation into ASER
        (suggestion: consider to use `insert_relations` if you want to insert multiple relations)

        :param relation: a relation to insert/update
        :type relation: aser.relation.Relation
        :return: the inserted/updated relation
        :rtype: aser.relation.Relation
        """

        if relation.rid not in self.rid2relation_cache:
            return self._insert_relation(relation)
        else:
            return self._update_relation(relation)

    def insert_relations(self, relations):
        """ Insert/Update relations into ASER

        :param relations: relations to insert/update
        :type relations: List[aser.relation.Relation]
        :return: the inserted/updated relations
        :rtype: List[aser.relation.Relation]
        """

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
        """ Retrieve an exact matched relation from ASER
        (suggestion: consider to use `get_exact_match_relations` if you want to retrieve multiple relations)

        :param relation: a relation that contains the rid or an eventuality pair that contains two eids
        :type relation: Union[aser.relation.Relation, Dict[str, object], str, Tuple[aser.eventuality.Eventuality, aser.eventuality.Eventuality], Tuple[str, str]]
        :return: the exact matched relation
        :rtype: aser.relation.Relation
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
                raise ValueError(
                    "Error: relation should be (an instance of Eventuality, an instance of Eventuality) or (hid, tid)."
                )
        else:
            raise ValueError(
                "Error: relation should be an instance of Relation, a dictionary, rid,"
                "(an instance of Eventuality, an instance of Eventuality), or (hid, tid)."
            )

        if rid not in self.rids:
            return None
        exact_match_relation = self.rid2relation_cache.get(rid, None)
        if not exact_match_relation:
            exact_match_relation = self._get_relation_and_store_in_cache(rid)
        return exact_match_relation

    def get_exact_match_relations(self, relations):
        """ Retrieve exact matched relations from ASER

        :param relations: a relations that contain the rids or eventuality pairs each of which contains two eids
        :type relations: Union[List[aser.relation.Relation], List[Dict[str, object]], List[str], List[Tuple[aser.eventuality.Eventuality, aser.eventuality.Eventuality]], List[Tuple[str, str]]]
        :return: the exact matched relations
        :rtype: List[aser.relation.Relation]
        """

        exact_match_relations = []
        if len(relations):
            if isinstance(relations[0], Relation):
                rids = [relation.rid for relation in relations]
            elif isinstance(relations[0], dict):
                rids = [relation["rid"] for relation in relations]
            elif isinstance(relations[0], str):
                rids = relations
            elif isinstance(relations[0], (tuple, list)) and len(relations[0]) == 2:
                if isinstance(relations[0][0], Eventuality) and isinstance(relations[0][1], Eventuality):
                    rids = [Relation.generate_rid(relation[0].eid, relation[1].eid) for relation in relations]
                elif isinstance(relations[0][0], str) and isinstance(relations[0][1], str):
                    rids = [Relation.generate_rid(relation[0], relation[1]) for relation in relations]
                else:
                    raise ValueError(
                        "Error: relations should be [(an instance of Eventuality, an instance of Eventuality), ...] or [(hid, tid), ...]."
                    )
            else:
                raise ValueError(
                    "Error: relations should be instances of Relation, dictionaries, rids, [(an instance of Eventuality, an instance of Eventuality), ...], or [(hid, tid), ...]."
                )

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
        """ Retrieve multiple partial matched relations by keys and values from ASER

        :param bys: the given columns to match
        :type bys: List[str]
        :param keys: the given values to match
        :type keys: List[str]
        :param order_bys: the columns whose value are used to sort rows
        :type order_bys: Union[List[str], None] (default = None)
        :param reverse: whether to sort in a reversed order
        :type reverse: bool (default = False)
        :param top_n: how many relations to return, default `None` for all relations
        :type top_n: Union[int, None]  (default = None)
        :return: the partial matched relations
        :rtype: List[aser.relation.Relation]
        """

        assert len(bys) == len(keys)
        for i in range(len(bys) - 1, -1, -1):
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
                key_match_relations = list(
                    map(
                        self._convert_row_to_relation,
                        self._conn.get_rows_by_keys(
                            self.relation_table_name, [bys[by_index]], [keys[by_index]], self.relation_columns
                        )
                    )
                )
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
        return list(
            map(
                self._convert_row_to_relation,
                self._conn.get_rows_by_keys(
                    self.relation_table_name,
                    bys,
                    keys,
                    self.relation_columns,
                    order_bys=order_bys,
                    reverse=reverse,
                    top_n=top_n
                )
            )
        )

    """
    Addtional APIs
    """

    def get_related_eventualities(self, eventuality):
        """ Retrieve related (connected) eventualities from ASER

        :param eventuality: an eventuality that contains the eid
        :type eventuality: Union[aser.eventuality.Eventuality, Dict[str, object], str]
        :return: the related eventualities
        :rtype: List[Tuple[aser.eventuality.Eventuality, aser.relation.Relation]]
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
                if eid in self.partial2rids_cache["hid"]:  # hit
                    related_rids = self.partial2rids_cache["hid"].get(eid, list())
                    related_relations = self.get_exact_match_relations(related_rids)
                    tids = [x.tid for x in related_relations]
                    t_eventualities = self.get_exact_match_eventualities(tids)
                else:  # miss
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


class ASERConceptConnection(object):
    """ Concept connection for ASER (including concepts, concept_instance_pairs, and relations)

    """
    def __init__(self, db_path, db="sqlite", mode='cache', chunksize=CHUNKSIZE):
        """

        :param db_path: database path
        :type db_path: str
        :param db: the backend database, e.g., "sqlite" or "mongodb"
        :type db: str (default = sqlite)
        :param mode: the mode to use the connection.
            "insert": this connection is only used to insert/update rows;
            "cache": this connection caches some contents that have been retrieved;
            "memory": this connection loads all contents in memory;
        :type mode: str (default = "cache")
        :param chunksize: the chunksize to load/write database
        :type chunksize: int (default = 32768)
        """

        if db == "sqlite":
            self._conn = SqliteDBConnection(db_path, chunksize)
        elif db == "mongodb":
            self._conn = MongoDBConnection(db_path, chunksize)
        else:
            raise NotImplementedError("Error: %s database is not supported!" % (db))
        self.mode = mode
        if self.mode not in ["insert", "cache", "memory"]:
            raise NotImplementedError("Error: only support insert/cache/memory modes.")

        self.concept_table_name = CONCEPT_TABLE_NAME
        self.concept_columns = CONCEPT_COLUMNS
        self.concept_column_types = CONCEPT_COLUMN_TYPES
        self.concept_instance_pair_table_name = CONCEPTINSTANCEPAIR_TABLE_NAME
        self.concept_instance_pair_columns = CONCEPTINSTANCEPAIR_COLUMNS
        self.concept_instance_pair_column_types = CONCEPTINSTANCEPAIR_COLUMN_TYPES
        self.relation_table_name = RELATION_TABLE_NAME
        self.relation_columns = RELATION_COLUMNS
        self.relation_column_types = RELATION_COLUMN_TYPES

        self.cids = set()
        self.eids = set()
        self.rids = set()

        self.cid2concept_cache = dict()
        self.cid2eid_pattern_scores = dict()
        self.rid2relation_cache = dict()
        self.eid2cid_scores = dict()
        self.partial2cids_cache = dict()
        self.partial2rids_cache = {"hid": dict()}

        self.init()

    def init(self):
        """ Initialize the ASERConceptConnection, including creating tables, loading cids, eids, rids, and building cache

        """

        for table_name, columns, column_types in zip(
            [self.concept_table_name, self.concept_instance_pair_table_name, self.relation_table_name],
            [self.concept_columns, self.concept_instance_pair_columns, self.relation_columns],
            [self.concept_column_types, self.concept_instance_pair_column_types, self.relation_column_types]
        ):
            if len(columns) == 0 or len(column_types) == 0:
                raise NotImplementedError(
                    "Error: %s_columns and %s_column_types must be defined" % (table_name, table_name)
                )
            try:
                self._conn.create_table(table_name, columns, column_types)
            except:
                pass

        if self.mode == 'memory':
            for c in map(
                self._convert_row_to_concept, self._conn.get_columns(self.concept_table_name, self.concept_columns)
            ):
                self.cids.add(c.cid)
                self.cid2concept_cache[c.cid] = c
                # handle another cache
                for k, v in self.partial2cids_cache.items():
                    if getattr(c, k) not in v:
                        v[getattr(c, k)] = [c.cid]
                    else:
                        v[getattr(c, k)].append(c.cid)
            for p in map(
                self._convert_row_to_concept_instance_pair,
                self._conn.get_columns(self.concept_instance_pair_table_name, self.concept_instance_pair_columns)
            ):
                self.eids.add(p.eid)
                # handle another cache
                if p.cid not in self.cid2eid_pattern_scores:
                    self.cid2eid_pattern_scores[p.cid] = [(p.eid, p.pattern, p.score)]
                else:
                    self.cid2eid_pattern_scores[p.cid].append((p.eid, p.pattern, p.score))
                if p.eid not in self.eid2cid_scores:
                    self.eid2cid_scores[p.eid] = [(p.cid, p.score)]
                else:
                    self.eid2cid_scores[p.eid].append((p.cid, p.score))
            for r in map(
                self._convert_row_to_relation, self._conn.get_columns(self.relation_table_name, self.relation_columns)
            ):
                self.rids.add(r.rid)
                self.rid2relation_cache[r.rid] = r
                # handle another cache
                for k, v in self.partial2rids_cache.items():
                    if getattr(r, k) not in v:
                        v[getattr(r, k)] = [r.rid]
                    else:
                        v[getattr(r, k)].append(r.rid)
        else:
            for x in self._conn.get_columns(self.concept_table_name, ["_id"]):
                self.cids.add(x["_id"])
            for x in self._conn.get_columns(self.concept_instance_pair_table_name, ["eid"]):
                self.eids.add(x["eid"])
            for x in self._conn.get_columns(self.relation_table_name, ["_id"]):
                self.rids.add(x["_id"])

    def close(self):
        """ Close the ASERConceptConnection safely

        """

        self._conn.close()
        self.cids.clear()
        self.eids.clear()
        self.rids.clear()
        self.cid2concept_cache.clear()
        self.cid2eid_pattern_scores.clear()
        self.eid2cid_scores.clear()
        self.rid2relation_cache.clear()
        for k in self.partial2cids_cache:
            self.partial2cids_cache[k].clear()
        for k in self.partial2rids_cache:
            self.partial2rids_cache[k].clear()

    """
    KG (Concepts)
    """

    def _convert_concept_to_row(self, concept):
        row = OrderedDict({"_id": concept.cid})
        for c in self.concept_columns[1:-1]:
            d = getattr(concept, c)
            if isinstance(d, list):
                row[c] = " ".join(d)
            else:
                row[c] = d
        row["info"] = concept.encode()
        return row

    def _convert_row_to_concept(self, row):
        concept = ASERConcept().decode(row["info"])
        concept.cid = row["_id"]
        return concept

    def get_concept_columns(self, columns):
        """ Get column information from concepts

        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
        return self._conn.get_columns(self.concept_table_name, columns)

    def _insert_concept(self, concept):
        row = self._convert_concept_to_row(concept)
        self._conn.insert_row(self.concept_table_name, row)
        if self.mode == "insert":
            self.cids.add(concept.cid)
        elif self.mode == "cache":
            self.cids.add(concept.cid)
            self.cid2concept_cache[concept.cid] = concept
            for k, v in self.partial2cids_cache.items():
                if concept.get(k) not in v:
                    v[concept.get(k)] = [concept.cid]
                else:
                    v[concept.get(k)].append(concept.cid)
        return concept

    def _insert_concepts(self, concepts):
        rows = list(map(self._convert_concept_to_row, concepts))
        self._conn.insert_rows(self.concept_table_name, rows)
        if self.mode == "insert":
            for concept in concepts:
                self.cids.add(concept.cid)
        elif self.mode == "cache":
            for concept in concepts:
                self.cids.add(concept.cid)
                self.cid2concept_cache[concept.cid] = concept
                for k, v in self.partial2cids_cache.items():
                    if concept.get(k) in v:
                        v[concept.get(k)].append(concept.cid)
        elif self.mode == "memory":
            for concept in concepts:
                self.cids.add(concept.cid)
                self.cid2concept_cache[concept.cid] = concept
                for k, v in self.partial2cids_cache.items():
                    if concept.get(k) not in v:
                        v[concept.get(k)] = [concept.cid]
                    else:
                        v[concept.get(k)].append(concept.cid)
        return concepts

    def _get_concept_and_store_in_cache(self, cid):
        return self._get_concepts_and_store_in_cache([cid])[0]

    def _get_concepts_and_store_in_cache(self, cids):
        concepts = list(
            map(
                self._convert_row_to_concept,
                self._conn.select_rows(self.concept_table_name, cids, self.concept_columns)
            )
        )
        for concept in concepts:
            if concept:
                self.cid2concept_cache[concept.cid] = concept
                cached_eid_pattern_scores = self.cid2eid_pattern_scores.get(concept.cid, None)
                if not cached_eid_pattern_scores:
                    eid_pattern_scores = self._conn.get_rows_by_keys(
                        self.concept_instance_pair_table_name,
                        bys=["cid"],
                        keys=[concept.cid],
                        columns=["eid", "pattern", "score"]
                    )
                    cached_eid_pattern_scores = [(x["eid"], x["pattern"], x["score"]) for x in eid_pattern_scores]
                    self.cid2eid_pattern_scores[concept.cid] = cached_eid_pattern_scores
                concept.instances = cached_eid_pattern_scores
        return concepts

    def _update_concept(self, concept):
        # append/update new instances
        updated_concept = self.cid2concept_cache.get(concept.cid, None)
        if not updated_concept:  # self.mode == "memory" or hit in cache
            if self.mode == "insert":
                updated_concept = self._convert_row_to_concept(
                    self._conn.select_row(self.concept_table_name, concept.cid, self.concept_columns)
                )
            else:
                updated_concept = self._get_concept_and_store_in_cache(concept.cid)

        for x in concept.instances:
            matched = False
            for y in updated_concept.instances:
                if y[0] == x[0]:
                    y[2] += x[2]
                    matched = True
                    break
            if not matched:
                updated_concept.instances.append(x)

        update_op = self._conn.get_update_op(["info"], "=")
        row = self._convert_concept_to_row(updated_concept)
        self._conn.update_row(self.concept_table_name, row, update_op, ["info"])

        if self.mode == "insert":
            return None  # don"t care
        else:
            return updated_concept

    def _update_concepts(self, concepts):
        updated_concepts = []
        missed_indices = []
        missed_cids = []
        for idx, concept in enumerate(concepts):
            if concept.cid not in self.cids:
                updated_concepts.append(None)
            else:
                updated_concept = self.cid2concept_cache.get(concept.cid, None)
                updated_concepts.append(updated_concept)
                if not updated_concept:
                    missed_indices.append(idx)
                    missed_cids.append(concept.cid)
        if self.mode == "insert":
            for idx, updated_concept in enumerate(
                map(
                    self._convert_row_to_concept,
                    self._conn.select_rows(self.concept_table_name, missed_cids, self.concept_columns)
                )
            ):
                updated_concepts[missed_indices[idx]] = updated_concept
        else:
            for idx, updated_concept in enumerate(self._get_concepts_and_store_in_cache(missed_cids)):
                updated_concepts[missed_indices[idx]] = updated_concept

        for idx, concept in enumerate(concepts):
            if not updated_concepts[idx]:
                updated_concepts[idx] = concept
            else:
                updated_concept = updated_concepts[idx]
                for x in concept.instances:
                    matched = False
                    for y in updated_concept.instances:
                        if y[0] == x[0]:
                            y[2] += x[2]
                            matched = True
                            break
                    if not matched:
                        updated_concept.instances.append(x)

        update_op = self._conn.get_update_op(["info"], "=")
        rows = list(map(self._convert_concept_to_row, updated_concepts))
        self._conn.update_rows(self.concept_table_name, rows, update_op, ["info"])

        if self.mode == "insert":
            return [None] * len(concepts)  # don"t care
        return updated_concepts

    def insert_concept(self, concept):
        """ Insert/Update a concept into ASER
        (suggestion: consider to use `insert_concepts` if you want to insert multiple concepts)

        :param concept: a concept to insert/update
        :type concept: aser.concept.ASERConcept
        :return: the inserted/updated concept
        :rtype: aser.concept.ASERConcept
        """

        if concept.cid not in self.cids:
            concept = self._insert_concept(concept)
        else:
            concept = self._update_concept(concept)
        return concept

    def insert_concepts(self, concepts):
        """ Insert/Update concepts into ASER

        :param concepts: concepts to insert/update
        :type concepts: List[aser.concept.ASERConcept]
        :return: the inserted/updated concepts
        :rtype: List[aser.concept.ASERConcept]
        """

        results = []
        new_concepts = []
        existing_indices = []
        existing_indices = []
        existing_concepts = []
        for idx, concept in enumerate(concepts):
            if concept.cid not in self.cids:
                new_concepts.append(concept)
                results.append(concept)
            else:
                existing_indices.append(idx)
                existing_concepts.append(concept)
                results.append(None)
        if len(new_concepts):
            self._insert_concepts(new_concepts)
        if len(existing_concepts):
            for idx, updated_concept in enumerate(self._update_concepts(existing_concepts)):
                results[existing_indices[idx]] = updated_concept

        return results

    def get_exact_match_concept(self, concept):
        """ Retrieve a exact matched concept from ASER
        (suggestion: consider to use `get_exact_match_concepts` if you want to retrieve multiple concepts)

        :param concept: a concept that contains the cid
        :type concept: Union[aser.concept.ASERConcept, Dict[str, object], str]
        :return: the exact matched concept
        :rtype: aser.concept.ASERConcept
        """

        if isinstance(concept, ASERConcept):
            cid = concept.cid
        elif isinstance(concept, dict):
            cid = concept["cid"]
        elif isinstance(concept, str):
            cid = concept
        else:
            raise ValueError("Error: conceptualize should be an instance of ASERConcept, a dictionary, or a cid.")

        if cid not in self.cids:
            return None
        exact_match_concept = self.cid2concept_cache.get(cid, None)
        if not exact_match_concept:
            exact_match_concept = self._get_concept_and_store_in_cache(cid)
        return exact_match_concept

    def get_exact_match_concepts(self, concepts):
        """ Retrieve multiple exact matched concepts from ASER

        :param concepts: concepts
        :type concepts: Union[List[aser.concept.ASERConcept], List[Dict[str, object]], List[str]]
        :return: the exact matched concepts
        :rtype: List[aser.concept.ASERConcept]
        """

        exact_match_concepts = []
        if len(concepts):
            if isinstance(concepts[0], ASERConcept):
                cids = [concept.cid for concept in concepts]
            elif isinstance(concepts[0], dict):
                cids = [concept["cid"] for concept in concepts]
            elif isinstance(concepts[0], str):
                cids = concepts
            else:
                raise ValueError("Error: concepts should instances of ASERConcept, dictionaries, or cids.")

            missed_indices = []
            missed_cids = []
            for idx, cid in enumerate(cids):
                if cid not in self.cids:
                    exact_match_concepts.append(None)
                exact_match_concept = self.cid2concept_cache.get(cid, None)
                exact_match_concepts.append(exact_match_concept)
                if not exact_match_concept:
                    missed_indices.append(idx)
                    missed_cids.append(cid)
            for idx, exact_match_concept in enumerate(self._get_concepts_and_store_in_cache(missed_cids)):
                exact_match_concepts[missed_indices[idx]] = exact_match_concept
        return exact_match_concepts

    def get_concepts_by_keys(self, bys, keys, order_bys=None, reverse=False, top_n=None):
        """ Retrieve multiple partial matched concepts by keys and values from ASER

        :param bys: the given columns to match
        :type bys: List[str]
        :param keys: the given values to match
        :type keys: List[str]
        :param order_bys: the columns whose value are used to sort rows
        :type order_bys: List[str]
        :param reverse: whether to sort in a reversed order
        :type reverse: bool
        :param top_n: how many concepts to return, default `None` for all concepts
        :type top_n: int
        :return: the partial matched concepts
        :rtype: List[aser.concept.Concepts]
        """

        assert len(bys) == len(keys)
        for i in range(len(bys) - 1, -1, -1):
            if bys[i] not in self.concept_columns:
                bys.pop(i)
                keys.pop(i)
        if len(bys) == 0:
            return []
        return list(
            map(
                self._convert_row_to_concept,
                self._conn.get_rows_by_keys(
                    self.concept_table_name,
                    bys,
                    keys,
                    self.concept_columns,
                    order_bys=order_bys,
                    reverse=reverse,
                    top_n=top_n
                )
            )
        )

    def get_concept_given_str(self, concept_str):
        """ Retrieve the exact matched concept given a string from ASER

        :param concept_str: a string representation of a concept
        :type concept_str: str
        :return: the exact matched concept
        :rtype: aser.concept.ASERConcept
        """

        cid = ASERConcept.generate_cid(concept_str)
        return self.get_exact_match_concept(cid)

    def get_concepts_given_strs(self, concept_strs):
        """ Retrieve the exact matched concepts given strings from ASER

        :param concept_str: string representations of concepts
        :type concept_str: List[str]
        :return: the exact matched concepts
        :rtype: List[aser.concept.ASERConcept]
        """

        cids = list(map(ASERConcept.generate_cid, concept_strs))
        return self.get_exact_match_concepts(cids)

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
        return Relation(
            row["hid"], row["tid"], {r: cnt
                                     for r, cnt in row.items() if isinstance(cnt, float) and cnt > 0.0}
        )

    def get_relation_columns(self, columns):
        """ Get column information from relations

        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
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
                if getattr(relation, k) in v:
                    v[getattr(relation, k)].append(relation.rid)
        elif self.mode == "memory":
            self.rids.add(relation.rid)
            self.rid2relation_cache[relation.rid] = relation
            for k, v in self.partial2rids_cache.items():
                if getattr(relation, k) not in v:
                    v[getattr(relation, k)] = [relation.rid]
                else:
                    v[getattr(relation, k)].append(relation.rid)
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
                    if getattr(relation, k) in v:
                        v[getattr(relation, k)].append(relation.rid)
        elif self.mode == "memory":
            for relation in relations:
                self.rids.add(relation.rid)
                self.rid2relation_cache[relation.rid] = relation
                for k, v in self.partial2rids_cache.items():
                    if getattr(relation, k) not in v:
                        v[getattr(relation, k)] = [relation.rid]
                    else:
                        v[getattr(relation, k)].append(relation.rid)
        return relations

    def _get_relation_and_store_in_cache(self, rid):
        return self._get_relations_and_store_in_cache([rid])[0]

    def _get_relations_and_store_in_cache(self, rids):
        relations = list(
            map(
                self._convert_row_to_relation,
                self._conn.select_rows(self.relation_table_name, rids, self.relation_columns)
            )
        )
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
        """ Insert/Update a relation into ASER
        (suggestion: consider to use `insert_relations` if you want to insert multiple relations)

        :param relation: a relation to insert/update
        :type relation: aser.relation.Relation
        :return: the inserted/updated relation
        :rtype: aser.relation.Relation
        """

        if relation.rid not in self.rid2relation_cache:
            return self._insert_relation(relation)
        else:
            return self._update_relation(relation)

    def insert_relations(self, relations):
        """ Insert/Update relations into ASER

        :param relations: relations to insert/update
        :type relations: List[aser.relation.Relation]
        :return: the inserted/updated relations
        :rtype: List[aser.relation.Relation]
        """

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
        """ Retrieve an exact matched relation from ASER
        (suggestion: consider to use `get_exact_match_relations` if you want to retrieve multiple relations)

        :param relation: a relation that contains the rid or a concept pair that contains two cids
        :type relation: Union[aser.relation.Relation, Dict[str, object], str, Tuple[aser.concept.ASERConcept, aser.concept.ASERConcept], Tuple[str, str]]
        :return: the exact matched relation
        :rtype: aser.relation.Relation
        """

        if isinstance(relation, Relation):
            rid = relation.rid
        elif isinstance(relation, dict):
            rid = relation["rid"]
        elif isinstance(relation, str):
            rid = relation
        elif isinstance(relation, (tuple, list)) and len(relation) == 2:
            if isinstance(relation[0], ASERConcept) and isinstance(relation[1], ASERConcept):
                rid = Relation.generate_rid(relation[0].cid, relation[1].cid)
            elif isinstance(relation[0], str) and isinstance(relation[1], str):
                rid = Relation.generate_rid(relation[0], relation[1])
            else:
                raise ValueError(
                    "Error: relation should be (an instance of ASERConcept, an instance of ASERConcept) or (hid, tid)."
                )
        else:
            raise ValueError(
                "Error: relation should be an instance of Relation, a dictionary, rid,"
                "(an instance of ASERConcept, an instance of ASERConcept), or (hid, tid)."
            )

        if rid not in self.rids:
            return None
        exact_match_relation = self.rid2relation_cache.get(rid, None)
        if not exact_match_relation:
            exact_match_relation = self._get_relation_and_store_in_cache(rid)
        return exact_match_relation

    def get_exact_match_relations(self, relations):
        """ Retrieve exact matched relations from ASER

        :param relations: a relations that contain the rids or concept pairs each of which contains two cids
        :type relations: Union[List[aser.relation.Relation], List[Dict[str, object]], List[str], List[Tuple[aser.concept.ASERConcept, aser.concept.ASERConcept]], List[Tuple[str, str]]]
        :return: the exact matched relations
        :rtype: List[aser.relation.Relation]
        """

        exact_match_relations = []
        if len(relations):
            if isinstance(relations[0], Relation):
                rids = [relation.rid for relation in relations]
            elif isinstance(relations[0], dict):
                rids = [relation["rid"] for relation in relations]
            elif isinstance(relations[0], str):
                rids = relations
            elif isinstance(relations[0], (tuple, list)) and len(relations[0]) == 2:
                if isinstance(relations[0][0], ASERConcept) and isinstance(relations[0][1], ASERConcept):
                    rids = [Relation.generate_rid(relation[0].cid, relation[1].cid) for relation in relations]
                elif isinstance(relations[0][0], str) and isinstance(relations[0][1], str):
                    rids = [Relation.generate_rid(relation[0], relation[1]) for relation in relations]
                else:
                    raise ValueError(
                        "Error: relations should be [(an instance of ASERConcept, an instance of ASERConcept), ...] or [(hid, tid), ...]."
                    )
            else:
                raise ValueError(
                    "Error: relations should be instances of Relation, dictionaries, rids, [(an instance of ASERConcept, an instance of ASERConcept), ...], or [(hid, tid), ...]."
                )

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
        """ Retrieve multiple partial matched relations by keys and values from ASER

        :param bys: the given columns to match
        :type bys: List[str]
        :param keys: the given values to match
        :type keys: List[str]
        :param order_bys: the columns whose value are used to sort rows
        :type order_bys: Union[List[str], None] (default = None)
        :param reverse: whether to sort in a reversed order
        :type reverse: bool (default = False)
        :param top_n: how many relations to return, default `None` for all relations
        :type top_n: Union[int, None] (default = None)
        :return: the partial matched relations
        :rtype: List[aser.relation.Relation]
        """

        assert len(bys) == len(keys)
        for i in range(len(bys) - 1, -1, -1):
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
                key_match_relations = list(
                    map(
                        self._convert_row_to_relation,
                        self._conn.get_rows_by_keys(
                            self.relation_table_name, [bys[by_index]], [keys[by_index]], self.relation_columns
                        )
                    )
                )
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
        return list(
            map(
                self._convert_row_to_relation,
                self._conn.get_rows_by_keys(
                    self.relation_table_name,
                    bys,
                    keys,
                    self.relation_columns,
                    order_bys=order_bys,
                    reverse=reverse,
                    top_n=top_n
                )
            )
        )

    """
    KG (ConceptInstancePairs)
    """

    def _convert_concept_instance_pair_to_row(self, concept_instance_pair):
        if isinstance(concept_instance_pair, ASERConceptInstancePair):
            row = OrderedDict(
                {
                    "_id": concept_instance_pair.pid,
                    "cid": concept_instance_pair.cid,
                    "eid": concept_instance_pair.eid,
                    "pattern": concept_instance_pair.pattern,
                    "score": concept_instance_pair.score
                }
            )
        elif isinstance(concept_instance_pair, (list, tuple)) and len(concept_instance_pair) == 3:
            pid = ASERConceptInstancePair.generate_pid(concept_instance_pair[0].cid, concept_instance_pair[1].eid)
            row = OrderedDict(
                {
                    "_id": pid,
                    "cid": concept_instance_pair[0].cid,
                    "eid": concept_instance_pair[1].eid,
                    "pattern": concept_instance_pair[1].pattern,
                    "score": concept_instance_pair[2]
                }
            )
        return row

    def _convert_row_to_concept_instance_pair(self, row):
        return ASERConceptInstancePair(row["cid"], row["eid"], row["pattern"], row["score"])

    def get_concept_instance_pair_columns(self, columns):
        """ Get column information from concepts

        :param columns: the columns to retrieve
        :type columns: List[str]
        :return: a list of retrieved rows
        :rtype: List[Dict[str, object]]
        """
        return self._conn.get_columns(self.concept_instance_pair_columns, columns)

    def _insert_concept_instance_pair(self, concept_instance_pair):
        row = self._convert_concept_instance_pair_to_row(concept_instance_pair)
        self._conn.insert_row(self.concept_instance_pair_table_name, row)
        if self.mode == "insert":
            self.eids.add(concept_instance_pair.eid)
        elif self.mode == "cache":
            self.eids.add(concept_instance_pair.eid)
            if concept_instance_pair.cid in self.cid2eid_pattern_scores:
                self.cid2eid_pattern_scores[concept_instance_pair.cid].append(
                    (concept_instance_pair.eid, concept_instance_pair.pattern, concept_instance_pair.score)
                )
            if concept_instance_pair.eid in self.eid2cid_scores:
                self.eid2cid_scores[concept_instance_pair.eid].append(
                    (concept_instance_pair.cid, concept_instance_pair.score)
                )
        elif self.mode != "memory":
            self.eids.add(concept_instance_pair.eid)
            if concept_instance_pair.cid not in self.cid2eid_pattern_scores:
                self.cid2eid_pattern_scores[concept_instance_pair.cid] = [
                    (concept_instance_pair.eid, concept_instance_pair.pattern, concept_instance_pair.score)
                ]
            else:
                self.cid2eid_pattern_scores[concept_instance_pair.cid].append(
                    (concept_instance_pair.eid, concept_instance_pair.pattern, concept_instance_pair.score)
                )
            if concept_instance_pair.eid not in self.eid2cid_scores:
                self.eid2cid_scores[concept_instance_pair.eid] = [
                    (concept_instance_pair.cid, concept_instance_pair.score)
                ]
            else:
                self.eid2cid_scores[concept_instance_pair.eid].append(
                    (concept_instance_pair.cid, concept_instance_pair.score)
                )
        return self._convert_row_to_concept_instance_pair(row)

    def _insert_concept_instance_pairs(self, concept_instance_pairs):
        rows = list(map(self._convert_concept_instance_pair_to_row, concept_instance_pairs))
        self._conn.insert_rows(self.concept_instance_pair_table_name, rows)
        if self.mode == "insert":
            for concept_instance_pair in concept_instance_pairs:
                self.eids.add(concept_instance_pair.eid)
        elif self.mode == "cache":
            for concept_instance_pair in concept_instance_pairs:
                self.eids.add(concept_instance_pair.eid)
                if concept_instance_pair.cid in self.cid2eid_pattern_scores:
                    self.cid2eid_pattern_scores[concept_instance_pair.cid].append(
                        (concept_instance_pair.eid, concept_instance_pair.pattern, concept_instance_pair.score)
                    )
                if concept_instance_pair.eid in self.eid2cid_scores:
                    self.eid2cid_scores[concept_instance_pair.eid].append(
                        (concept_instance_pair.cid, concept_instance_pair.score)
                    )
        elif self.mode == "memory":
            for concept_instance_pair in concept_instance_pairs:
                self.eids.add(concept_instance_pair.eid)
                if concept_instance_pair.cid not in self.cid2eid_pattern_scores:
                    self.cid2eid_pattern_scores[concept_instance_pair.cid] = [
                        (concept_instance_pair.eid, concept_instance_pair.pattern, concept_instance_pair.score)
                    ]
                else:
                    self.cid2eid_pattern_scores[concept_instance_pair.cid].append(
                        (concept_instance_pair.eid, concept_instance_pair.pattern, concept_instance_pair.score)
                    )
                if concept_instance_pair.eid not in self.eid2cid_scores:
                    self.eid2cid_scores[concept_instance_pair.eid] = [
                        (concept_instance_pair.cid, concept_instance_pair.score)
                    ]
                else:
                    self.eid2cid_scores[concept_instance_pair.eid].append(
                        (concept_instance_pair.cid, concept_instance_pair.score)
                    )
        return [self._convert_row_to_concept_instance_pair(row) for row in rows]

    def _update_concept_instance_pair(self, concept_instance_pair):
        # update db
        update_op = self._conn.get_update_op(["score"], "+")
        row = self._convert_concept_instance_pair_to_row(concept_instance_pair)
        self._conn.update_row(self.concept_instance_pair_table_name, row, update_op, ["score"])

        # updata cache
        updated_score = None
        if self.mode == "insert":
            return None  # don"t care
        cached_cid_scores = self.eid2cid_scores.get(concept_instance_pair.eid, None)
        if cached_cid_scores:
            for idx, cid_score in enumerate(cached_cid_scores):
                if concept_instance_pair.cid == cid_score[0]:
                    updated_score = cid_score[1] + concept_instance_pair.score
                    cached_cid_scores[idx] = (cid_score[0], updated_score)
                    break
        cached_eid_pattern_scores = self.cid2eid_pattern_scores.get(concept_instance_pair.cid, None)
        if cached_eid_pattern_scores:
            for idx, eid_pattern_score in enumerate(cached_eid_pattern_scores):
                if concept_instance_pair.eid == eid_pattern_score[0]:
                    updated_score = eid_pattern_score[2] + concept_instance_pair.score
                    cached_eid_pattern_scores[idx] = (eid_pattern_score[0], eid_pattern_score[1], updated_score)
                    break
        if updated_score is None:
            updated_score = self._conn.select_row(self.concept_instance_pair_table_name, row["_id"], ["score"])["score"]
        return ASERConceptInstancePair(
            concept_instance_pair.cid, concept_instance_pair.eid, concept_instance_pair.pattern, updated_score
        )

    def _update_concept_instance_pairs(self, concept_instance_pairs):
        # update db
        update_op = self._conn.get_update_op(["score"], "+")
        rows = list(map(self._convert_concept_instance_pair_to_row, concept_instance_pairs))
        self._conn.update_rows(self.concept_instance_pair_table_name, rows, update_op, ["score"])

        # update cache
        if self.mode == "insert":
            return [None] * len(concept_instance_pairs)  # don"t care
        results = []
        updated_scores = []
        missed_indices = []
        missed_pids = []
        for idx, concept_instance_pair in enumerate(concept_instance_pairs):
            cached_cid_scores = self.eid2cid_scores.get(concept_instance_pair.eid, None)
            if cached_cid_scores:
                for idx, cid_score in enumerate(cached_cid_scores):
                    if concept_instance_pair.cid == cid_score[0]:
                        updated_score = cid_score[1] + concept_instance_pair.score
                        cached_cid_scores[idx] = (cid_score[0], updated_score)
                        break
            cached_eid_pattern_scores = self.cid2eid_pattern_scores.get(concept_instance_pair.cid, None)
            if cached_eid_pattern_scores:
                for idx, eid_pattern_score in enumerate(cached_eid_pattern_scores):
                    if concept_instance_pair.eid == eid_pattern_score[0]:
                        updated_score = eid_pattern_score[2] + concept_instance_pair.score
                        cached_eid_pattern_scores[idx] = (eid_pattern_score[0], eid_pattern_score[1], updated_score)
                        break
            if updated_score is None:
                missed_indices.append(idx)
                updated_scores.append(None)
                missed_pids.append(concept_instance_pair.pid)
            else:
                updated_scores.append(updated_score)
        if len(missed_indices):
            for idx, updated_row in enumerate(
                self._conn.select_rows(self.concept_instance_pair_table_name, missed_pids, ["score"])
            ):
                updated_scores[missed_indices[idx]] = updated_row["score"]
        return [
            ASERConceptInstancePair(
                concept_instance_pair.cid, concept_instance_pair.eid, concept_instance_pair.pattern, updated_score
            ) for concept_instance_pair, updated_score in zip(concept_instance_pairs, updated_score)
        ]

    def insert_concept_instance_pair(self, concept_instance_pair):
        """Insert/Update a concept_instance_pair into ASER
        (suggestion: consider to use `insert_concept_instance_pairs` if you want to insert multiple pairs)

        :param concept_instance_pair: a concept-instance pair to insert/update
        :type concept_instance_pair: Union[aser.concept.ASERConceptInstancePair, Tuple[aser.concept.ASERConcpet, aser.event.Eventuality, float]]
        :return: the inserted/updated concept-instance pair
        :rtype: aser.concept.ASERConceptInstancePair
        """

        if not isinstance(concept_instance_pair, ASERConceptInstancePair):
            concept_instance_pair = ASERConceptInstancePair(
                concept_instance_pair[0].cid,
                concept_instance_pair[1].eid,
                concept_instance_pair[1].pattern,
                concept_instance_pair[2]
            )
        if concept_instance_pair.cid in self.cids and concept_instance_pair.eid in self.eids:
            return self._update_concept_instance_pair(concept_instance_pair)
        else:
            return self._insert_concept_instance_pair(concept_instance_pair)

    def insert_concept_instance_pairs(self, concept_instance_pairs):
        """Insert/Update concept_instance_pairs into ASER

        :param concept_instance_pairs: concept-instance pairs to insert/update
        :type concept_instance_pairs: Union[List[aser.concept.ASERConceptInstancePair], List[Tuple[aser.concept.ASERConcpet, aser.event.Eventuality, float]]]
        :return: the inserted/updated concept-instance pairs
        :rtype: List[aser.concept.ASERConceptInstancePair]
        """

        results = [None] * len(concept_instance_pairs)
        new_concept_instance_pairs = []
        existing_indices = []
        existing_concept_instance_pairs = []
        for idx, concept_instance_pair in enumerate(concept_instance_pairs):
            if not isinstance(concept_instance_pair, ASERConceptInstancePair):
                concept_instance_pair = ASERConceptInstancePair(
                    concept_instance_pair[0].cid,
                    concept_instance_pair[1].eid,
                    concept_instance_pair[1].pattern,
                    concept_instance_pair[2]
                )
            if concept_instance_pair.cid in self.cids and concept_instance_pair.eid in self.eids:
                existing_indices.append(idx)
                existing_concept_instance_pairs.append(concept_instance_pair)
                results.append(None)
            else:
                new_concept_instance_pairs.append(concept_instance_pair)
                results.append(concept_instance_pair)
        if len(new_concept_instance_pairs):
            self._insert_concept_instance_pairs(new_concept_instance_pairs)
        if len(existing_indices):
            for idx, updated_pair in enumerate(self._update_concept_instance_pairs(existing_concept_instance_pairs)):
                results[existing_indices[idx]] = updated_pair
        return results

    def get_eventualities_given_concept(self, concept):
        """ Retrieve original eventualities given a concept from ASER

        :param concept: concept that corresponds to some eventualities
        :type concept: Union[aser.concept.ASERConcpet, Dict[str, object], str]
        :return: the linked eventualities
        :rtype: List[aser.eventuality.Eventuality]
        """

        if self.mode == "insert":
            return []
        if isinstance(concept, ASERConcept):
            cid = concept.cid
        elif isinstance(concept, dict):
            cid = concept["cid"]
        elif isinstance(concept, str):
            cid = concept
        else:
            raise ValueError("Error: conceptualize should be an instance of ASERConcept, a dictionary, or a cid.")

        cached_eid_pattern_scores = self.cid2eid_pattern_scores.get(cid, None)
        if cached_eid_pattern_scores:
            return cached_eid_pattern_scores
        else:
            eid_pattern_scores = self._conn.get_rows_by_keys(
                self.concept_instance_pair_table_name, bys=["cid"], keys=[cid], columns=["eid", "pattern", "score"]
            )
            return eid_pattern_scores

    def get_concepts_given_eventuality(self, eventuality):
        """ Retrieve concepts given an eventuality from ASER

        :param eventuality: eventuality that conceptualizes to the given concept
        :type eventuality: Union[aser.eventuality.Eventuality, Dict[str, object], str]
        :return: the linked concepts
        :rtype: List[aser.concept.ASERConcepts]
        """

        if self.mode == "insert":
            return []
        if isinstance(eventuality, Eventuality):
            eid = eventuality.eid
        elif isinstance(eventuality, dict):
            eid = eventuality["eid"]
        elif isinstance(eventuality, str):
            eid = eventuality
        else:
            raise ValueError("Error: conceptualize should be an instance of Eventuality, a dictionary, or a eid.")

        cached_cid_scores = self.eid2cid_scores.get(eid, None)
        if cached_cid_scores:
            cids = [cid_score[0] for cid_score in cached_cid_scores]
            scores = [cid_score[1] for cid_score in cached_cid_scores]
        else:
            cid_scores = self._conn.get_rows_by_keys(
                self.concept_instance_pair_table_name, bys=["eid"], keys=[eid], columns=["cid", "score"]
            )
            cids = [cid_score["cid"] for cid_score in cid_scores]
            scores = [cid_score["score"] for cid_score in cid_scores]
        concepts = self.get_exact_match_concepts(cids)
        return list(zip(concepts, scores))

    """
    Additional APIs
    """

    def get_related_concepts(self, concept):
        """ Retrieve related (connected) concepts from ASER

        :param eventuality: a concept that contains the eid
        :type concept: Union[aser.concept.ASERConcept, Dict[str, object], str]
        :return: the related concepts
        :rtype: List[Tuple[aser.concept.ASERConcept, aser.relation.Relation]]
        """

        if self.mode == "insert":
            return []
        if isinstance(concept, ASERConcept):
            cid = concept.cid
        elif isinstance(concept, dict):
            cid = concept["cid"]
        elif isinstance(concept, str):
            cid = concept
        else:
            raise ValueError("Error: conceptualize should be an instance of ASERConcept, a dictionary, or a cid.")

        # cid == hid
        results = []
        if self.mode == "memory":
            if "hid" in self.partial2rids_cache:
                related_rids = self.partial2rids_cache["hid"].get(cid, list())
                related_relations = self.get_exact_match_relations(related_rids)
            else:
                related_relations = self.get_relations_by_keys(bys=["hid"], keys=[cid])
            tids = [x.tid for x in related_relations]
            t_concepts = self.get_exact_match_concepts(tids)
        elif self.mode == "cache":
            if "hid" in self.partial2rids_cache:
                if cid in self.partial2rids_cache["hid"]:  # hit
                    related_rids = self.partial2rids_cache["hid"].get(cid, list())
                    related_relations = self.get_exact_match_relations(related_rids)
                    tids = [x.tid for x in related_relations]
                    t_concepts = self.get_exact_match_concepts(tids)
                else:  # miss
                    related_relations = self.get_relations_by_keys(bys=["hid"], keys=[cid])
                    tids = [x.tid for x in related_relations]
                    t_concepts = self.get_exact_match_concepts(tids)
                    # update cache
                    self.partial2rids_cache["hid"][cid] = [relation.rid for relation in related_relations]
            else:
                related_relations = self.get_relations_by_keys(bys=["hid"], keys=[cid])
                tids = [x.tid for x in related_relations]
                t_concepts = self.get_exact_match_concepts(tids)
        return sorted(zip(t_concepts, related_relations), key=lambda x: sum(x[1].relations.values()))
