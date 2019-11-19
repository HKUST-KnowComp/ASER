import pickle
import os
import random
import heapq
import operator
import hashlib
from copy import copy
from functools import partial
from collections import defaultdict, OrderedDict
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.concept import ASERConcept, ASERConceptInstancePair
from aser.database.base import SqliteConnection, MongoDBConnection

CHUNKSIZE = 32768

CONCEPT_TABLE_NAME = "Concepts"
CONCEPT_TABLE_COLUMNS = ["_id", "pattern", "info"]
CONCEPT_TABLE_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "BLOB"]

RELATION_TABLE_NAME = "Relations":
RELATION_COLUMNS = ["_id", "hid", "tid"] + relation_senses
RELATION_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT"] + ["REAL"] * len(relation_senses)

CONCEPTINSTANCEPAIR_TABLE_NAME = "ConceptInstancePairs"
CONCEPTINSTANCEPAIR_COLUMNS = ["_id", "cid", "eid", "score"]
CONCEPTINSTANCEPAIR_COLUMN_TYPES = ["PRIMARY KEY", "TEXT", "TEXT", "REAL"]

class ASERConceptConnection(object):
    def __init__(self, db_path, db="sqlite", mode='cache', chunksize=-1):
        if db == "sqlite":
            self._conn = SqliteConnection(db_path, chunksize if chunksize > 0 else CHUNKSIZE)
        elif db == "mongoDB":
            self._conn = MongoDBConnection(db_path, chunksize if chunksize > 0 else CHUNKSIZE)
        else:
            raise NotImplementedError("Error: %s database is not supported!" % (db))
        self.mode = mode
        if self.mode not in ["insert", "cache", "memory"]:
            raise NotImplementedError("Error: only support insert/cache/memory modes.")

        self.concept_table_name = CONCEPT_TABLE_NAME
        self.concept_columns = CONCEPT_COLUMNS
        self.concept_column_types = CONCEPT_COLUMN_TYPES
        self.relation_table_name = RELATION_TABLE_NAME
        self.relation_columns = RELATION_COLUMNS
        self.relation_column_types = RELATION_COLUMN_TYPES
        self.concept_instance_pair_table_name = CONCEPTINSTANCEPAIR_TABLE_NAME
        self.concept_instance_pair_columns = CONCEPTINSTANCEPAIR_COLUMNS
        self.concept_instance_pair_column_types = CONCEPTINSTANCEPAIR_COLUMN_TYPES

        self.cids = set()
        self.rids = set()
        self.eids = set()

        self.cid2concept_cache = dict()
        self.rid2relation_cahce = dict()
        self.cid2eid_pattern_scores = dict()
        self.eid2cid_scores = dict()
        self.partial2cids_cache = dict()
        self.partial2rids_cache = {"hid": dict()}
        
        self.init()
        
    def init(self):
        """
        create tables
        load id sets
        load cache
        """
        for table_name, columns, column_types in zip(
            [self.concept_table_name, self.relation_table_name, self.concept_instance_pair_table_name], 
            [self.concept_columns, self.relation_columns, self.concept_instance_pair_columns],
            [self.concept_column_types, self.relation_column_types, self.concept_instance_pair_column_types]):
            if len(columns) == 0 or len(column_types) == 0:
                raise NotImplementedError("Error: %s_columns and %s_column_types must be defined" % (table_name, table_name))
            try:
                self._conn.create_table(table_name, columns, column_types)
            except:
                pass

        if self.mode == 'memory':
            for c in map(self._convert_row_to_concept, self._conn.get_columns(self.concept_table_name, self.concept_columns)):
                self.cids.add(c.cid)
                self.cid2concept_cache[c.cid] = c
                # handle another cache
                for k, v in self.partial2cids_cache.items():
                    if getattr(c, k) not in v:
                        v[getattr(c, k)] = [c.cid]
                    else:
                        v[getattr(c, k)].append(c.cid)
            for r in map(self._convert_row_to_relation, self._conn.get_columns(self.relation_table_name, self.relation_columns)):
                self.rids.add(r.rid)
                self.rid2relation_cache[r.rid] = r
                # handle another cache
                for k, v in self.partial2rids_cache.items():
                    if getattr(r, k) not in v:
                        v[getattr(r, k)] = [r.rid]
                    else:
                        v[getattr(r, k)].append(r.rid)
            for p in map(self._convert_row_to_concept_instance_pair, self._conn.get_columns(self.concept_instance_pair_table_name, self.concept_instance_pair_columns)):
                self.eids.append(p.eid)
                # handle another cache
                if p.cid not in self.cid2eid_pattern_scores:
                    self.cid2eid_pattern_scores[p.cid] = [(p.eid, p.pattern, p.score)]
                else:
                    self.cid2eid_pattern_scores[p.cid].append((p.eid, p.pattern, p.score))
                if p.eid not in self.eid2cid_scores:
                    self.eid2cid_scores[p.eid] = [(p.eid p.score)]
                else:
                    self.eid2cid_scores[p.eid].append((p.cid, p.score))
        else:
            for x in self._conn.get_columns(self.concept_table_name, ["_id"]):
                self.cids.add(x["_id"])
            for x in self._conn.get_columns(self.relation_table_name, ["_id"]):
                self.rids.add(x["_id"])
            for x in self._conn.get_columns(self.concept_instance_pair_table_name, ["eid"]):
                self.eids.add(x["eid"])

    def close(self):
        self._conn.close()
        self.eids.clear()
        self.rids.clear()
        self.eids.clear()
        self.cid2concept_cache.clear()
        self.rid2relation_cahce.clear()
        self.cid2eid_pattern_scores.clear()
        self.eid2cid_scores.clear()
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
        row["info"] = json.dumps(concept.to_dict().decode("utf-8"))
        return row

    def _convert_row_to_concept(self, row):
        concept = ASERConcept().from_dict(json.loads(row["info"].decode("utf-8")))
        concept.cid = row["_id"]
        return concept

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
        concepts = list(map(self._convert_row_to_concept, self._conn.select_rows(self.concept_table_name, cids, self.concept_columns)))
        for concept in concepts:
            if concept:
                self.cid2concept_cache[concept.cid] = concept
                # It seems not to need to append
                # if self.mode == "cache":
                #     for k, v in self.partial2cids_cache.items():
                #         if concept.get(k) in v:
                #             v[concept.get(k)].append(concept.cid)
                # elif self.mode == "memory":
                #     for k, v in self.partial2cids_cache.items():
                #         if concept.get(k) not in v:
                #             v[concept.get(k)] = [concept.cid]
                #         else:
                #             v[concept.get(k)].append(concept.cid)
        return concepts

    def _update_concept(self, concept):
        # update db
        # update_op = self._conn.get_update_op(["frequency"], "+")
        # row = self._convert_concept_to_row(concept)
        # self._conn.update_row(self.concept_table_name, row, update_op, ["frequency"])

        # updata cache
        # if self.mode == "insert":
        #     return None  # don"t care
        # updated_concept = self.cid2concept_cache.get(concept.cid, None)
        # if updated_concept:  # self.mode == "memory" or hit in cache
        #     updated_concept.frequency += concept.frequency
        # else:  # self.mode == "cache" and miss in cache
        #     updated_concept = self._get_concept_and_store_in_cache(concept.cid)
        
        # TODO: Add frequency
        if self.mode == "insert":
            return None   # don"t care
        return concept

    def _update_concepts(self, concepts):
        # update db
        # update_op = self._conn.get_update_op(["frequency"], "+")
        # rows = list(map(self._convert_concept_to_row, concepts))
        # self._conn.update_rows(self.concept_table_name, rows, update_op, ["frequency"])

        # update cache
        # if self.mode == "insert":
        #     return [None] * len(concepts)  # don"t care
        # updated_concepts = []
        # missed_indices = []
        # missed_cids = []
        # for idx, concept in enumerate(concepts):
        #     if concept.cid not in self.cids:
        #         updated_concepts.append(None)
        #     updated_concept = self.cid2concept_cache.get(concept.cid, None)
        #     updated_concepts.append(updated_concept)
        #     if updated_concept:
        #         updated_concept.frequency += concept.frequency
        #     else:
        #         missed_indices.append(idx)
        #         missed_cids.append(concept.cid)
        # for idx, updated_concept in enumerate(self._get_concepts_and_store_in_cache(missed_cids)):
        #     updated_concepts[missed_indices[idx]] = updated_concept

        # TODO: Add frequency
        if self.mode == "insert":
            return [None] * len(concepts) # don"t care
        return updated_concepts
        
    def insert_concept(self, concept, eventuality=None, score=None):
        if concept.cid not in self.cids:
            concept = self._insert_concept(concept)
            if eventuality is not None:
                self.insert_concept_instance_pair(concept, eventuality, score)
        else:
            concept = self._update_concept(concept)
            if eventuality is not None:
                self.insert_concept_instance_pair(concept, eventuality, score)
        return concept

    def insert_concepts(self, concepts, eventualities=None, scores=None):
        results = []
        new_indices = []
        existing_indices = []
        for idx, concept in enumerate(concepts):
            if concept.cid not in self.cids:
                new_indices.append(idx)
                results.append(concept)
            else:
                existing_indices.append(idx)
                results.append(None)
        if len(new_indices):
            new_concepts = [concepts[idx] for idx in new_indices]
            self._insert_concepts(new_concepts)
            if eventualities is not None:
                new_eventualities = [eventualities[idx] for idx in new_indices]
                new_scores = [scores[idx] for idx in new_indices]
                self.insert_concept_instance_pairs(new_concepts, new_eventualities, new_scores)
        if len(existing_indices):
            existing_concepts = [concepts[idx] for idx in existing_indices]
            for idx, updated_concept in enumerate(self._update_concepts(existing_concepts)):
                results[existing_indices[idx]] = updated_concept
            if eventualities is not None:
                existing_eventualities = [eventualities[idx] for idx in existing_indices]
                existing_scores = [scores[idx] for idx in existing_indices]
                self.insert_concept_instance_pairs(existing_concepts, existing_eventualities, existing_scores)
        return results

    def get_exact_match_concept(self, concept):
        """
        concept can be ASERConcept, Dictionary, str
        """
        if isinstance(concept, ASERConcept):
            cid = concept.cid
        elif isinstance(concept, dict):
            cid = concept["cid"]
        elif isinstance(concept, str):
            cid = concept
        else:
            raise ValueError("Error: concept should be an instance of ASERConcept, a dictionary, or a cid.")

        if cid not in self.cids:
            return None
        exact_match_concept = self.cid2concept_cache.get(cid, None)
        if not exact_match_concept:
            exact_match_concept = self._get_concept_and_store_in_cache(cid)
        return exact_match_concept

    def get_exact_match_concepts(self, concepts):
        """
        concepts can be ASERConcepts, Dictionaries, strs
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
        assert len(bys) == len(keys)
        for i in range(len(bys)-1, -1, -1):
            if bys[i] not in self.concept_columns:
                bys.pop(i)
                keys.pop(i)
        if len(bys) == 0:
            return []
        # cache = None
        # by_index = -1
        # for k in ["words", "skeleton_words", "verbs"]:
        #     if k in bys and k in self.partial2cids_cache:
        #         cache = self.partial2cids_cache[k]
        #         by_index = bys.index(k)
        #         break
        # if cache:
        #     if keys[by_index] in cache:
        #         key_match_concepts = [self.cid2concept_cache[cid] for cid in cache[keys[by_index]]]
        #     else:
        #         if self.mode == "memory":
        #             return []
        #         key_cache = []
        #         key_match_concepts = list(map(self._convert_row_to_concept, 
        #             self._conn.get_rows_by_keys(self.concept_table_name, [bys[by_index]], [keys[by_index]], self.concept_columns)))
        #         for key_match_concept in key_match_concepts:
        #             if key_match_concept.cid not in self.cid2concept_cache:
        #                 self.cid2concept_cache[key_match_concept.cid] = key_match_concept
        #             key_cache.append(key_match_concept.cid)
        #         cache[keys[by_index]] = key_cache
        #     for i in range(len(bys)):
        #         if i == by_index:
        #             continue
        #         key_match_concepts = list(filter(lambda x: x[bys[i]] == keys[i], key_match_concepts))
        #     if order_bys:
        #         key_match_concepts.sort(key=operator.itemgetter(*order_bys), reverse=reverse)
        #     if top_n:
        #         key_match_concepts = key_match_concepts[:top_n]
        #     return key_match_concepts
        return list(map(self._convert_row_to_concept, 
            self._conn.get_rows_by_keys(self.concept_table_name, bys, keys, self.concept_columns, order_bys=order_bys, reverse=reverse, top_n=top_n)))

    def get_concept_given_str(self, concept_str):
        cid = ASERConcept.generate_cid(concept_str)
        return self.get_exact_match_concept(cid)

    def get_concepts_given_strs(self, concept_strs):
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
        return Relation(row["hid"], row["tid"], {r: cnt for r, cnt in row.items() if isinstance(cnt, float) and cnt > 0.0})

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
        relation can be Relation, Dictionary, str, (ASERConcept, ASERConcept), (str, str)
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
                raise ValueError("Error: relation should be (an instance of ASERConcept, an instance of ASERConcept) or (hid, tid).")
        else:
            raise ValueError("Error: relation should be an instance of Relation, a dictionary, rid, (an instance of ASERConcept, an instance of ASERConcept), or (hid, tid).")
        
        if rid not in self.rids:
            return None
        exact_match_relation = self.rid2relation_cache.get(rid, None)
        if not exact_match_relation:
            exact_match_relation = self._get_relation_and_store_in_cache(rid)
        return exact_match_relation

    def get_exact_match_relations(self, relations):
        """
        relations can be Relations, Dictionaries, strs, [(ASERConcept, ASERConcept), ...], [(str, str), ...]
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
                if isinstance(relations[0][0], ASERConcept) and isinstance(relations[0][1], ASERConcept):
                    rids = [Relation.generate_rid(relation[0].cid, relation[1].cid) for relation in relations]
                elif isinstance(relations[0][0], str) and isinstance(relations[0][1], str):
                    rids = [Relation.generate_rid(relation[0], relation[1]) for relation in relations]
                else:
                    raise ValueError("Error: relations should be [(an instance of ASERConcept, an instance of ASERConcept), ...] or [(hid, tid), ...].")
            else:
                raise ValueError("Error: relations should be instances of Relation, dictionaries, rids, [(an instance of ASERConcept, an instance of ASERConcept), ...], or [(hid, tid), ...].")

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
    KG (ConceptInstancePairs)
    """
    def _convert_concept_instance_pair_to_row(self, concept, eventuality, score):
        c_i_pair = ASERConceptInstancePair(concept.cid, eventuality.eid, score)
        row = OrderedDict{"_id": c_i_pair.pid, "cid": c_i_pair.cid, "eid": c_i_pair.eid, "score": c_i_pair.score}
        return row

    def _convert_row_to_concept_instance_pair(self, row):
        return ASERConceptInstancePair(row["cid"], row["eid"], row["score"])

    def _insert_concept_instance_pair(self, concept, eventuality, score):
        row = self._convert_concept_instance_pair_to_row(concept, eventuality, score)
        self._conn.insert_row(self.concept_instance_pair_table_name, row)
        if self.mode == "insert":
            self.eids.add(eventuality.eid)
        elif self.mode == "cache":
            self.eids.add(eventuality.eid)
            if concept.cid in self.cid2eid_pattern_scores:
                self.cid2eid_pattern_scores[concept.cid].append((eventuality.eid, eventuality.pattern, score))
            if eventuality.eid in self.eid2cid_scores:
                self.eid2cid_scores[eventuality.eid].append((concept.cid, score))
        elif self.mode != "memory":
            self.eids.add(eventuality.eid)
            if concept.cid not in self.cid2eid_pattern_scores:
                self.cid2eid_pattern_scores[concept.cid] = [(eventuality.eid, eventuality.pattern, score)]
            else:
                self.cid2eid_pattern_scores[concept.cid].append(eventuality.eid, eventuality.pattern, score)
            if eventuality.eid not in self.eid2cid_scores:
                self.eid2cid_scores[eventuality.eid] = [(concept.cid, score)]
            else:
                self.eid2cid_scores[eventuality.eid].append(concept.cid, score)
        return self._convert_row_to_concept_instance_pair(row)

    def _insert_concept_instance_pairs(self, concepts, eventualities, scores):
        rows = list(map(self._convert_concept_instance_pair_to_row, zip(concepts, eventualities, scores)))
        self._conn.insert_rows(self.concept_instance_pair_table_name, rows)
        if self.mode == "insert":
            for eventuality in eventualities:
                self.eids.add(eventuality.eid)
        elif self.mode == "cache":
            for concept, eventuality, score in zip(concepts, eventualities, scores):
                self.eids.add(eventuality.id)
                if concept.cid in self.cid2eid_pattern_scores:
                    self.cid2eid_pattern_scores[concept.cid].append((eventuality.eid, eventuality.pattern, score))
                if eventuality.eid in self.eid2cid_scores:
                    self.eid2cid_scores[eventuality.eid].append((concept.cid, score))
        elif self.mode == "memory":
            for concept, eventuality, score in zip(concepts, eventualities, scores):
                self.eids.add(eventuality.eid)
                if concept.cid not in self.cid2eid_pattern_scores:
                    self.cid2eid_pattern_scores[concept.cid] = [(eventuality.eid, eventuality.pattern, score)]
                else:
                    self.cid2eid_pattern_scores[concept.cid].append(eventuality.eid, eventuality.pattern, score)
                if eventuality.eid not in self.eid2cid_scores:
                    self.eid2cid_scores[eventuality.eid] = [(concept.cid, score)]
                else:
                    self.eid2cid_scores[eventuality.eid].append(concept.cid, score)
        return [self._convert_row_to_concept_instance_pair(row) for row in rows]

    def _update_concept_instance_pair(self, concept, eventuality, score):
        # update db
        update_op = self._conn.get_update_op(["score"], "+")
        row = self._convert_concept_instance_pair_to_row(concept, eventuality, score)
        self._conn.update_row(self.concept_instance_pair_table_name, row, update_op, ["score"])

        # updata cache
        updated_score = None
        if self.mode == "insert":
            return None  # don"t care
        cached_cid_scores = self.eid2cid_scores.get(eventuality.eid, None)
        if cached_cid_scores:
            for idx, cid_score in enumerate(cached_cid_scores):
                if concept.cid == cid_score[0]:
                    updated_score = cid_score[1]+score
                    cache_cid_scores[idx] = (cid_score[0], updated_score)
                    break
        cached_eid_pattern_scores = self.cid2eid_pattern_scores.get(concept.cid, None)
        if cached_eid_pattern_scores:
            for idx, eid_pattern_score in enumerate(cached_eid_pattern_scores):
                if eventuality.eid == eid_pattern_score[0]:
                    updated_score = eid_pattern_score[2]+score
                    cached_eid_pattern_scores[idx] = (eid_pattern_score[0], eid_pattern_score[1], updated_score)
                    break
        if updated_score is None:
            updated_score = self._conn.select_row(self.concept_instance_pair_table_name, row["_id"], ["score"])["score"]
        return ASERConceptInstancePair(concept.cid, eventuality.eid, updated_score)

    def _update_concept_instance_pairs(self, concepts, eventualities, scores):
        # update db
        update_op = self._conn.get_update_op(["score"], "+")
        rows = list(map(self._convert_concept_instance_pair_to_row, zip(concepts, eventualities, scores)))
        self._conn.update_rows(self.concept_instance_pair_table_name, rows, update_op, ["score"])

        # update cache
        if self.mode == "insert":
            return [None] * len(concept_instance_pairs)  # don"t care
        results = []
        updated_scores = []
        missed_indices = []
        for idx, (concept, eventuality, score) in enumerate(zip(concepts, eventualities, scores)):
            cached_cid_scores = self.eid2cid_scores.get(eventuality.eid, None)
            if cached_cid_scores:
                for idx, cid_score in enumerate(cached_cid_scores):
                    if concept.cid == cid_score[0]:
                        updated_score = cid_score[1]+score
                        cache_cid_scores[idx] = (cid_score[0], updated_score)
                        break
            cached_eid_pattern_scores = self.cid2eid_pattern_scores.get(concept.cid, None)
            if cached_eid_pattern_scores:
                for idx, eid_pattern_score in enumerate(cached_eid_pattern_scores):
                    if eventuality.eid == eid_pattern_score[0]:
                        updated_score = eid_pattern_score[2]+score
                        cached_eid_pattern_scores[idx] = (eid_pattern_score[0], eid_pattern_score[1], updated_score)
                        break
            if updated_score is None:
                missed_indices.append(idx)
                updated_scores.append(None)
            else:
                updated_scores.append(updated_score)
        if len(missed_indices):
            for idx, updated_row in enumerate(self._conn.select_rows(self.concept_instance_pair_table_name, row["_id"], ["score"])):
                updated_scores[missed_indices[idx]] = updated_row["score"]
        return [ASERConceptInstancePair(concept.cid, eventuality.eid, updated_score) for concept, eventuality, updated_score in zip(concepts, eventualities, updated_scores)]

    def insert_concept_instance_pair(self, concept, eventuality, score):
        if concept.cid in self.cids and eventuality.eid in self.eids:
            return self._update_concept_instance_pair(concept, eventuality, score)
        else:
            return self._insert_concept_instance_pair(concept, eventuality, score)

    def insert_concept_instance_pairs(self, concepts, eventualities, scores):
        results = [None] * len(concepts)
        new_indices = []
        new_concepts, new_eventualities, new_scores = [], [], []
        existing_indices = []
        existing_concepts, existing_eventualities, existing_scores = [], [], []
        for idx, (concept, eventuality, score) in enumerate(concepts, eventualities, scores):
            if concept.cid in self.cids and eventuality.eid in self.eids:
                existing_concepts.append(concept)
                existing_eventualities.append(eventuality)
                existing_scores.append(score)
                existing_indices.append(idx)
            else:
                new_concepts.append(concept)
                new_eventualities.append(eventuality)
                new_scores.append(score)
                new_indices.append(idx)
        if len(new_indices):
            for idx, new_pair in enumerate(self._insert_concept_instance_pairs(new_concepts, new_eventualities, new_scores)):
                results[new_indices[idx]] = new_pair
        if len(existing_indices):
            for idx, updated_pair in enumerate(self._update_concept_instance_pairs(new_concepts, new_eventualities, new_scores)):
                results[existing_indices[idx]] = updated_pair
        return results

    def get_eventualities_given_concept(self, concept):
        """
        concept can be ASERConcept, Dictionary, or cid
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
            raise ValueError("Error: concept should be an instance of ASERConcept, a dictionary, or a cid.")
        
        cached_eid_pattern_scores = self.cid2eid_pattern_scores.get(cid, None)
        if cached_eid_pattern_scores:
            eids = [eid_pattern_score[0] for eid_pattern_score in cached_eid_pattern_scores]
            patterns = [eid_pattern_score[1] for eid_pattern_score in cached_eid_pattern_scores]
            scores = [eid_pattern_score[2] for eid_pattern_score in cached_eid_pattern_scores]
        else:
            eid_pattern_scores = self._conn.get_rows_by_keys(self.concept_instance_pair_table_name, bys=["cid"], keys=[cid], columns=["eid", "pattern", "score"])
            eids = [eid_pattern_score["eid"] for eid_pattern_score in eid_pattern_scores]
            patterns = [eid_pattern_score["pattern"] for eid_pattern_score in eid_pattern_scores]
            scores = [eid_pattern_score["score"] for eid_pattern_score in eid_pattern_scores]
        # eventualities = self.get_exact_match_eventualities(eids) # not implement
        eventualities = eids
        return list(zip(eventualities, patterns, scores))

    def get_concepts_given_eventuality(self, eventuality):
        """
        eventuality can be Eventuality, Dictionary, or eid
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
            raise ValueError("Error: concept should be an instance of Eventuality, a dictionary, or a eid.")
        
        cached_cid_scores = self.eid2cid_scores.get(eid, None)
        if cached_cid_scores:
            cids = [cid_score[0] for cid_score in cached_cid_scores]
            scores = [cid_score[1] for cid_score in cached_cid_scores]
        else:
            cid_scores = self._conn.get_rows_by_keys(self.concept_instance_pair_table_name, bys=["eid"], keys=[eid], columns=["cid", "score"])
            cids = [cid_score["cid"] for cid_score in cid_scores]
            scores = [cid_score["score"] for cid_score in cid_scores]
        concepts = self.get_exact_match_concepts(cids)
        return list(zip(concepts, scores))

    """
    Additional apis
    """
    def get_related_concepts(self, concept):
        """
        concept can be ASERConcept, Dictionary, or cid
        """
        if self.mode == "insert":
            return []
        if isinstance(concept, ASERConcept):
            cid = concept.cid
        elif isinstance(concept, dict):
            cid = concept["cid"]
        elif isinstance(concept, str):
            cid = concept_id
        else:
            raise ValueError("Error: concept should be an instance of ASERConcept, a dictionary, or a cid.")
        
        # cid == hid
        results = []
        if self.mode == "memory":
            if "hid" in self.partial2rids_cache:
                related_rids = self.partial2rids_cache["hid"].get(cid, list())
                related_relations = self.get_exact_match_relations(related_rids)
            else:
                related_relations = self.get_relations_by_keys(bys=["hid"], keys=[cid])
            tids = [x["tid"] for x in related_relations]
            t_concepts = self.get_exact_match_concepts(tids)
        elif self.mode == "cache":
            if "hid" in self.partial2rids_cache:
                if cid in self.partial2rids_cache["hid"]: # hit
                    related_rids = self.partial2rids_cache["hid"].get(cid, list())
                    related_relations = self.get_exact_match_relations(related_rids)
                    tids = [x["tid"] for x in related_relations]
                    t_concepts = self.get_exact_match_concepts(tids)
                else: # miss
                    related_relations = self.get_relations_by_keys(bys=["hid"], keys=[cid])
                    tids = [x["tid"] for x in related_relations]
                    t_concepts = self.get_exact_match_concepts(tids)
                    # update cache
                    self.partial2rids_cache["hid"][cid] = [x["_id"] for x in related_relations]
            else:
                related_relations = self.get_relations_by_keys(bys=["hid"], keys=[cid])
                tids = [x["tid"] for x in related_relations]
                t_concepts = self.get_exact_match_concepts(tids)
        return sorted(zip(t_concepts, related_relations), key=lambda x: x[1].relations["Co_Occurrence"])
