import itertools
import pickle
import random
import sys
import time
from tqdm import tqdm_notebook as tqdm
import json
from aser.database.db_API_skeleton_words import KG_Connection, preprocess_event
from aser.database.connection_v2 import relation_senses


class ASER():
    def __init__(self, db_path, inv_table_path):
        st = time.time()
        print("[ASER] Connecting KG...")
        self.db = KG_Connection(db_path=db_path)
        with open(inv_table_path, "rb") as f:
            self.inverted_table = pickle.load(f)
        print("[ASER] Finished in {:.2f} s...".format(time.time() - st))

    @staticmethod
    def get_aser_relation(binary_repr):
        rels = []
        for i, x in enumerate(binary_repr):
            if int(x) and i < 14:
                rels.append(relation_senses[i])
        return rels

    def inference(self, record, max_related_event_num=8, max_event_triple_num=20):
        # Matching events
        event_list = []
        if "post_activity" not in record:
            return []
        for act_type, activity in record["post_activity"]:
            event = preprocess_event(activity, pattern=act_type)
            event_list.append(event)
        matched_event_list = []
        for event in event_list:
            matched_event = self.db.get_exact_match_event(event)
            if matched_event:
                matched_event_list.append(dict(matched_event))

        # Find related events
        event_triples = []
        for event in matched_event_list:
            eid = event["_id"]
            if eid in self.inverted_table:
                related_events = self.inverted_table[eid][:max_related_event_num] \
                    if max_related_event_num > 0 else self.inverted_table[eid]
                for related_eid, rel_binary in related_events:
                    for rel in self.get_aser_relation(rel_binary):
                        event_triples.append(eid + "$" + rel + "$" + related_eid)

        return event_triples if len(event_triples) < max_event_triple_num \
            else random.sample(event_triples, max_event_triple_num)

    def report_coverage(self, records):
        covered_cnt = 0
        covered_events = set()
        for record in tqdm(records):
            event_triples = self.inference(record, -1, sys.maxsize)
            covered_cnt += len(event_triples) > 0
            for triple in event_triples:
                e1, *_ = triple.split("$")
                covered_events.add(e1)
        print("[ASER] Number of covered pair: ", covered_cnt)
        print("[ASER] Number of covered events: ", len(covered_events))


class OMCS():
    def __init__(self, omcs_path, rel_set=None):
        self.rel_set = rel_set
        self.verb2triple = dict()
        self.noun2triple = dict()
        self.key2triple = dict()
        st = time.time()
        print("[OMCS] Connecting KG...")
        self.build_db(omcs_path)
        print("[OMCS] Finished in {:.2f} s...".format(time.time() - st))

    def build_db(self, omcs_path):
        with open(omcs_path) as f:
            omcs_records = [json.loads(t) for t in f]
        for i, record in tqdm(enumerate(omcs_records)):
            e1, e2, r, _, e1_parsed, e2_parsed = record
            if self.rel_set and r not in self.rel_set:
                continue
            concept_id = e1 + "$" + r + "$" + e2
            if not e1_parsed or not e1_parsed[0]["parsed_relations"]:
                noun = e1.lower()
                if noun not in self.noun2triple:
                    self.noun2triple[noun] = []
                self.noun2triple[noun].append(concept_id)
                key = None
            else:
                nodes = []
                key = []
                have_verb = None
                for (node1, _, node2) in e1_parsed[0]['parsed_relations']:
                    nodes.extend([node1, node2])
                for node in nodes:
                    if node[-1] == "VB":
                        v = node[1]
                        if v not in self.verb2triple:
                            self.verb2triple[v] = []
                        self.verb2triple[v].append(concept_id)
                        key.append(v)
                        have_verb = True
                        break
                    if node[-1] == "NN":
                        n = node[1]
                        if n not in self.noun2triple:
                            self.noun2triple[n] = []
                        self.noun2triple[n].append(concept_id)
                        key.append(n)
                if have_verb:
                    key = tuple(sorted(set(e1.split())))
                else:
                    key = None
            if not key:
                continue
            if key not in self.key2triple:
                self.key2triple[key] = []
            self.key2triple[key].append(concept_id)

    def inference(self, record, method="exact", use_noun=False, max_related_event_num=8, max_event_triple_num=20):
        event_triples = []
        if method == "exact":
            record_tokens_set = set(record["post"].lower().split())
            for key in self.key2triple:
                key_token_set = set(key)
                if key_token_set & record_tokens_set == key_token_set:
                    if max_related_event_num > 0:
                        event_triples.extend(self.key2triple[key][:max_related_event_num])
                    else:
                        event_triples.extend(self.key2triple[key])
        else:
            for prased_res in record["post_parsed_relations"]:
                node_list = set()
                for node1, _, node2 in prased_res["parsed_relations"]:
                    node_list.update([tuple(node1), tuple(node2)])
                node_list = list(sorted(node_list, key=lambda x: x[0]))
                for _, token, tag in node_list:
                    if use_noun:
                        if tag == "NN" and token in self.noun2triple:
                            if max_related_event_num > 0:
                                event_triples.extend(self.noun2triple[token][:max_related_event_num])
                            else:
                                event_triples.extend(self.noun2triple[token])
                    if tag == "VB" and token in self.verb2triple:
                        if max_related_event_num > 0:
                            event_triples.extend(self.verb2triple[token][:max_related_event_num])
                        else:
                            event_triples.extend(self.verb2triple[token])
        return event_triples if len(event_triples) < max_event_triple_num \
            else random.sample(event_triples, max_event_triple_num)

    def report_coverage(self, records):
        covered_cnt = 0
        covered_events = set()
        for record in tqdm(records):
            event_triples = self.inference(record, "exact", False, -1, sys.maxsize)
            covered_cnt += len(event_triples) > 0
            for triple in event_triples:
                e1, *_ = triple.split("$")
                covered_events.add(e1)
        print("[OMCS] Number of covered pair: ", covered_cnt)
        print("[OMCS] Number of covered events: ", len(covered_events))


class KnowlyWood(object):
    def __init__(self, kw_path, rel_set):
        self.rel_set = rel_set
        self.verb2triple = dict()
        self.key2triple = dict()
        st = time.time()
        print("[KnowlyWood] Connecting KG...")
        self.build_db(kw_path)
        print("[KnowlyWood] Finished in {:.2f} s...".format(time.time() - st))

    def build_db(self, kw_path):
        def extract_verb(item):
            for word in item.split(";"):
                if "#v" in word:
                    return word.split("#")[0]

        with open(kw_path) as f:
            for _ in tqdm(range(10211391)):
                line = f.readline()
                e1, r, e2, n2 = line.strip().split("\t")
                if self.rel_set and r not in self.rel_set:
                    continue
                concept_id = e1 + "$" + r + "$" + e2
                verb = extract_verb(e1)
                if verb not in self.verb2triple:
                    self.verb2triple[verb] = []
                self.verb2triple[verb].append(concept_id)

                match_key = tuple([t.split("#")[0] for t in e1.split(";")])
                if match_key not in self.key2triple:
                    self.key2triple[match_key] = []
                self.key2triple[match_key].append(concept_id)

    def inference(self, record, method="exact", max_related_event_num=8, max_event_triple_num=20):
        event_triples = []
        if method == "exact":
            record_tokens = list(set(record["post"].lower().split()))
            record_keys = []
            for i in range(len(record_tokens)):
                for j in range(i + 1, len(record_tokens)):
                    record_keys.append((record_tokens[i], record_tokens[j]))
            for tmp in record_keys:
                key = tuple(sorted(set(tmp)))
                if key in self.key2triple:
                    if max_related_event_num > 0:
                        event_triples.extend(self.key2triple[key][:max_related_event_num])
                    else:
                        event_triples.extend(self.key2triple[key])
        else:
            for prased_res in record["post_parsed_relations"]:
                node_list = set()
                for node1, _, node2 in prased_res["parsed_relations"]:
                    node_list.update([tuple(node1), tuple(node2)])
                node_list = list(sorted(node_list, key=lambda x: x[0]))
                for _, token, tag in node_list:
                    if tag == "VB" and token in self.verb2triple:
                        if max_related_event_num > 0:
                            event_triples.extend(self.verb2triple[token][:max_related_event_num])
                        else:
                            event_triples.extend(self.verb2triple[token])
        return event_triples if len(event_triples) < max_event_triple_num \
        else random.sample(event_triples, max_event_triple_num)

    def report_coverage(self, records):
        covered_cnt = 0
        covered_events = set()
        for record in tqdm(records):
            event_triples = self.inference(record, "exact", -1, sys.maxsize)
            covered_cnt += len(event_triples) > 0
            for triple in event_triples:
                e1, *_ = triple.split("$")
                covered_events.add(e1)
        print("[KnowlyWood] Number of covered pair: ", covered_cnt)
        print("[KnowlyWood] Number of covered events: ", len(covered_events))