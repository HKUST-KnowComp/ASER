import hashlib
try:
    import ujson as json
except:
    import json
import pprint
from aser.base import JsonSerializedObject

relation_senses = [
    'Precedence', 'Succession', 'Synchronous',
    'Reason', 'Result',
    'Condition', 'Contrast', 'Concession',
    'Conjunction', 'Instantiation', 'Restatement', 
    'ChosenAlternative', 'Alternative', 'Exception',
    'Co_Occurrence']

class Relation(JsonSerializedObject):
    def __init__(self, hid=None, tid=None, relations=None):
        self.hid = hid if hid else ""
        self.tid = tid if tid else ""
        self.rid = Relation.generate_rid(self.hid, self.tid)
        
        self.relations = dict()
        self.update(relations)

    @staticmethod
    def generate_rid(hid, tid):
        key = hid + "$" + tid
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def to_triples(self):
        triples = []
        for r in sorted(self.relations.keys()):
            triples.extend([(self.hid, r, self.tid)] * int(self.relations[r]))
        return triples

    def update(self, x):
        if x is not None:
            if isinstance(x, dict):
                for r, cnt in x.items():
                    if r not in self.relations:
                        self.relations[r] = cnt
                    else:
                        self.relations[r] += cnt
            elif isinstance(x, (list, tuple)):
                # cnt = 1.0/len(x) if len(x) > 0 else 0.0
                cnt = 1.0
                for r in x:
                    if r not in self.relations:
                        self.relations[r] = cnt
                    else:
                        self.relations[r] += cnt
            elif isinstance(x, Relation):
                if self.hid == x.hid and self.tid == x.tid:
                    for r, cnt in x.relations.items():
                        if r not in self.relations:
                            self.relations[r] = cnt
                        else:
                            self.relations[r] += cnt

    def __str__(self):
        repr_dict = {
            "rid": self.rid,
            "hid": self.hid,
            "tid": self.tid,
            "relations": self.relations.__str__()
        }
        return pprint.pformat(repr_dict)

    def __repr__(self):
        return "(%s, %s, %s)" % (self.hid, self.tid, self.relations)