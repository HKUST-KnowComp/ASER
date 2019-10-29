import hashlib
try:
    import ujson as json
except:
    import json
from aser.base import JsonSerializedObject

relation_senses = [
    'Precedence', 'Succession', 'Synchronous',
    'Reason', 'Result',
    'Condition', 'Contrast', 'Concession',
    'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception',
    'Co_Occurrence']

class Relation(JsonSerializedObject):
    def __init__(self, hid=None, tid=None, relations=None):
        self.hid = hid if hid else ""
        self.tid = tid if tid else ""
        self.rid = Relation.generate_rid(self.hid, self.tid)
        
        self.relations = dict()
        self.update_relations(relations)

    @classmethod
    def generate_rid(cls, hid, tid):
        key = hid + "$" + tid
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def update_relations(self, x):
        if x is not None:
            if isinstance(x, dict):
                for r, cnt in x.items():
                    if r not in self.relations:
                        self.relations[r] = cnt
                    else:
                        self.relations[r] += cnt
            elif isinstance(x, (list, tuple)):
                for r in x:
                    if r not in self.relations:
                        self.relations[r] = 1.0
                    else:
                        self.relations[r] += 1.0
            elif isinstance(x, Relation):
                if self.hid == x.hid and self.tid == x.tid:
                    for r, cnt in x.relations.items():
                        if r not in self.relations:
                            self.relations[r] = cnt
                        else:
                            self.relations[r] += cnt