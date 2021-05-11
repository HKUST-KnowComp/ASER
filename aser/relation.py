import hashlib
import pprint
from .object import JsonSerializedObject

relation_senses = [
    "Precedence", "Succession", "Synchronous",
    "Reason", "Result",
    "Condition", "Contrast", "Concession",
    "Conjunction", "Instantiation", "Restatement",
    "ChosenAlternative", "Alternative", "Exception",
    "Co_Occurrence"
]


class Relation(JsonSerializedObject):
    """ ASER Relation

    """
    def __init__(self, hid="", tid="", relations=None):
        """

        :param hid: the unique eid to the head eventuality or conceptualied eventuality
        :type hid: str
        :param tid: the unique eid to the tail eventuality or conceptualied eventuality
        :type tid: str
        :param relations: the corresponding relations
        :type relations: Union[None, Dict[str, float], aser.relation.Relation]
        """

        self.hid = hid
        self.tid = tid
        self.rid = Relation.generate_rid(self.hid, self.tid)

        self.relations = dict()
        self.update(relations)

    @staticmethod
    def generate_rid(hid, tid):
        """

        :param hid: the unique eid to the head eventuality or conceptualied eventuality
        :type hid: str
        :param tid: the unique eid to the tail eventuality or conceptualied eventuality
        :type tid: str
        :return: the unique rid to the pair
        :rtype: str
        """

        key = hid + "$" + tid
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def to_triplets(self):
        """ Convert a relation to triplets

        :return: a list of triplets
        :rtype: List[Tuple[str, str]]
        """

        triplets = []
        for r in sorted(self.relations.keys()):
            triplets.extend([(self.hid, r, self.tid)] * int(self.relations[r]))
        return triplets

    def update(self, x):
        """  Update the relation ('s frequency)

        :param x: the given relation
        :type x: Union[Dict[str, float], Tuple[str], aser.relation.Relation]
        :return: the updated relation
        :rtype: aser.relation.Relation
        """

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
            else:
                raise ValueError("Error: the input of Relation.update is invalid.")
        return self

    def __str__(self):
        return pprint.pformat(self.to_dict())

    def __repr__(self):
        return "(%s, %s, %s)" % (self.hid, self.tid, self.relations)