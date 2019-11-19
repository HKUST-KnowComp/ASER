import hashlib
import re
import pprint
from aser.base import JsonSerializedObject
from collections import Counter

class SeedConcept(object):
    def __init__(self):
        self.person = "__PERSON__"
        self.url = "__URL__"
        self.digit = "__DIGIT__"
        self.year = "__YEAR__"
        self.person_pronoun_set = frozenset(
            ["he", "she", "i", "him", "her", "me", "woman", "man", "boy", "girl", "you", "we", "they"])
        self.pronouns = self.person_pronoun_set | frozenset(['it'])
        self.url_pattern = r'((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\.([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*'

    def check_is_person(self, word):
        return word in self.person_pronoun_set

    def check_is_year(self, word):
        if not word.isdigit() or len(word) != 4:
            return False
        d = int(word)
        return 1600 <= d <= 2100

    def check_is_digit(self, word):
        return word.isdigit()

    def check_is_url(self, word):
        if re.match(self.url_pattern, word):
            return True
        else:
            return False

    def is_seed_concept(self, word):
        return word in self.__dict__.values()

    def is_pronoun(self, word):
        return word in self.pronouns


seedConcept = SeedConcept()


class ASERConcept(JsonSerializedObject):
    def __init__(self, words=None, instances=None):
        """

        :type words: list
        :type instances: list
        :param words: list of word of concept
        :param instances: list of (eid, pattern) s, ...
        """
        super().__init__()
        self.words = words
        self.instances = instances
        self.cid = ASERConcept.generate_cid(self.__str__())

    @classmethod
    def generate_cid(cls, concept_str):
        return hashlib.sha1(concept_str.encode('utf-8')).hexdigest()

    @property
    def pattern(self):
        cnter = Counter([t[1] for t in self.instances])
        return cnter.most_common(1)[0][0]

    def __str__(self):
        return " ".join(self.words)

    def __repr__(self):
        return " ".join(self.words)

    def to_str(self):
        return self.__str__()

    def instantiate(self, kg_conn=None):
        if kg_conn:
            eventualities = kg_conn.get_exact_match_events(
                [t[0] for t in self.instances])
            return eventualities
        else:
            return self.instances

class ASERConceptInstancePair(JsonSerializedObject):
    def __init__(self, cid=None, eid=None, score=None):
        """

        :type words: list
        :type instances: list
        :param words: list of word of concept
        :param instances: list of (eid, pattern) s, ...
        """
        super().__init__()
        self.cid = cid
        self.eid = eid
        self.score = score
        self.pid = ASERConceptInstancePair.generate_pid(self.__str__())

    @classmethod
    def generate_pid(cls, concept_str):
        key = self.cid + "$" + self.eid
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def __str__(self):
        repr_dict = {
            "pid": self.pid,
            "cid": self.cid,
            "eid": self.eid,
            "score": self.score
        }
        return pprint.pformat(repr_dict)

    def __repr__(self):
        return self.__str__()

    def to_str(self):
        return self.__str__()