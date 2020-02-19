import hashlib
import pprint
import time
import pickle
from tqdm import tqdm
from aser.base import JsonSerializedObject
from collections import Counter


class ASERConcept(JsonSerializedObject):
    def __init__(self, words=None, instances=None):
        """

        :type words: list
        :type instances: list
        :param words: list of word of concept
        :param instances: list of (eid, pattern, score) s, ...
        """
        super().__init__()
        self.words = words if words else ""
        self.instances = instances if instances else []
        self.cid = ASERConcept.generate_cid(self.__str__())

    @staticmethod
    def generate_cid(concept_str):
        return hashlib.sha1(concept_str.encode('utf-8')).hexdigest()

    @property
    def pattern(self):
        if len(self.instances) > 0:
            cnter = Counter([t[1] for t in self.instances])
            return cnter.most_common(1)[0][0]
        else:
            return ""

    def __str__(self):
        return " ".join(self.words)

    def __repr__(self):
        return " ".join(self.words)

    def to_str(self):
        return self.__str__()

    def instantiate(self, kg_conn=None):
        if kg_conn:
            eventualities = kg_conn.get_exact_match_eventualities(
                [t[0] for t in self.instances])
            return eventualities
        else:
            return self.instances

class ASERConceptInstancePair(JsonSerializedObject):
    def __init__(self, cid=None, eid=None, pattern=None, score=None):
        """

        :type words: list
        :type instances: list
        :param words: list of word of concept
        :param instances: list of (eid, pattern) s, ...
        """
        super().__init__()
        self.cid = cid if cid else ""
        self.eid = eid if eid else ""
        self.pattern = pattern if pattern else ""
        self.score = score if score else 0
        self.pid = ASERConceptInstancePair.generate_pid(cid, eid)

    @staticmethod
    def generate_pid(cid, eid):
        key = cid + "$" + eid
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def __str__(self):
        repr_dict = {
            "pid": self.pid,
            "cid": self.cid,
            "eid": self.eid,
            "pattern": self.pattern,
            "score": self.score
        }
        return pprint.pformat(repr_dict)

    def __repr__(self):
        return self.__str__()

    def to_str(self):
        return self.__str__()


class ProbaseConcept(object):
    """
        Copied from https://github.com/ScarletPan/probase-concept
    """
    def __init__(self, data_concept_path=None):
        self.concept2idx = dict()
        self.idx2concept = dict()
        self.concept_inverted_list = dict()
        self.instance2idx = dict()
        self.idx2instance = dict()
        self.instance_inverted_list = dict()
        if data_concept_path:
            self._load_raw_data(data_concept_path)

    def _load_raw_data(self, data_concept_path):
        st = time.time()
        print("[probase-concept] Loading Probase files...")
        with open(data_concept_path) as f:
            triple_lines = [line.strip() for line in f]

        print("[probase-concept] Building index...")
        for line in tqdm(triple_lines):
            concept, instance, freq = line.split('\t')
            if concept not in self.concept2idx:
                self.concept2idx[concept] = len(self.concept2idx)
            concept_idx = self.concept2idx[concept]
            if instance not in self.instance2idx:
                self.instance2idx[instance] = len(self.instance2idx)
            instance_idx = self.instance2idx[instance]
            if concept_idx not in self.concept_inverted_list:
                self.concept_inverted_list[concept_idx] = list()
            self.concept_inverted_list[concept_idx].append((instance_idx, int(freq)))
            if instance_idx not in self.instance_inverted_list:
                self.instance_inverted_list[instance_idx] = list()
            self.instance_inverted_list[instance_idx].append((concept_idx, int(freq)))

        self.idx2concept = {val: key for key, val in self.concept2idx.items()}
        self.idx2instance = {val: key for key, val in self.instance2idx.items()}
        print("[probase-concept] Loading data finished in {:.2f} s".format(time.time() - st))


    def conceptualize(self, instance, score_method="likelihood"):
        """ Conceptualize given instance
        :type instance: str
        :type score_method: str
        :param instance: input instance such as "microsoft"
        :param score_method: "likelihood" or "pmi"
        :return:
        """
        if instance not in self.instance2idx:
            return []
        instance_idx = self.instance2idx[instance]
        instance_freq = self.get_instance_freq(instance_idx)
        concept_list = self.instance_inverted_list[instance_idx]
        rst_list = list()
        for concept_idx, co_occurrence in concept_list:
            if score_method == "pmi":
                score = co_occurrence / \
                        self.get_concept_freq(concept_idx) / \
                        instance_freq
            elif score_method == "likelihood":
                score = co_occurrence / instance_freq
            else:
                raise NotImplementedError
            rst_list.append((self.idx2concept[concept_idx], score))
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def instantiate(self, concept):
        """ Retrieve all instances of a concept
        :type concept: str
        :param concept: input concept such as "company"
        :return:
        """
        if concept not in self.concept2idx:
            return []
        concept_idx = self.concept2idx[concept]
        rst_list = [(self.idx2instance[idx], freq) for idx, freq
                    in self.concept_inverted_list[concept_idx]]
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def get_concept_chain(self, instance, max_chain_length=5):
        if instance in self.concept2idx:
            chain = [instance]
        else:
            chain = list()
        tmp_instance = instance
        while True:
            concepts = self.conceptualize(tmp_instance, score_method="likelihood")
            if concepts:
                chain.append(concepts[0][0])
            else:
                break
            if len(chain) >= max_chain_length:
                break
            tmp_instance = chain[-1]
        if chain and chain[0] != instance:
            return [instance] + chain
        else:
            return chain

    def get_concept_freq(self, concept):
        if isinstance(concept, str):
            if concept not in self.concept2idx:
                return 0
            concept = self.concept2idx[concept]
        elif isinstance(concept, int):
            if concept not in self.idx2concept:
                return 0
        return sum([t[1] for t in self.concept_inverted_list[concept]])

    def get_instance_freq(self, instance):
        if isinstance(instance, str):
            if instance not in self.instance2idx:
                return 0
            instance = self.instance2idx[instance]
        elif isinstance(instance, int):
            if instance not in self.idx2instance:
                return 0
        return sum([t[1] for t in self.instance_inverted_list[instance]])

    def save(self, saved_path):
        st = time.time()
        print("[probase-concept] Loading data to {}".format(saved_path))
        with open(saved_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print("[probase-concept] Saving data finished in {:.2f} s".format(time.time() - st))

    def load(self, load_path):
        st = time.time()
        print("[probase-concept] Loading data from {}".format(load_path))
        with open(load_path, "rb") as f:
            tmp_dict = pickle.load(f)
        for key, val in tmp_dict.items():
            self.__setattr__(key, val)
        print("[probase-concept] Loading data finished in {:.2f} s".format(time.time() - st))

    @property
    def concept_size(self):
        return len(self.concept2idx)

    @property
    def instance_size(self):
        return len(self.instance2idx)