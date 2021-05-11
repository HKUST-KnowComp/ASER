import hashlib
import pprint
import time
import pickle
from tqdm import tqdm
from collections import Counter
from .object import JsonSerializedObject


class ASERConcept(JsonSerializedObject):
    """ ASER Conceptualied Eventuality

    """
    def __init__(self, words=None, instances=None):
        """

        :param words: the word list of a concept
        :type words: List[str]
        :param instances: a list of (eid, pattern, score)
        :type instances: List[Tuple[str, str, float]]
        """

        super().__init__()
        self.words = words if words else ""
        self.instances = instances if instances else []
        self.cid = ASERConcept.generate_cid(self.__str__())

    @staticmethod
    def generate_cid(concept_str):
        """ Generate the cid to a concept

        :param concept_str: concept representation (words connected by " ")
        :type concept_str: List[str]
        :return: the corresponding unique cid
        :rtype: str
        """

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

    def instantiate(self, kg_conn=None):
        """ Retrieve the instances that are associated with this concept

        :param kg_conn: an KG connection to ASER
        :type kg_conn: aser.database.kg_connection.ASERKGConnection
        :return: a list of (eid, pattern, score)
        :rtype: List[Tuple[str, str, float]]
        """

        if kg_conn:
            eventualities = kg_conn.get_exact_match_eventualities(
                [t[0] for t in self.instances])
            return eventualities
        else:
            return self.instances

class ASERConceptInstancePair(JsonSerializedObject):
    def __init__(self, cid="", eid="", pattern="unknown", score=0.0):
        """

        :param cid: the unique cid to the conceptualized eventuality
        :type cid: str
        :param eid: the unique eid to the eventuality
        :type eid: str
        :param pattern: the corresponding pattern
        :type pattern: str
        :param score: the conceptualization probability
        :type score: float
        """

        super().__init__()
        self.cid = cid
        self.eid = eid
        self.pattern = pattern
        self.score = score
        self.pid = ASERConceptInstancePair.generate_pid(cid, eid)

    @staticmethod
    def generate_pid(cid, eid):
        """ Generate the pid to a pair

        :param cid: the unique cid to the conceptualized eventuality
        :type cid: str
        :param eid: the unique eid to the eventuality
        :type eid: str
        :return: the unique pid to the pair
        :rtype: str
        """
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


class ProbaseConcept(object):
    """ Copied from https://github.com/ScarletPan/probase-concept

    """
    def __init__(self, data_concept_path=""):
        """

        :param data_concept_path: Probase .txt file path
        :type data_concept_path: str
        """
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
        print("[probase-conceptualize] Loading Probase files...")
        with open(data_concept_path) as f:
            triplet_lines = [line.strip() for line in f]

        print("[probase-conceptualize] Building index...")
        for line in tqdm(triplet_lines):
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
        print("[probase-conceptualize] Loading data finished in {:.2f} s".format(time.time() - st))


    def conceptualize(self, instance, score_method="likelihood"):
        """ Conceptualize the given instance

        :param instance:  the given instance
        :type instance: str
        :param score_method: the method to compute sscores ("likelihood" or "pmi")
        :type score_method: str
        :return: a list of (concept, score)
        :rtype: List[Tuple[aser.concept.ProbaseConcept, float]]
        """

        if instance not in self.instance2idx:
            return []
        instance_idx = self.instance2idx[instance]
        instance_freq = self.get_instance_freq(instance_idx)
        concept_list = self.instance_inverted_list[instance_idx]
        rst_list = list()
        for concept_idx, co_occurrence in concept_list:
            if score_method == "pmi":
                score = co_occurrence / self.get_concept_freq(concept_idx) / instance_freq
            elif score_method == "likelihood":
                score = co_occurrence / instance_freq
            else:
                raise NotImplementedError
            rst_list.append((self.idx2concept[concept_idx], score))
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def instantiate(self, concept):
        """ Retrieve all instances of a concept

        :param concept: the given concept
        :type concept: str
        :return: a list of instances
        :rtype: List[Tuple[str, float]]
        """

        if concept not in self.concept2idx:
            return []
        concept_idx = self.concept2idx[concept]
        rst_list = [(self.idx2instance[idx], freq) for idx, freq
                    in self.concept_inverted_list[concept_idx]]
        rst_list.sort(key=lambda x: x[1], reverse=True)
        return rst_list

    def get_concept_chain(self, instance, max_chain_length=5):
        """ Conceptualize the given instance in a chain

        :param instance: the given instance
        :type instance: str
        :param max_chain_length: the maximum length of the chain
        :type max_chain_length: int (default = 5)
        :return: a chain that contains concepts
        :rtype: List[str]
        """

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
        """ Get the frequency of a concept

        :param concept: the given concept
        :type concept: str
        :return: the corresponding frequency
        :rtype: float
        """

        if isinstance(concept, str):
            if concept not in self.concept2idx:
                return 0
            concept = self.concept2idx[concept]
        elif isinstance(concept, int):
            if concept not in self.idx2concept:
                return 0
        return sum([t[1] for t in self.concept_inverted_list[concept]])

    def get_instance_freq(self, instance):
        """ Get the frequency of an instance

        :param instance: the given instance
        :type instance: str
        :return: the corresponding frequency
        :rtype: float
        """

        if isinstance(instance, str):
            if instance not in self.instance2idx:
                return 0
            instance = self.instance2idx[instance]
        elif isinstance(instance, int):
            if instance not in self.idx2instance:
                return 0
        return sum([t[1] for t in self.instance_inverted_list[instance]])

    def save(self, file_name):
        """

        :param file_name: the file name to save the probase concepts
        :type file_name: str
        """

        with open(file_name, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, file_name):
        """

        :param file_name: the file name to load the probase concepts
        :type file_name: str
        """

        with open(file_name, "rb") as f:
            tmp_dict = pickle.load(f)
        for key, val in tmp_dict.items():
            self.__setattr__(key, val)

    @property
    def concept_size(self):
        return len(self.concept2idx)

    @property
    def instance_size(self):
        return len(self.instance2idx)