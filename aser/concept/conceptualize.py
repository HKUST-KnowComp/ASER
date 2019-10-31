from itertools import combinations
import os
import pickle
from pbconcept.conceptualize import ProbaseConcept
from aser.concept import ASERConcept, seedConcept
from aser.relation import relation_senses


class ASERConceptExtractor(object):
    def __init__(self, source="probase", probase_path=None, probase_topk=None):
        self.source = source
        if source=="probase":
            self.probase = ProbaseConcept(probase_path)
            self.probase_topk = probase_topk
        else:
            self.probase = None
            self.probase_topk = None

    def conceptualize(self, eventuality):
        """ Conceptualization given a eventuality

        :type eventuality: aser.eventuality.Eventuality
        :param eventuality: `Eventuality` class object
        :return: concept
        """
        return self.conceptualize_from_skeleton(
            eventuality.skeleton_words, eventuality.pattern)

    def conceptualize_from_skeleton(self, skeleton_words, pattern):
        """ Conceptualization given a skeleton words and pattern

        :type skeleton_words: list of str
        :type pattern: str
        :param skeleton_words: skeleton words of eventuality or concept_str.split(" ")
        :param pattern: eventuality pattern
        :return: concept
        """
        concept_after_seed_rule = self._get_seed_concepts(skeleton_words)
        if self.source == "probase":
            concept_strs = self._get_probase_concepts(concept_after_seed_rule, pattern)
            if not concept_strs and concept_after_seed_rule != " ".join(skeleton_words):
                concept_strs = [(concept_after_seed_rule, 1.0)]
        else:
            raise NotImplementedError

        concepts = [(ASERConcept(words=concept_str, instances=list()), score)
                    for concept_str, score in concept_strs]
        return concepts

    def _get_seed_concepts(self, skeleton_words):
        output_words = list()
        persons = dict()
        for word in skeleton_words:
            if seedConcept.check_is_year(word):
                output_words.append(seedConcept.year)
            elif seedConcept.check_is_digit(word):
                output_words.append(seedConcept.digit)
            elif seedConcept.check_is_url(word):
                output_words.append(seedConcept.url)
            elif seedConcept.check_is_person(word):
                if word not in persons:
                    persons[word] = len(persons)
                output_words.append(seedConcept.person + "%d" % persons[word])
            else:
                output_words.append(word)

        if len(persons) == 1:
            for i, word in enumerate(output_words):
                if word.startswith(seedConcept.person):
                    output_words[i] = seedConcept.person

        return output_words

    def _get_probase_concepts(self, skeleton_words, pattern):
        words, patterns = skeleton_words, \
                          pattern.split('-')
        matched_probase_concepts = dict()

        for i in range(len(patterns)):
            if i >= len(words):
                break
            word = words[i]
            pattern = patterns[i]
            if pattern == 's' or pattern == 'o':
                if seedConcept.is_seed_concept(word) or seedConcept.is_pronoun(word):
                    continue
                else:
                    concepts = self.probase.conceptualize(word, score_method="likelihood")
                    if concepts:
                        matched_probase_concepts[i] = \
                            [(t[0].replace(" ", "-"), t[1]) for t in concepts[:self.probase_topk]]
                    else:
                        continue

        matched_indices = list(matched_probase_concepts.keys())

        replace_indices_tuples = list()
        for i in range(1, len(matched_indices) + 1):
            replace_indices_tuples.extend(list(combinations(matched_indices, i)))

        output_words_list = list()
        for indices_tuple in replace_indices_tuples:
            tmp_words_list = [[words, 1.0]]
            for idx in indices_tuple:
                new_tmp_words_list = list()
                for tmp_words, prob in tmp_words_list:
                    for concept, c_prob in matched_probase_concepts[idx]:
                        _tmp_words = tmp_words[:]
                        _tmp_words[idx] = concept
                        new_tmp_words_list.append([_tmp_words, prob * c_prob])
                del tmp_words_list
                tmp_words_list = new_tmp_words_list
            output_words_list.extend(tmp_words_list)

        output_words_list.sort(key=lambda x: x[1], reverse=True)
        return output_words_list


class ASERConceptDB(object):
    def __init__(self, db_path=None, overwrite=False):
        self.db_path = db_path
        self.id2concepts = dict()
        self.concept_to_instances = dict()
        self.instance_to_concept = dict()
        self.concept_relations = dict()
        self.concept_to_related_concepts = dict()
        if os.path.exists(db_path) and not overwrite:
            self.load(db_path)

    def save(self):
        with open(self.db_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, db_path):
        with open(db_path, "rb") as f:
            index_files = pickle.load(f)
        for key, val in index_files.items():
            self.__setattr__(key, val)

    def insert_instance(self, eventuality, concept, score):
        if concept.cid not in self.id2concepts:
            self.id2concepts[concept.cid] = concept.to_str()
            self.concept_to_instances[concept.cid] = list()

        concept_instance_eids = [t[0] for t in self.concept_to_instances[concept.cid]]
        if eventuality.eid not in concept_instance_eids:
            self.concept_to_instances[concept.cid].append(
                (eventuality.eid, eventuality.pattern, score))

        if eventuality.eid not in self.instance_to_concept:
            self.instance_to_concept[eventuality.eid] = list()
        instance_concept_cids = [t[0] for t in self.instance_to_concept[eventuality.eid]]
        if concept.cid not in instance_concept_cids:
            self.instance_to_concept[eventuality.eid].append((concept.cid, score))

    def insert_relation(self, hcid, tcid, relation_sense, score):
        rid = hcid + "$" + tcid
        if rid not in self.concept_relations:
            self.concept_relations[rid] = dict()
            for rel_sense in relation_senses:
                self.concept_relations[rid][rel_sense] = 0.0
        self.concept_relations[rid][relation_sense] += score

    def build_concept_to_related_concepts(self):
        for rid in self.concept_relations:
            hcid, tcid = rid.split("$")
            if hcid not in self.concept_to_related_concepts:
                self.concept_to_related_concepts[hcid] = list()
            self.concept_to_related_concepts[hcid].append(tcid)

    def get_exact_match_concept(self, cid):
        if cid in self.id2concepts:
            return ASERConcept(words=self.id2concepts[cid].split(" "),
                               instances=self.concept_to_instances[cid])
        else:
            return None

    def get_exact_match_concepts(self, concept_ids):
        return [self.get_exact_match_concept(cid) for cid in concept_ids]

    def get_concepts_of_eventuality(self, eventuality):
        if eventuality.eid not in self.instance_to_concept:
            return []
        else:
            concept_ids = [cid for cid, _ in self.instance_to_concept[eventuality.eid]]
            scores = [score for _, score in self.instance_to_concept[eventuality.eid]]
            return list(zip(self.get_exact_match_concepts(concept_ids), scores))

    def get_concept_from_str(self, concept_str):
        cid = ASERConcept.generate_cid(concept_str)
        exact_match_concept = self.get_exact_match_concept(cid)
        if exact_match_concept is not None:
            return exact_match_concept
        else:
            return ASERConcept(words=concept_str.split(" "),
                               instances=list())

    def get_concepts_from_str(self, concept_strs):
        return [(self.get_concept_from_str(concept_str), score)
                for concept_str, score in concept_strs]

    def get_related_concepts(self, concept):
        rst_list = list()
        if concept.cid not in self.concept_to_related_concepts:
            return rst_list
        related_concept_ids = self.concept_to_related_concepts[concept.cid]
        hcid = concept.cid
        for tcid in related_concept_ids:
            rid = hcid + "$" + tcid
            for rel_sense, score in self.concept_relations[rid].items():
                if score > 0.0:
                    rst_list.append((self.get_exact_match_concept(tcid), rel_sense, score))
        rst_list.sort(key=lambda x: x[-1], reverse=True)
        return rst_list


class ASERConceptAPI(object):
    def __init__(self, concept_db, aser_concept_extractor , aser_kg_conn):
        self.aser_concept_extractor = aser_concept_extractor
        self.aser_concept_db = concept_db
        self.aser_kg_conn = aser_kg_conn

    def conceptualize(self, eventuality):
        concepts = self.aser_concept_db.get_concepts_of_eventuality(eventuality)
        if not concepts:
            tmp_concepts = self.aser_concept_extractor.conceptualize(eventuality)
            concepts = list()
            for i, (concept, prob) in enumerate(tmp_concepts):
                tmp = self.aser_concept_db.get_exact_match_concept(concept.cid)
                if tmp:
                    concepts.append((tmp, prob))
                else:
                    concepts.append((concept, prob))
        return concepts

    def instantiate(self, concept):
        tmp =  self.aser_concept_db.get_exact_match_concept(concept.cid)
        concept = tmp if tmp else concept
        return concept.instantiate(self.aser_kg_conn)

    def get_related_concepts(self, concept):
        return self.aser_concept_db.get_related_concepts(concept)