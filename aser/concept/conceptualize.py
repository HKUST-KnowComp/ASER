import copy
import pickle
from tqdm import tqdm
from pbconcept.conceptualize import ProbaseConcept
from aser.concept import ASERConcept, seedConcept
from aser.relation import relation_senses


class ASERConceptDB(object):
    def __init__(self, db_path=None, source=None, probase_path=None, probase_topk=10):
        self.source = source
        self.id2concepts = dict()
        self.concept_count = dict()
        self.concept_to_instances = dict()
        self.instance_to_concept = dict()
        self.concept_relations = dict()
        self.concept_to_related_concepts = dict()
        if db_path:
            self.load(db_path)
        else:
            if source == "probase":
                self.probase = ProbaseConcept(probase_path)
                self.probase_topk = probase_topk
            else:
                raise NotImplementedError

    def save(self, db_path):
        with open(db_path, "wb") as f:
            pickle.dump(self.__dict__, f)

    def load(self, db_path):
        with open(db_path, "rb") as f:
            index_files = pickle.load(f)
        for key, val in index_files.items():
            self.__setattr__(key, val)

    def get_concept(self, concept_str):
        cid = ASERConcept.generate_cid(concept_str)
        exact_match_concept = self.get_exact_match_concept(cid)
        if exact_match_concept is not None:
            return exact_match_concept
        else:
            return ASERConcept(words=concept_str.split(" "),
                               instances=list(),
                               source=self.source)

    def get_concepts(self, concept_strs):
        return [(self.get_concept(concept_str), score)
                for concept_str, score in concept_strs]

    def get_exact_match_concept(self, cid):
        if cid in self.id2concepts:
            return ASERConcept(words=self.id2concepts[cid].split(" "),
                               instances=self.concept_to_instances[cid],
                               source=self.source)
        else:
            return None

    def get_exact_match_concepts(self, concept_ids):
        return [self.get_exact_match_concept(cid) for cid in concept_ids]

    def get_exact_match_instance(self, eventuality):
        pass

    def insert_instance(self, eventuality, concept_list):
        for concept, _ in concept_list:
            if concept.cid not in self.id2concepts:
                self.id2concepts[concept.cid] = concept.to_str()
                self.concept_to_instances[concept.cid] = list()
            if eventuality.eid not in self.concept_to_instances[concept.cid]:
                self.concept_to_instances[concept.cid].append(eventuality.eid)
            if concept.cid not in self.concept_count:
                self.concept_count = [0.0, 0.0]
            self.concept_count[0] += 1
            self.concept_count[1] += eventuality.frequency
        self.instance_to_concept[eventuality.eid] = \
            [(c.cid, score) for c, score in concept_list]

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

    def build_concept_db_from_aser_kg(self, aser_kg_conn):
        print("Building Concepts")
        for i, eid in tqdm(enumerate(aser_kg_conn.eids)):
            event = aser_kg_conn.get_exact_match_event(eid)
            concepts = self.conceptualize(event)
            self.insert_instance(event, concepts)
        print("Building Concepts relations")
        for h_concept_id in tqdm(self.id2concepts):
            instances = self.concept_to_instances[h_concept_id]
            for h_eid in instances:
                relations = aser_kg_conn.get_relations_by_keys(
                    bys=["heid"], keys=[h_eid])
                for rel in relations:
                    t_event = aser_kg_conn.get_exact_match_event(rel.teid)
                    t_concepts = self.conceptualize(t_event)
                    for rel_sense, count in rel.relations.items():
                        for t_concept, prob in t_concepts:
                            self.insert_relation(
                                h_concept_id, t_concept.cid, rel_sense, count * prob)
        self.build_concept_to_related_concepts()

    def build_concept_db_from_lower_level(self):
        pass

    def conceptualize(self, eventuality):
        if eventuality.eid in self.instance_to_concept:
            concept_ids, scores = zip(*self.instance_to_concept[eventuality.eid])
            concepts = list(zip(self.get_exact_match_concepts(concept_ids), scores))
            return concepts
        else:
            concept_strs = self._conceptualize_from_raw(eventuality)
            return self.get_concepts(concept_strs)

    def instantiate(self, concept):
        pass

    def _conceptualize_from_raw(self, eventuality):
        concept_after_seed_rule = self._get_seed_concepts(eventuality)
        if self.source == "probase":
            concept_strs = self._get_probase_concepts(concept_after_seed_rule, eventuality.pattern)
        else:
            raise NotImplementedError
        return concept_strs

    def _get_seed_concepts(self, eventuality):
        output_words = list()
        persons = dict()
        for word in eventuality.skeleton_words:
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

        return " ".join(output_words)

    def _get_probase_concepts(self, skeleton_words, patterns):
        words, patterns = skeleton_words.split(" "), \
                          patterns.split('-')
        output_words_list = [[[], 1.0]]
        for i in range(len(patterns)):
            if i >= len(words):
                break
            word = words[i]
            pattern = patterns[i]
            if pattern == 's' or pattern == 'o':
                if seedConcept.is_seed_concept(word) or seedConcept.is_pronoun(word):
                    for output_words, _ in output_words_list:
                        output_words.append(word)
                else:
                    concepts = self.probase.conceptualize(word, score_method="likelihood")
                    if concepts:
                        new_output_words_list = list()
                        for output_words, prob in output_words_list:
                            for concept, c_prob in concepts[:self.probase_topk]:
                                tmp_output_words = copy.deepcopy(output_words)
                                tmp_output_words.append(concept)
                                new_output_words_list.append(
                                    [tmp_output_words, prob * c_prob])
                            del output_words
                        del output_words_list
                        output_words_list = new_output_words_list
                    else:
                        for output_words, _ in output_words_list:
                            output_words.append(word)
            else:
                for output_words, _ in output_words_list:
                    output_words.append(word)
        for i in range(len(output_words_list)):
            tmp = output_words_list[i][0]
            output_words_list[i][0] = " ".join(tmp)
            del tmp
        output_words_list.sort(key=lambda x: x[1])
        return output_words_list