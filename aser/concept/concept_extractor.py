from itertools import combinations
import os
from aser.concept import ASERConcept, seedConcept, ProbaseConcept

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
        concept_score_pairs = self.conceptualize_from_skeleton(
            eventuality.skeleton_words, eventuality.pattern)
        for concept, score in concept_score_pairs:
            concept.instances.append((eventuality.eid, eventuality.pattern))
        return concept_score_pairs

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

        concept_score_pairs = [(ASERConcept(words=concept_str, instances=list()), score)
                    for concept_str, score in concept_strs]
        return concept_score_pairs

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
