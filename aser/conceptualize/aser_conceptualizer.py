import heapq
from collections import defaultdict
from itertools import combinations
from ..concept import ASERConcept, ProbaseConcept


class BaseASERConceptualizer(object):
    """ Base ASER eventuality conceptualizer to conceptualize eventualities

    """
    def __init__(self):
        pass

    def close(self):
        """ Close the ASER Conceptualizer safely

        """
        pass

    def conceptualize(self, eventuality):
        """ Conceptualize an eventuality

        :param eventuality: an eventuality
        :type eventuality: aser.eventuality.Eventuality
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[aser.concept.ASERConcept, float]]
        """

        raise NotImplementedError


class SeedRuleASERConceptualizer(BaseASERConceptualizer):
    """ ASER eventuality conceptualizer based on rules and NERs

    """
    def __init__(self, **kw):
        super().__init__()
        self.selected_ners = frozenset(
            [
                "TIME", "DATE", "DURATION", "MONEY", "PERCENT", "NUMBER", "COUNTRY", "STATE_OR_PROVINCE", "CITY",
                "NATIONALITY", "PERSON", "RELIGION", "URL"
            ]
        )
        self.seed_concepts = frozenset([self._render_ner(ner) for ner in self.selected_ners])

        self.person_pronoun_set = frozenset(
            ["he", "she", "i", "him", "her", "me", "woman", "man", "boy", "girl", "you", "we", "they"]
        )
        self.pronouns = self.person_pronoun_set | frozenset(['it'])

    def conceptualize(self, eventuality):
        """ Conceptualization based on rules and NERs given an eventuality

        :param eventuality: an eventuality
        :type eventuality: aser.eventuality.Eventuality
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[aser.concept.ASERConcept, float]]
        """

        concept_strs = self.conceptualize_from_text(eventuality.phrases, eventuality.phrases_ners)
        return [(" ".join(concept_strs), 1.0)]

    def conceptualize_from_text(self, words, ners):
        """ Conceptualization based on rules and NERs given a word list an a ner list

        :param words: a word list
        :type words: List[str]
        :param ners: a ner list
        :type ners: List[str]
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[aser.concept.ASERConcept, float]]
        """

        output_words = list()
        ners_dict = {ner: dict() for ner in self.selected_ners}
        for word, ner in zip(words, ners):
            if ner in self.selected_ners:
                if word not in ners_dict[ner]:
                    ners_dict[ner][word] = len(ners_dict[ner])
                output_words.append(self._render_ner(ner) + "%d" % ners_dict[ner][word])
            elif word in self.person_pronoun_set:
                if word not in ners_dict["PERSON"]:
                    ners_dict["PERSON"][word] = len(ners_dict["PERSON"])
                output_words.append(self._render_ner("PERSON") + "%d" % ners_dict["PERSON"][word])
            else:
                output_words.append(word)
        return output_words

    def is_seed_concept(self, word):
        return word in self.seed_concepts

    def is_pronoun(self, word):
        return word in self.pronouns

    def _render_ner(self, ner):
        return "__" + ner + "__"


class ProbaseASERConceptualizer(BaseASERConceptualizer):
    """ ASER eventuality conceptualizer based on Probase and NERs

    """
    def __init__(self, probase_path=None, probase_topk=3):
        super().__init__()
        self.seed_conceptualizer = SeedRuleASERConceptualizer()
        self.probase = ProbaseConcept(probase_path)
        self.probase_topk = probase_topk

    def close(self):
        """ Close the ASER Conceptualizer safely

        """
        del self.probase
        self.probase = None

    def conceptualize(self, eventuality):
        """ Conceptualization use probase given an eventuality

        :param eventuality: an eventuality
        :type eventuality:  aser.eventuality.Eventuality
        :return: a list of (conceptualized eventuality, score) pair
        :rtype: List[Tuple[aser.concept.ASERConcept, float]]
        """
        concept_after_seed_rule = self.seed_conceptualizer.conceptualize_from_text(
            eventuality.skeleton_phrases, eventuality.skeleton_phrases_ners
        )
        concept_strs = self._get_probase_concepts(concept_after_seed_rule, eventuality.skeleton_pos_tags)
        if not concept_strs and concept_after_seed_rule != " ".join(eventuality.skeleton_phrases):
            concept_strs = [(concept_after_seed_rule, 1.0)]

        concept_score_pairs = [
            (ASERConcept(words=concept_str, instances=list()), score) for concept_str, score in concept_strs
        ]
        return concept_score_pairs

    def _get_probase_concepts(self, words, pos_tags):
        assert len(words) == len(pos_tags)

        word2indices = defaultdict(list)
        for idx, word in enumerate(words):
            word2indices[word].append(idx)

        word2concepts = dict()
        for i in range(len(pos_tags)):
            word = words[i]
            tag = pos_tags[i]

            if tag.startswith("NN"):
                if self.seed_conceptualizer.is_seed_concept(word) or self.seed_conceptualizer.is_pronoun(word):
                    continue
                elif word not in word2concepts:
                    concepts = self.probase.conceptualize(word, score_method="likelihood")
                    if concepts:
                        concept_set = set()
                        valid_indices = list()
                        for idx, (tmp_concept, score) in enumerate(concepts):
                            tmp = tmp_concept.replace(" ", "-")
                            if tmp not in concept_set:
                                valid_indices.append(idx)
                                concept_set.add(tmp)
                            if len(valid_indices) >= self.probase_topk:
                                break
                        word2concepts[word] = \
                            [(concepts[idx][0].replace(" ", "-"), concepts[idx][1]) for idx in valid_indices]
                    else:
                        continue

        matched_words = list(word2concepts.keys())
        replace_word_tuples = list()
        for i in range(1, len(word2concepts) + 1):
            replace_word_tuples.extend(list(combinations(matched_words, i)))

        output_words_heap = list()
        max_len = self.probase_topk ** self.probase_topk
        pre_min_score = 1.0
        min_score = -1.0
        pre_comb_len = 0
        comb_len = 1
        for word_tuples in replace_word_tuples:
            tmp_words_list = [(1.0, words)]
            for word in word_tuples:
                new_tmp_words_list = list()
                # can be further optimized...
                for prob, tmp_words in tmp_words_list:
                    for concept, c_prob in word2concepts[word]:
                        _tmp_words = tmp_words[:]
                        for idx in word2indices[word]:
                            _tmp_words[idx] = concept
                        new_tmp_words_list.append((prob * c_prob, _tmp_words))
                del tmp_words_list
                tmp_words_list = new_tmp_words_list

            for tmp in tmp_words_list:
                if len(output_words_heap) >= max_len:
                    tmp = heapq.heappushpop(output_words_heap, tmp)
                else:
                    heapq.heappush(output_words_heap, tmp)
                if min_score < tmp[0]:
                    min_score = tmp[0]
            comb_len = len(word_tuples)
            if pre_min_score == min_score and pre_comb_len + 1 < comb_len and len(output_words_heap) >= max_len:
                break
            if pre_min_score != min_score:
                pre_min_score = min_score
                pre_comb_len = comb_len

        output_words_list = [heapq.heappop(output_words_heap)[::-1] for i in range(len(output_words_heap))][::-1]
        return output_words_list
