from itertools import combinations
from aser.concept import ASERConcept, ProbaseConcept


class BaseConceptExtractor(object):
    def __init__(self):
        pass

    def conceptualize(self, eventuality):
        raise NotImplementedError


class ASERSeedConceptExtractor(BaseConceptExtractor):
    """ Conceptualization based on rules and NERs
    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        self.selected_ners = frozenset([
            "TIME", "DATE", "DURATION", "MONEY", "PERCENT", "NUMBER",
            "COUNTRY", "STATE_OR_PROVINCE", "CITY",
            "NATIONALITY", "PERSON", "RELIGION",
            "URL"])
        self.seed_concepts = frozenset([self._render_ner(ner) for ner in self.selected_ners])

        self.person_pronoun_set = frozenset(
            ["he", "she", "i", "him", "her", "me", "woman", "man", "boy", "girl", "you", "we", "they"])
        self.pronouns = self.person_pronoun_set | frozenset(['it'])

    def conceptualize(self, eventuality):
        concept_strs = self.conceptualize_from_text(eventuality.phrases, eventuality.phrases_ners)
        return [(" ".join(concept_strs), 1.0)]

    def conceptualize_from_text(self, words, ners):
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


class ASERProbaseConceptExtractor(BaseConceptExtractor):
    def __init__(self, probase_path=None, probase_topk=None):
        super().__init__()
        self.seed_concept_extractor = ASERSeedConceptExtractor()
        self.probase = ProbaseConcept(probase_path)
        self.probase_topk = probase_topk

    def conceptualize(self, eventuality):
        """ Conceptualization use probase given a eventuality

        :type eventuality: aser.eventuality.Eventuality
        :param eventuality: `Eventuality` class object
        :return: a list of (concept, score) pair
        """
        concept_after_seed_rule = self.seed_concept_extractor.conceptualize_from_text(
            eventuality.skeleton_phrases, eventuality.skeleton_phrases_ners)
        concept_strs = self._get_probase_concepts(
            concept_after_seed_rule, eventuality.skeleton_pos_tags)
        if not concept_strs and concept_after_seed_rule != " ".join(eventuality.skeleton_phrases):
            concept_strs = [(concept_after_seed_rule, 1.0)]

        concept_score_pairs = [(ASERConcept(words=concept_str, instances=list()), score)
                    for concept_str, score in concept_strs]
        return concept_score_pairs

    def _get_probase_concepts(self, skeleton_words, skeleton_pos_tags):
        words, pos_tags = skeleton_words, skeleton_pos_tags
        matched_probase_concepts = dict()

        for i in range(len(pos_tags)):
            if i >= len(words):
                break
            word = words[i]
            tag = pos_tags[i]
            if tag.startswith("NN"):
                if self.seed_concept_extractor.is_seed_concept(word) \
                        or self.seed_concept_extractor.is_pronoun(word):
                    continue
                else:
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
                        matched_probase_concepts[i] = \
                            [(concepts[idx][0].replace(" ", "-"), concepts[idx][1]) for idx in valid_indices]
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

        for i, (output_words, score) in enumerate(output_words_list):
            output_words_list[i] = [[word.replace(" ", "-") for word in output_words], score]

        output_words_list.sort(key=lambda x: x[1], reverse=True)
        return output_words_list


class ASERConceptExtractor(object):
    def __init__(self, method="probase", *args, **kwargs):
        """
        :type method: str
        :param method: "probase" or "seed"
        """
        assert method in ["probase", "seed"]
        if method == "seed":
            self.concept_extractor = ASERSeedConceptExtractor(*args, **kwargs)
        elif method == "probase":
            self.concept_extractor = ASERProbaseConceptExtractor(*args, **kwargs)
        else:
            raise NotImplementedError

    def conceptualize(self, eventuality):
        """ Conceptualization given a eventuality

        :type eventuality: aser.eventuality.Eventuality
        :param eventuality: `Eventuality` class object
        :return: a list of (concept, score) pair
        """
        return self.concept_extractor.conceptualize(eventuality)
