from itertools import chain
from copy import deepcopy
from .discourse_parser import ConnectiveExtractor, ArgumentPositionClassifier, \
    SSArgumentExtractor, PSArgumentExtractor, ExplicitSenseClassifier
from .rule import SEED_CONNECTIVE_DICT
from .utils import EMPTY_SENT_PARSED_RESULT
from ..relation import Relation, relation_senses


class BaseRelationExtractor(object):
    """ Base ASER relation rxtractor to extract relations

    """
    def __init__(self, **kw):
        pass

    def close(self):
        pass

    def __del__(self):
        self.close()

    def extract_from_parsed_result(
        self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
    ):
        """ Extract relations from the parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param para_eventualities: eventualities in the paragraph
        :type para_eventualities: List[aser.eventuality.Eventuality]
        :param output_format: which format to return, "Relation" or "triplet"
        :type output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted relations
        :rtype: Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

                [{'dependencies': [(1, 'nmod:poss', 0),
                                   (3, 'nsubj', 1),
                                   (3, 'aux', 2),
                                   (3, 'dobj', 5),
                                   (3, 'punct', 6),
                                   (5, 'nmod:poss', 4)],
                  'lemmas': ['my', 'army', 'will', 'find', 'you', 'boat', '.'],
                  'mentions': [],
                  'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O'],
                  'parse': '(ROOT (S (NP (PRP$ My) (NN army)) (VP (MD will) (VP (VB find) (NP '
                           '(PRP$ your) (NN boat)))) (. .)))',
                  'pos_tags': ['PRP$', 'NN', 'MD', 'VB', 'PRP$', 'NN', '.'],
                  'text': 'My army will find your boat.',
                  'tokens': ['My', 'army', 'will', 'find', 'your', 'boat', '.']},
                 {'dependencies': [(2, 'case', 0),
                                   (2, 'det', 1),
                                   (6, 'nmod:in', 2),
                                   (6, 'punct', 3),
                                   (6, 'nsubj', 4),
                                   (6, 'cop', 5),
                                   (6, 'ccomp', 9),
                                   (6, 'punct', 13),
                                   (9, 'nsubj', 7),
                                   (9, 'aux', 8),
                                   (9, 'iobj', 10),
                                   (9, 'dobj', 12),
                                   (12, 'amod', 11)],
                  'lemmas': ['in',
                             'the',
                             'meantime',
                             ',',
                             'I',
                             'be',
                             'sure',
                             'we',
                             'could',
                             'find',
                             'you',
                             'suitable',
                             'accommodation',
                             '.'],
                  'mentions': [],
                  'ners': ['O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O',
                           'O'],
                  'parse': '(ROOT (S (PP (IN In) (NP (DT the) (NN meantime))) (, ,) (NP (PRP '
                           "I)) (VP (VBP 'm) (ADJP (JJ sure) (SBAR (S (NP (PRP we)) (VP (MD "
                           'could) (VP (VB find) (NP (PRP you)) (NP (JJ suitable) (NNS '
                           'accommodations)))))))) (. .)))',
                  'pos_tags': ['IN',
                               'DT',
                               'NN',
                               ',',
                               'PRP',
                               'VBP',
                               'JJ',
                               'PRP',
                               'MD',
                               'VB',
                               'PRP',
                               'JJ',
                               'NNS',
                               '.'],
                  'text': "In the meantime, I'm sure we could find you suitable "
                          'accommodations.',
                  'tokens': ['In',
                             'the',
                             'meantime',
                             ',',
                             'I',
                             "'m",
                             'sure',
                             'we',
                             'could',
                             'find',
                             'you',
                             'suitable',
                             'accommodations',
                             '.']}],
                [[my army will find you boat],
                 [i be sure, we could find you suitable accommodation]]

                Output:

                [[],
                 [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
                 [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]]
        """

        raise NotImplementedError


class SeedRuleRelationExtractor(BaseRelationExtractor):
    """ ASER relation extractor based on rules to extract relations (for ASER v1.0)

    """
    def __init__(self, **kw):
        super().__init__(**kw)

    def extract_from_parsed_result(
        self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
    ):
        if output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Relation or triplet.")

        connective_dict = kw.get("connective_dict", SEED_CONNECTIVE_DICT)

        para_relations = list()
        for sent_parsed_result, eventualities in zip(parsed_result, para_eventualities):
            relations_in_sent = list()
            for head_eventuality in eventualities:
                for tail_eventuality in eventualities:
                    if not head_eventuality.position < tail_eventuality.position:
                        continue
                    heid = head_eventuality.eid
                    teid = tail_eventuality.eid
                    extracted_senses = self._extract_from_eventuality_pair_in_one_sentence(
                        connective_dict, sent_parsed_result, head_eventuality, tail_eventuality
                    )
                    if len(extracted_senses) > 0:
                        relations_in_sent.append(Relation(heid, teid, extracted_senses))
            para_relations.append(relations_in_sent)

        for i in range(len(parsed_result) - 1):
            eventualities1, eventualities2 = para_eventualities[i], para_eventualities[i + 1]
            relations_between_sents = list()
            if len(eventualities1) == 1 and len(eventualities2) == 1:
                s1_tokens, s2_tokens = parsed_result[i]["tokens"], parsed_result[i + 1]["tokens"]
                s1_eventuality, s2_eventuality = eventualities1[0], eventualities2[0]
                heid, teid = s1_eventuality.eid, s2_eventuality.eid
                extracted_senses = self._extract_from_eventuality_pair_in_two_sentence(
                    connective_dict, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
                )
                if len(extracted_senses) > 0:
                    relations_between_sents.append(Relation(heid, teid, extracted_senses))
            para_relations.append(relations_between_sents)

        if in_order:
            if output_format == "triplet":
                para_relations = [sorted(chain.from_iterable([r.to_triplets() for r in relations]))
                                  for relations in para_relations]
            return para_relations
        else:
            if output_format == "Relation":
                rid2relation = dict()
                for relation in chain(*para_relations):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deepcopy(relation)
                    else:
                        rid2relation[relation.rid].update(relation)
                relations = sorted(rid2relation.values(), key=lambda r: r.rid)
            elif output_format == "triplet":
                relations = sorted([r.to_triplets() for relations in para_relations for r in relations])
            return relations

    def _extract_from_eventuality_pair_in_one_sentence(
        self, connective_dict, sent_parsed_result, head_eventuality, tail_eventuality
    ):
        extracted_senses = ['Co_Occurrence']
        for sense in relation_senses:
            for connective_words in connective_dict[sense]:
                if self._verify_connective_in_one_sentence(
                    connective_words, head_eventuality, tail_eventuality, sent_parsed_result["dependencies"],
                    sent_parsed_result["tokens"]
                ):
                    extracted_senses.append(sense)
                    break
        return extracted_senses

    def _extract_from_eventuality_pair_in_two_sentence(
        self, connective_dict, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
    ):
        extracted_senses = list()
        for sense in relation_senses:
            for connective_words in connective_dict[sense]:
                if self._verify_connective_in_two_sentence(
                    connective_words, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
                ):
                    extracted_senses.append(sense)
                    break

        return extracted_senses

    def _verify_connective_in_one_sentence(
        self, connective_words, head_eventuality, tail_eventuality, sentence_dependencies, sentence_tokens
    ):
        def get_connective_position(connective_words):
            tmp_positions = list()
            for w in connective_words:
                tmp_positions.append(sentence_tokens.index(w))
            return sum(tmp_positions) / len(tmp_positions) if tmp_positions else 0.0

        # Connective Words need to be presented in sentence
        if set(connective_words) - set(sentence_tokens):
            return False
        # Connective phrase need to be presented in sentence
        connective_string = " ".join(connective_words)
        sentence_string = " ".join(sentence_tokens)
        if connective_string not in sentence_string:
            return False
        shrinked_dependencies = self._shrink_sentence_dependencies(
            head_eventuality._raw_dependencies, tail_eventuality._raw_dependencies, sentence_dependencies
        )
        if not shrinked_dependencies:
            return False
        found_advcl = False
        for (governor, dep, dependent) in shrinked_dependencies:
            if governor == '_H_' and dependent == '_T_' and 'advcl' in dep:
                found_advcl = True
                break
        if not found_advcl:
            return False
        connective_position = get_connective_position(connective_words)
        e1_position, e2_position = head_eventuality.position, tail_eventuality.position
        if 'instead' not in connective_words:
            if e1_position < connective_position < e2_position:
                return True
            else:
                return False
        else:
            if e1_position < e2_position < connective_position:
                return True
            else:
                return False

    def _verify_connective_in_two_sentence(
        self, connective_words, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens
    ):
        def get_connective_position():
            tmp_positions = list()
            for w in connective_words:
                if w in s1_tokens:
                    tmp_positions.append(s1_tokens.index(w))
                elif w in s2_tokens:
                    tmp_positions.append(s2_tokens.index(w) + len(s1_tokens))
            return sum(tmp_positions) / len(tmp_positions) if tmp_positions else 0.0

        sentence_tokens = s1_tokens + s2_tokens
        # Connective Words need to be presented in sentence
        if set(connective_words) - set(sentence_tokens):
            return False
        # Connective phrase need to be presented in sentence
        connective_string = " ".join(connective_words)
        sentence_string = " ".join(sentence_tokens)
        if connective_string not in sentence_string:
            return False
        connective_position = get_connective_position()
        e1_position, e2_position = s1_eventuality.position, \
                                   s2_eventuality.position + len(s1_tokens)
        if 'instead' not in connective_words:
            if e1_position < connective_position < e2_position and e2_position - e1_position < 10:
                return True
            else:
                return False
        else:
            if e1_position < e2_position < connective_position and e2_position - e1_position < 10:
                return True
            else:
                return False

    def _shrink_sentence_dependencies(self, head_dependencies, tail_dependencies, sentence_dependencies):
        head_nodes = set()
        for governor, _, dependent in head_dependencies:
            head_nodes.add(governor)
            head_nodes.add(dependent)
        tail_nodes = set()
        for governor, _, dependent in tail_dependencies:
            tail_nodes.add(governor)
            tail_nodes.add(dependent)
        if head_nodes & tail_nodes:
            return None

        new_dependencies = list()
        for governor, dep, dependent in sentence_dependencies:
            if governor in head_nodes:
                new_governor = '_H_'
            elif governor in tail_nodes:
                new_governor = '_T_'
            else:
                new_governor = governor
            if dependent in head_nodes:
                new_dependent = '_H_'
            elif dependent in tail_nodes:
                new_dependent = '_T_'
            else:
                new_dependent = dependent
            if new_governor != new_dependent:
                new_dependencies.append((new_governor, dep, new_dependent))
        return new_dependencies


class DiscourseRelationExtractor(BaseRelationExtractor):
    """ ASER relation extractor based on discourse parsing to extract relations (for ASER v2.0)

    """

    def __init__(self, **kw):
        super().__init__(**kw)
        self.conn_extractor = ConnectiveExtractor(**kw)
        self.argpos_classifier = ArgumentPositionClassifier(**kw)
        self.ss_extractor = SSArgumentExtractor(**kw)
        self.ps_extractor = PSArgumentExtractor(**kw)
        self.explicit_classifier = ExplicitSenseClassifier(**kw)

    def extract_from_parsed_result(
        self, parsed_result, para_eventualities, output_format="triplet", in_order=False, **kw
    ):
        if output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Relation or triplet.")

        similarity = kw.get("similarity", "simpson").lower()
        threshold = kw.get("threshold", 0.8)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Error: threshold should be between 0.0 and 1.0.")
        if similarity == "simpson":
            similarity_func = self._match_argument_eventuality_by_Simpson
        elif similarity == "jaccard":
            similarity_func = self._match_argument_eventuality_by_Jaccard
        elif similarity == "discourse":
            similarity_func = self._match_argument_eventuality_by_dependencies
        else:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Simpson or Jaccard.")

        syntax_tree_cache = kw.get("syntax_tree_cache", dict())

        len_sentences = len(parsed_result)
        if len_sentences == 0:
            if in_order:
                return [list()]
            else:
                return list()

        para_relations = [list() for _ in range(2 * len_sentences - 1)]

        # replace sentences that contains no eventuality with empty sentences
        filtered_parsed_result = list()
        for sent_idx, (sent_parsed_result, sent_eventualities) in enumerate(zip(parsed_result, para_eventualities)):
            if len(sent_eventualities) > 0:
                filtered_parsed_result.append(sent_parsed_result)
                relations_in_sent = para_relations[sent_idx]
                for head_e in sent_eventualities:
                    heid = head_e.eid
                    for tail_e in sent_eventualities:
                        if not head_e.position < tail_e.position:
                            continue
                        teid = tail_e.eid
                        relations_in_sent.append(Relation(heid, teid, ["Co_Occurrence"]))
            else:
                filtered_parsed_result.append(EMPTY_SENT_PARSED_RESULT)  # empty sentence
                # filtered_parsed_result.append(sent_parsed_result)

        connectives = self.conn_extractor.extract(filtered_parsed_result, syntax_tree_cache)
        SS_connectives, PS_connectives = self.argpos_classifier.classify(
            filtered_parsed_result, connectives, syntax_tree_cache
        )
        SS_connectives = self.ss_extractor.extract(filtered_parsed_result, SS_connectives, syntax_tree_cache)
        PS_connectives = self.ps_extractor.extract(filtered_parsed_result, PS_connectives, syntax_tree_cache)
        connectives = self.explicit_classifier.classify(
            filtered_parsed_result, SS_connectives + PS_connectives, syntax_tree_cache
        )
        connectives.sort(key=lambda x: (x["sent_idx"], x["indices"][0] if len(x["indices"]) > 0 else -1))

        for connective in connectives:
            conn_indices = connective.get("indices", None)
            arg1 = connective.get("arg1", None)
            arg2 = connective.get("arg2", None)
            sense = connective.get("sense", None)
            if conn_indices and arg1 and arg2 and (sense and sense != "None"):
                arg1_sent_idx = arg1["sent_idx"]
                arg2_sent_idx = arg2["sent_idx"]
                relation_list_idx = arg1_sent_idx if arg1_sent_idx == arg2_sent_idx else arg1_sent_idx + len_sentences
                relations = para_relations[relation_list_idx]
                sent_parsed_result1, sent_eventualities1 = parsed_result[arg1_sent_idx], para_eventualities[
                    arg1_sent_idx]
                sent_parsed_result2, sent_eventualities2 = parsed_result[arg2_sent_idx], para_eventualities[
                    arg2_sent_idx]
                arg1_eventualities = [e for e in sent_eventualities1 if \
                    similarity_func(sent_parsed_result1, arg1, e, threshold=threshold, conn_indices=conn_indices)]
                arg2_eventualities = [e for e in sent_eventualities2 if \
                    similarity_func(sent_parsed_result2, arg2, e, threshold=threshold, conn_indices=conn_indices)]
                cnt = 0.0
                if len(arg1_eventualities) > 0 and len(arg2_eventualities) > 0:
                    cnt = 1.0 / (len(arg1_eventualities) * len(arg2_eventualities))
                for e1 in arg1_eventualities:
                    heid = e1.eid
                    for e2 in arg2_eventualities:
                        teid = e2.eid
                        is_existed = False
                        for relation in relations:
                            if relation.hid == heid and relation.tid == teid:
                                relation.update({sense: cnt})
                                is_existed = True
                                break
                        if not is_existed:
                            relations.append(Relation(heid, teid, {sense: cnt}))

        if in_order:
            if output_format == "triplet":
                para_relations = [sorted(chain.from_iterable([r.to_triplets() for r in relations]))
                                  for relations in para_relations]
            return para_relations
        else:
            if output_format == "Relation":
                rid2relation = dict()
                for relation in chain(*para_relations):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deepcopy(relation)
                    else:
                        rid2relation[relation.rid].update(relation)
                relations = sorted(rid2relation.values(), key=lambda r: r.rid)
            elif output_format == "triplet":
                relations = sorted([r.to_triplets() for relations in para_relations for r in relations])
            return relations


    @staticmethod
    def _match_argument_eventuality_by_Simpson(sent_parsed_result, argument, eventuality, **kw):
        threshold = kw.get("threshold", 0.8)
        match = False
        if eventuality.raw_sent_mapping:
            argument_indices = set(argument["indices"])
            eventuality_indices = set(eventuality.raw_sent_mapping.values())
            try:
                Simpson = len(argument_indices &
                              eventuality_indices) / min(len(argument_indices), len(eventuality_indices))
                match = Simpson >= threshold
            except ZeroDivisionError:
                match = False
        else:
            argument_tokens = set([sent_parsed_result["lemmas"][idx].lower() for idx in argument["indices"]])
            eventuality_tokens = set(eventuality.words)
            try:
                Simpson = len(argument_tokens & eventuality_tokens) / min(len(argument_tokens), len(eventuality_tokens))
                match = Simpson >= threshold
            except ZeroDivisionError:
                match = False
        return match

    @staticmethod
    def _match_argument_eventuality_by_Jaccard(sent_parsed_result, argument, eventuality, **kw):
        threshold = kw.get("threshold", 0.8)
        match = False
        if eventuality.raw_sent_mapping:
            argument_indices = set(argument["indices"])
            eventuality_indices = set(eventuality.raw_sent_mapping.values())
            try:
                Jaccard = len(argument_indices & eventuality_indices) / len(argument_indices | eventuality_indices)
                match = Jaccard >= threshold
            except ZeroDivisionError:
                match = False
        else:
            argument_tokens = set([sent_parsed_result["lemmas"][idx].lower() for idx in argument["indices"]])
            eventuality_tokens = set(eventuality.words)
            try:
                Jaccard = len(argument_tokens & eventuality_tokens) / len(argument_tokens | eventuality_tokens)
                match = Jaccard >= threshold
            except ZeroDivisionError:
                match = False
        return match

    @staticmethod
    def _match_argument_eventuality_by_dependencies(sent_parsed_result, argument, eventuality, **kw):
        conn_indices = kw.get("conn_indices", list())
        match = False
        conn_indices = set(conn_indices)
        if eventuality.raw_sent_mapping:
            argument_indices = set(argument["indices"])
            eventuality_indices = set(eventuality.raw_sent_mapping.values())

            for (governor, dep, dependent) in sent_parsed_result["dependencies"]:
                # find the word linked to the connective
                if dependent in conn_indices and governor in argument_indices and governor in eventuality_indices:
                    match = True
                    break
                elif governor in conn_indices and dependent in argument_indices and dependent in eventuality_indices:
                    match = True
                    break
        else:
            argument_tokens = set([sent_parsed_result["lemmas"][idx].lower() for idx in argument["indices"]])
            eventuality_token_pos_tags = set(zip(eventuality.words, eventuality.pos_tags))

            argument_indices = set(argument["indices"])
            for (governor, dep, dependent) in sent_parsed_result["dependencies"]:
                # find the word linked to the connective
                if dependent in conn_indices and governor in argument_indices:
                    token_pos_tag = (
                        sent_parsed_result["lemmas"][governor].lower(), sent_parsed_result["pos_tags"][governor]
                    )
                    if token_pos_tag in eventuality_token_pos_tags:
                        match = True
                        break
                elif governor in conn_indices and dependent in argument_indices:
                    token_pos_tag = (
                        sent_parsed_result["lemmas"][dependent].lower(), sent_parsed_result["pos_tags"][dependent]
                    )
                    if token_pos_tag in eventuality_token_pos_tags:
                        match = True
                        break
        return match
