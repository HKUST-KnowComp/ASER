import bisect
import subprocess
import numpy as np
import time
import os
try:
    import ujson as json
except:
    import json
from functools import partial
from itertools import chain
from pprint import pprint
from copy import deepcopy
from aser.extract.discourse_parser import ConnectiveExtractor, ArgumentPositionClassifier, \
    SSArgumentExtractor, PSArgumentExtractor, ExplicitSenseClassifier
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.extract.rule import SEED_CONNECTIVE_DICT
from aser.extract.utils import EMPTY_SENT_PARSED_RESULT


class BaseRelationExtractor(object):
    def __init__(self, **kw):
        pass

    def close(self):
        pass

    def __del__(self):
        self.close()

    def extract_from_parsed_result(self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw):
        """ This methods extract relations among extracted eventualities

            :type parsed_result: list
            :type para_eventualities: list
            :type output_format: str
            :type in_order bool
            :param parsed_result: a list of dict
            :param para_eventualities: a list of lists of `Eventuality` objects
            :param output_format: `Relation` or triple
            :param in_order: in order or out of order
            :return: a list of `Relation` objects, a list of triples, a list of lists of `Relation` objects, or a list of lists of triples

            .. highlight:: python
            .. code-block:: python

                Input Example:
                    [{'dependencies': [(1, 'nsubj', 0),
                                    (1, 'nmod:to', 3),
                                    (1, 'advcl:because', 7),
                                    (1, 'punct', 8),
                                    (3, 'case', 2),
                                    (7, 'mark', 4),
                                    (7, 'nsubj', 5),
                                    (7, 'cop', 6)],
                        'lemmas': ['I', 'go', 'to', 'lunch', 'because', 'I', 'be', 'hungry', '.'],
                        'pos_tags': ['PRP', 'VBP', 'TO', 'NN', 'IN', 'PRP', 'VBP', 'JJ', '.'],
                        'tokens': ['I', 'go', 'to', 'lunch', 'because', 'I', 'am', 'hungry', '.']}],
                    [
                        [Eventuatlity({'dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'I', 'PRP')),
                                                        ((2, 'hungry', 'JJ'), 'cop', (1, 'be', 'VBP'))],
                                        'eid': 'eae8741fad51a57e78092017def1b5cb4f620d7e',
                                        'pattern': 's-be-a',
                                        'pos_tags': ['PRP', 'VBP', 'JJ'],
                                        'skeleton_dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'I', 'PRP')),
                                                                ((2, 'hungry', 'JJ'), 'cop', (1, 'be', 'VBP'))],
                                        'skeleton_words': ['I', 'be', 'hungry'],
                                        'verbs': ['be'],
                                        'words': ['I', 'be', 'hungry']}),
                        Eventuatlity({'dependencies': [((1, 'go', 'VBP'), 'nsubj', (0, 'I', 'PRP')),
                                                       ((1, 'go', 'VBP'), 'nmod:to', (3, 'lunch', 'NN')),
                                                       ((3, 'lunch', 'NN'), 'case', (2, 'to', 'TO'))],
                                        'eid': '12b4aa577e56f2f5d96f4716bc97c633d6272ec4',
                                        'pattern': 's-v-X-o',
                                        'pos_tags': ['PRP', 'VBP', 'TO', 'NN'],
                                        'skeleton_dependencies': [((1, 'go', 'VBP'), 'nsubj', (0, 'I', 'PRP')),
                                                                ((1, 'go', 'VBP'), 'nmod:to', (3, 'lunch', 'NN')),
                                                                ((3, 'lunch', 'NN'), 'case', (2, 'to', 'TO'))],
                                        'skeleton_words': ['I', 'go', 'to', 'lunch'],
                                        'verbs': ['go'],
                                        'words': ['I', 'go', 'to', 'lunch']})
                        ]],
                    "Relation",
                    True
        """
        raise NotImplementedError

class SeedRuleRelationExtractor(BaseRelationExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)

    def extract_from_parsed_result(self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw):
        if output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Relation or triple.")

        connective_dict = kw.get("connective_dict", SEED_CONNECTIVE_DICT)

        para_relations = list()
        for sent_parsed_result, eventualities in zip(parsed_result, para_eventualities):
            relations_in_sent = list()
            for head_eventuality in eventualities:
                for tail_eventuality in eventualities:
                    if head_eventuality.position < tail_eventuality.position:
                        heid = head_eventuality.eid
                        teid = tail_eventuality.eid
                        extracted_senses = self._extract_from_eventuality_pair_in_one_sentence(
                            connective_dict, sent_parsed_result, head_eventuality, tail_eventuality)
                        relations_in_sent.append(Relation(heid, teid, extracted_senses))
            para_relations.append(relations_in_sent)

        for i in range(len(parsed_result) - 1):
            eventualities1, eventualities2 = para_eventualities[i], para_eventualities[i+1]
            relations_between_sents = list()
            if len(eventualities1) == 1 and len(eventualities2) == 1:
                s1_tokens, s2_tokens = parsed_result[i]["tokens"], parsed_result[i+1]["tokens"]
                s1_eventuality, s2_eventuality = eventualities1[0], eventualities2[0]
                heid, teid = s1_eventuality.eid, s2_eventuality.eid
                extracted_senses = self._extract_from_eventuality_pair_in_two_sentence(
                    connective_dict, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens)
                relations_between_sents.append(Relation(heid, teid, extracted_senses))
            para_relations.append(relations_between_sents)

        if in_order:
            if output_format == "Relation":
                return para_relations
            elif output_format == "triple":
                return [sorted(chain.from_iterable([r.to_triples() for r in relations])) \
                    for relations in para_relations]
        else:
            if output_format == "Relation":
                rid2relation = dict()
                for relation in chain(*para_relations):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deeocopy(relation)
                    else:
                        rid2relation[relation.rid].update_relations(relation)
                return sorted(rid2relation.values(), key=lambda r: r.rid)
            if output_format == "triple":
                return sorted([r.to_triples() for relations in para_relations for r in relations])
            

    def _extract_from_eventuality_pair_in_one_sentence(self, connective_dict, sent_parsed_result, head_eventuality, tail_eventuality):
        extracted_senses = ['Co_Occurrence']
        for sense in relation_senses:
            for connective_words in connective_dict[sense]:
                if self._verify_connective_in_one_sentence(
                    connective_words, head_eventuality, tail_eventuality,
                    sent_parsed_result["dependencies"],
                    sent_parsed_result["tokens"]):
                    extracted_senses.append(sense)
                    break
        return extracted_senses

    def _extract_from_eventuality_pair_in_two_sentence(self, connective_dict, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens):
        extracted_senses = list()
        for sense in relation_senses:
            for connective_words in connective_dict[sense]:
                if self._verify_connective_in_two_sentence(connective_words, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens):
                    extracted_senses.append(sense)
                    break

        return extracted_senses

    def _verify_connective_in_one_sentence(self, connective_words, head_eventuality, tail_eventuality, sentence_dependencies, sentence_tokens):
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
            head_eventuality._raw_dependencies,
            tail_eventuality._raw_dependencies,
            sentence_dependencies)
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


    def _verify_connective_in_two_sentence(self, connective_words, s1_eventuality, s2_eventuality, s1_tokens, s2_tokens):
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


    def _shrink_sentence_dependencies(self, head_dependencies, tail_dependencies,
                                      sentence_dependencies):
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
    def __init__(self, **kw):
        super().__init__(**kw)
        self.conn_extractor = ConnectiveExtractor(**kw)
        self.argpos_classifier = ArgumentPositionClassifier(**kw)
        self.ss_extractor = SSArgumentExtractor(**kw)
        self.ps_extractor = PSArgumentExtractor(**kw)
        self.explicit_classifier = ExplicitSenseClassifier(**kw)

    def extract_from_parsed_result(self, parsed_result, para_eventualities, output_format="triple", in_order=False, **kw):
        if output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Relation or triple.")
        
        similarity = kw.get("similarity", "simpson").lower()
        threshold = kw.get("threshold", 0.6)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Error: threshold should be between 0.0 and 1.0.")
        if similarity == "simpson":
            similarity_func = partial(self._match_argument_eventuality_by_Simpson, threshold=threshold)
        elif similarity == "jaccard":
            similarity_func = partial(self._match_argument_eventuality_by_Jaccard, threshold=threshold)
        else:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Simpson or Jaccard.")

        syntax_tree_cache = kw.get("syntax_tree_cache", dict())

        len_sentences = len(parsed_result)
        if len_sentences == 0:
            if in_order:
                return [list()]
            else:
                return list()

        para_relations = [list() for _ in range(2*len_sentences-1)]

        # replace sentences that contains no eventuality with empty sentences
        filtered_parsed_result = list()
        for sent_idx, (sent_parsed_result, sent_eventualities) in enumerate(zip(parsed_result, para_eventualities)):
            if len(sent_eventualities) > 0:
                filtered_parsed_result.append(sent_parsed_result)
                relations_in_sent = para_relations[sent_idx]
                for e1_idx in range(len(sent_eventualities)-1):
                    heid = sent_eventualities[e1_idx].eid
                    for e2_idx in range(e1_idx+1, len(sent_eventualities)):
                        teid = sent_eventualities[e2_idx].eid
                        relations_in_sent.append(Relation(heid, teid, ["Co_Occurrence"]))
            else:
                filtered_parsed_result.append(EMPTY_SENT_PARSED_RESULT) # empty sentence
                # filtered_parsed_result.append(sent_parsed_result)

        connectives = self.conn_extractor.extract(filtered_parsed_result, syntax_tree_cache)
        SS_connectives, PS_connectives = self.argpos_classifier.classify(filtered_parsed_result, connectives, syntax_tree_cache)
        SS_connectives = self.ss_extractor.extract(filtered_parsed_result, SS_connectives, syntax_tree_cache)
        PS_connectives = self.ps_extractor.extract(filtered_parsed_result, PS_connectives, syntax_tree_cache)
        connectives = self.explicit_classifier.classify(filtered_parsed_result, SS_connectives+PS_connectives, syntax_tree_cache)
        connectives.sort(key=lambda x: (x["sent_idx"], x["indices"][0] if len(x["indices"]) > 0 else -1))
        
        # For CoNLL share task 2015
        # sent_offset = 0
        # for sent_parsed_result in parsed_result:
        #     sent_parsed_result["sentence_offset"] = sent_offset
        #     sent_offset += len(sent_parsed_result["tokens"])
        # with open("aser.json", "a") as f:
        #     for conn_idx, connective in enumerate(connectives):
        #         sense = connective.get("sense", None)
        #         arg1 = connective.get("arg1", None)
        #         arg2 = connective.get("arg2", None)
        #         if arg1 and arg2 and sense and sense != "None":
        #             x = {
        #                 "DocID": sentences[0][0]["doc"], 
        #                 "ID": conn_idx, 
        #                 "Connective": {
        #                     "RawText": connective["connective"],
        #                     "TokenList": [i+sentences[connective["sent_idx"]][0]["sentence_offset"] for i in connective["indices"]],
        #                     "Tokens": [sentences[connective["sent_idx"]][0]["tokens"][i] for i in connective["indices"]]},
        #                 "Arg1": {
        #                     "RawText": " ".join([sentences[arg1["sent_idx"]][0]["tokens"][i] for i in arg1["indices"]]),
        #                     "TokenList": [i+sentences[arg1["sent_idx"]][0]["sentence_offset"] for i in arg1["indices"]],
        #                     "Tokens": [sentences[arg1["sent_idx"]][0]["tokens"][i] for i in arg1["indices"]]},
        #                 "Arg2": {
        #                     "RawText": " ".join([sentences[arg2["sent_idx"]][0]["tokens"][i] for i in arg2["indices"]]),
        #                     "TokenList": [i+sentences[arg2["sent_idx"]][0]["sentence_offset"] for i in arg2["indices"]],
        #                     "Tokens": [sentences[arg2["sent_idx"]][0]["tokens"][i] for i in arg2["indices"]]},
        #                 "Type": "Explicit",
        #                 "Sense": [connective["sense"]]}
        #             f.write(json.dumps(x))
        #             f.write("\n")

        for connective in connectives:
            conn_indices = connective.get("indices", None)
            arg1 = connective.get("arg1", None)
            arg2 = connective.get("arg2", None)
            sense = connective.get("sense", None)
            if conn_indices and arg1 and arg2 and (sense and sense != "None"):
                arg1_sent_idx = arg1["sent_idx"]
                arg2_sent_idx = arg2["sent_idx"]
                if arg1_sent_idx == arg2_sent_idx:
                    relation_list_idx = arg1_sent_idx
                    relations = para_relations[relation_list_idx]
                    sent_parsed_result, sent_eventualities = parsed_result[arg1_sent_idx], para_eventualities[arg1_sent_idx]
                    for e1_idx in range(len(sent_eventualities)-1):
                        e1 = sent_eventualities[e1_idx]
                        if not similarity_func(sent_parsed_result, arg1, e1):
                            continue
                        heid = e1.eid
                        for e2_idx in range(e1_idx+1, len(sent_eventualities)):
                            e2 = sent_eventualities[e2_idx]
                            if not similarity_func(sent_parsed_result, arg2, e2):
                                continue
                            teid = e2.eid
                            existed_relation = False
                            for relation in relations:
                                if relation.hid == heid and relation.tid == teid:
                                    relation.update_relations([sense])
                                    existed_relation = True
                                    break
                            if not existed_relation:
                                relations.append(Relation(heid, teid, [sense]))
                elif arg1_sent_idx+1 == arg2_sent_idx:
                    relation_list_idx = arg1_sent_idx + len_sentences
                    relations = para_relations[relation_list_idx]
                    sent_parsed_result1, sent_eventualities1 = parsed_result[arg1_sent_idx], para_eventualities[arg1_sent_idx]
                    sent_parsed_result2, sent_eventualities2 = parsed_result[arg2_sent_idx], para_eventualities[arg2_sent_idx]
                    for e1 in sent_eventualities1:
                        if not similarity_func(sent_parsed_result, arg1, e1):
                            continue
                        heid = e1.eid
                        for e2 in sent_eventualities2:
                            if not similarity_func(sent_parsed_result, arg2, e2):
                                continue
                            teid = e2.eid
                            existed_relation = False
                            for relation in relations:
                                if relation.hid == heid and relation.tid == teid:
                                    relation.update_relations([sense])
                                    existed_relation = True
                                    break
                            if not existed_relation:
                                relations.append(Relation(heid, teid, [sense]))
        if in_order:
            if output_format == "Relation":
                return para_relations
            elif output_format == "triple":
                return [sorted(chain.from_iterable([r.to_triples() for r in relations])) \
                    for relations in para_relations]
        else:
            if output_format == "Relation":
                rid2relation = dict()
                for relation in chain(*para_relations):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deeocopy(relation)
                    else:
                        rid2relation[relation.rid].update_relations(relation)
                return sorted(rid2relation.values(), key=lambda r: r.rid)
            if output_format == "triple":
                return sorted([r.to_triples() for relations in para_relations for r in relations])

    @staticmethod
    def _match_argument_eventuality_by_Simpson(sent_parsed_result, argument, eventuality, threshold=0.6):
        match = False
        if eventuality.raw_sent_mapping:
            argument_indices = set(argument["indices"])
            event_indices = set(eventuality.raw_sent_mapping.values())
            Simpson = len(argument_indices & event_indices) / min(len(argument_indices), len(event_indices))
            match = Simpson >= threshold
        else:
            argument_tokens = set([sent_parsed_result["lemmas"][idx].lower() for idx in argument["indices"]])
            event_tokens = set(eventuality.words)
            Simpson = len(argument_tokens & event_tokens) / min(len(argument_tokens), len(event_tokens))
            match = Simpson >= threshold
        return match
    
    @staticmethod
    def _match_argument_eventuality_by_Jaccard(sent_parsed_result, argument, eventuality, threshold=0.6):
        match = False
        if eventuality.raw_sent_mapping:
            argument_indices = set(argument["indices"])
            event_indices = set(eventuality.raw_sent_mapping.values())
            Jaccard = len(argument_indices & event_indices) / len(argument_indices | event_indices)
            match = Jaccard >= threshold
        else:
            argument_tokens = set([sent_parsed_result["lemmas"][idx].lower() for idx in argument["indices"]])
            event_tokens = set(eventuality.words)
            Jaccard = len(argument_tokens & event_tokens) / len(argument_tokens | event_tokens)
            match = Jaccard >= threshold
        return match

    @staticmethod
    def _match_argument_eventuality_by_dependencies(sent_parsed_result, conn_indices, argument, eventuality):
        match = False
        conn_indices = set(conn_indices)
        if eventuality.raw_sent_mapping:
            argument_indices = set(argument["indices"])
            event_indices = set(eventuality.raw_sent_mapping.values())

            for (governor, dep, dependent) in sent_parsed_result["dependencies"]:
                # find the word linked to the connective
                if dependent in conn_indices and governor in argument_indices and governor in event_indices:
                    match = True
                    break
                elif governor in conn_indices and dependent in argument_indices and dependent in event_indices:
                    match = True
                    break
        else:
            argument_tokens = set([sent_parsed_result["lemmas"][idx].lower() for idx in argument["indices"]])
            event_token_pos_tags = set(zip(eventuality.words, eventuality.pos_tags))

            argument_indices = set(argument["indices"])
            for (governor, dep, dependent) in sent_parsed_result["dependencies"]:
                # find the word linked to the connective
                if dependent in conn_indices and governor in argument_indices:
                    token_pos_tag = (sent_parsed_result["lemmas"][governor].lower(), sent_parsed_result["pos_tags"][governor])
                    if token_pos_tag in event_token_pos_tags:
                        match = True
                        break
                elif governor in conn_indices and dependent in argument_indices:
                    token_pos_tag = (sent_parsed_result["lemmas"][dependent].lower(), sent_parsed_result["pos_tags"][dependent])
                    if token_pos_tag in event_token_pos_tags:
                        match = True
                        break
        return match
            

class NeuralRelationExtractor(BaseRelationExtractor):
    def __init__(self, **kw):
        super().__init__(kw)
        raise NotImplementedError

    def extract(self, eventualtity_pair):
        raise NotImplementedError