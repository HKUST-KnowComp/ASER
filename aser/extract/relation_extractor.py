import bisect
import subprocess
import numpy as np
import time
import os
try:
    import ujson as json
except:
    import json
from pprint import pprint
from copy import deepcopy
from aser.extract.discourse_parser import ConnectiveExtractor, ArgumentPositionClassifier, \
    SSArgumentExtractor, PSArgumentExtractor, ExplicitSenseClassifier
from itertools import chain
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses


class BaseRelationExtractor(object):
    def __init__(self, **kw):
        pass

    def close(self):
        pass

    def __del__(self):
        self.close()

    def extract(self, eventuality_pair):
        raise NotImplementedError


class SeedRuleRelationExtractor(BaseRelationExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seed_connective_dict = {
            'Precedence': [['before']],
            'Succession': [['after']],
            'Synchronous': [['meanwhile'], ['at', 'the', 'same', 'time']],
            'Reason': [['because']],
            'Result': [['so'], ['thus'], ['therefore']],
            'Condition': [['if']],
            'Contrast': [['but'], ['however']],
            'Concession': [['although']],
            'Conjunction': [['and'], ['also']],
            'Instantiation': [['for', 'example'], ['for', 'instance']],
            'Restatement': [['in', 'other', 'words']],
            'Alternative': [['or'], ['unless']],
            'ChosenAlternative': [['instead']],
            'Exception': [['except']],
            'Co_Occurrence': list()
        }

    def extract(self, sentences, output_format="triple", in_order=False):
        """ This methods extract relations among extracted eventualities based on seed rule

            :type sentences: list
            :param sentences: list of (sentence_parsed_result, EventualityList)
            :return: list of triples of format (eid1, relation, eid2)

            .. highlight:: python
            .. code-block:: python

                Input Example:
                    [({'dependencies': [(1, 'nsubj', 0),
                                    (1, 'nmod:to', 3),
                                    (1, 'advcl:because', 7),
                                    (1, 'punct', 8),
                                    (3, 'case', 2),
                                    (7, 'mark', 4),
                                    (7, 'nsubj', 5),
                                    (7, 'cop', 6)],
                        'lemmas': ['I', 'go', 'to', 'lunch', 'because', 'I', 'be', 'hungry', '.'],
                        'pos_tags': ['PRP', 'VBP', 'TO', 'NN', 'IN', 'PRP', 'VBP', 'JJ', '.'],
                        'tokens': ['I', 'go', 'to', 'lunch', 'because', 'I', 'am', 'hungry', '.']},

                        EventualityList([
                            Eventuatlity({'dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'I', 'PRP')),
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
                        ])]

                Output Example:
                    [('a53fd728f8a4dd955e7ed2bd72ff07ffabb8e7f5',
                      'Co_Occurrence',
                      'c08b06c1b3a3e9ada88dd7034618d0969ae2b244'),
                     ('a53fd728f8a4dd955e7ed2bd72ff07ffabb8e7f5',
                      'Reason',
                      'c08b06c1b3a3e9ada88dd7034618d0969ae2b244')
        """
        if output_format not in ["triple", "relation"]:
            raise NotImplementedError

        extracted_relations_in_order = list()
        for sent_parsed_result, eventualities in sentences:
            relations_in_sent = list()
            for head_eventuality in eventualities:
                for tail_eventuality in eventualities:
                    if head_eventuality.position < tail_eventuality.position:
                        heid = head_eventuality.eid
                        teid = tail_eventuality.eid
                        extracted_senses = self._extract_from_eventuality_pair_in_one_sentence(
                            head_eventuality, tail_eventuality, sent_parsed_result)
                        if output_format == "triple":
                            for sense in extracted_senses:
                                relations_in_sent.append((heid, sense, teid))
                        elif output_format == "relation":
                            relations_in_sent.append(Relation(heid, teid, extracted_senses))
            extracted_relations_in_order.append(relations_in_sent)

        for i in range(len(sentences) - 1):
            s1_eventuality_list = sentences[i][1]
            s2_eventuality_list = sentences[i + 1][1]
            relations_between_sents = list()
            if len(s1_eventuality_list) == 1 and len(s2_eventuality_list) == 1:
                s1_sentence_tokens = sentences[i][0]["tokens"]
                s2_sentence_tokens = sentences[i + 1][0]["tokens"]
                head_eventuality = s1_eventuality_list[0]
                tail_eventuality = s2_eventuality_list[0]
                heid = head_eventuality.eid
                teid = tail_eventuality.eid
                extracted_senses = self._extract_from_eventuality_pair_in_two_sentence(
                    head_eventuality, tail_eventuality,
                    s1_sentence_tokens,
                    s2_sentence_tokens)
                if output_format == "triple":
                    for sense in extracted_senses:
                        relations_between_sents.append((heid, sense, teid))
                elif output_format == "relation":
                    relations_between_sents.append(Relation(heid, teid, extracted_senses))
            extracted_relations_in_order.append(relations_between_sents)
        if in_order:
            return extracted_relations_in_order
        else:
            if output_format == "triple":
                return sorted(chain(*extracted_relations_in_order))
            elif output_format == "relation":
                rid2relation = dict()
                for relation in chain(*extracted_relations_in_order):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deeocopy(relation)
                    else:
                        rid2relation[relation.rid].update_relations(relation)
                return sorted(rid2relation.values(), key=lambda x: x.rid)

    def _extract_from_eventuality_pair_in_one_sentence(self,
                                                       head_eventuality,
                                                       tail_eventuality,
                                                       sent_parsed_result):
        extracted_senses = ['Co_Occurrence']
        for sense in relation_senses:
            for connective_words in self.seed_connective_dict[sense]:
                if self._verify_connective_in_one_sentence(
                    connective_words, head_eventuality, tail_eventuality,
                    sent_parsed_result["dependencies"],
                    sent_parsed_result["tokens"]):
                    extracted_senses.append(sense)
                    break
        return extracted_senses

    def _extract_from_eventuality_pair_in_two_sentence(self,
                                                      head_eventuality,
                                                      tail_eventuality,
                                                      s1_sentence_tokens,
                                                      s2_sentence_tokens):
        extracted_senses = list()
        for sense in relation_senses:
            for connective_words in self.seed_connective_dict[sense]:
                if self._verify_connective_in_two_sentence(
                        connective_words,
                        head_eventuality, tail_eventuality,
                        s1_sentence_tokens, s2_sentence_tokens):
                    extracted_senses.append(sense)
                    break

        return extracted_senses

    def _verify_connective_in_one_sentence(self, connective_words,
                                           head_eventuality, tail_eventuality,
                                           sentence_dependencies, sentence_tokens):
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


    def _verify_connective_in_two_sentence(self, connective_words,
                                           s1_head_eventuality,
                                           s2_tail_eventuality,
                                           s1_sentence_tokens,
                                           s2_sentence_tokens):
        def get_connective_position():
            tmp_positions = list()
            for w in connective_words:
                if w in s1_sentence_tokens:
                    tmp_positions.append(s1_sentence_tokens.index(w))
                elif w in s2_sentence_tokens:
                        tmp_positions.append(s2_sentence_tokens.index(w) + len(s1_sentence_tokens))
            return sum(tmp_positions) / len(tmp_positions) if tmp_positions else 0.0
        sentence_tokens = s1_sentence_tokens + s2_sentence_tokens
        # Connective Words need to be presented in sentence
        if set(connective_words) - set(sentence_tokens):
            return False
        # Connective phrase need to be presented in sentence
        connective_string = " ".join(connective_words)
        sentence_string = " ".join(sentence_tokens)
        if connective_string not in sentence_string:
            return False
        connective_position = get_connective_position()
        e1_position, e2_position = s1_head_eventuality.position, \
                                   s2_tail_eventuality.position + len(s1_sentence_tokens)
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

        self.empty_sent_parsed_result = {
            'text': '.',
            'dependencies': list(),
            'parse': '(ROOT\r\n  (NP (. .)))',
            'tokens': ['.'],
            'lemmas': ['.'],
            'pos_tags': ['.']}

    def extract(self, sentences, output_format="triple", in_order=False):
        """ This methods extract relations among extracted eventualities based on seed rule

            :type sentences: list
            :param sentences: list of (sentence_parsed_result, EventualityList)
            :return: list of triples of format (eid1, relation, eid2)

            .. highlight:: python
            .. code-block:: python

                Input Example:
                    [({'dependencies': [(1, 'nsubj', 0),
                                    (1, 'nmod:to', 3),
                                    (1, 'advcl:because', 7),
                                    (1, 'punct', 8),
                                    (3, 'case', 2),
                                    (7, 'mark', 4),
                                    (7, 'nsubj', 5),
                                    (7, 'cop', 6)],
                        'lemmas': ['I', 'go', 'to', 'lunch', 'because', 'I', 'be', 'hungry', '.'],
                        'pos_tags': ['PRP', 'VBP', 'TO', 'NN', 'IN', 'PRP', 'VBP', 'JJ', '.'],
                        'tokens': ['I', 'go', 'to', 'lunch', 'because', 'I', 'am', 'hungry', '.']},

                        EventualityList([
                            Eventuatlity({'dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'I', 'PRP')),
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
                        ])]

                Output Example:
                    [('a53fd728f8a4dd955e7ed2bd72ff07ffabb8e7f5',
                      'Co_Occurrence',
                      'c08b06c1b3a3e9ada88dd7034618d0969ae2b244'),
                     ('a53fd728f8a4dd955e7ed2bd72ff07ffabb8e7f5',
                      'Reason',
                      'c08b06c1b3a3e9ada88dd7034618d0969ae2b244')
        """
        if output_format not in ["triple", "relation"]:
            raise NotImplementedError

        len_sentences = len(sentences)
        if len_sentences == 0:
            return list()

        syntax_tree_cache = dict()
        extracted_relations_in_order = [list() for _ in range(2*len_sentences-1)]

        # replace sentences that contains no eventuality with empty sentences
        sent_offset = 0
        doc_parsed_result = list()
        for sent_idx, (sent_parsed_result, eventualities) in enumerate(sentences):
            sent_parsed_result["sentence_offset"] = sent_offset
            sent_offset += len(sent_parsed_result["tokens"])
            if len(eventualities) > 0:
                doc_parsed_result.append(sent_parsed_result)
                relations_in_sent = extracted_relations_in_order[sent_idx]
                for head_eventuality in eventualities:
                    for tail_eventuality in eventualities:
                        if head_eventuality.position < tail_eventuality.position:
                            heid = head_eventuality.eid
                            teid = tail_eventuality.eid
                            if output_format == "triple":
                                relations_in_sent.append((heid, "Co_Occurrence", teid))
                            elif output_format == "relation":
                                relations_in_sent.append(Relation(heid, teid, ["Co_Occurrence"]))
            else:
                doc_parsed_result.append(self.empty_sent_parsed_result) # empty sentence
                # doc_parsed_result.append(sent_parsed_result)

        doc_connectives = self.conn_extractor.extract(doc_parsed_result, syntax_tree_cache)
        SS_connectives, PS_connectives = self.argpos_classifier.classify(doc_parsed_result, doc_connectives, syntax_tree_cache)
        SS_connectives = self.ss_extractor.extract(doc_parsed_result, SS_connectives, syntax_tree_cache)
        PS_connectives = self.ps_extractor.extract(doc_parsed_result, PS_connectives, syntax_tree_cache)
        doc_connectives = self.explicit_classifier.classify(doc_parsed_result, SS_connectives+PS_connectives, syntax_tree_cache)
        doc_connectives.sort(key=lambda x: (x["sent_idx"], x["indices"][0] if len(x["indices"]) > 0 else -1))
        
        # For CoNLL share task 2015
        # with open("aser.json", "a") as f:
        #     for conn_idx, connective in enumerate(doc_connectives):
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

        for connective in doc_connectives:
            conn_indices = connective.get("indices", None)
            arg1 = connective.get("arg1", None)
            arg2 = connective.get("arg2", None)
            sense = connective.get("sense", None)
            if conn_indices and arg1 and arg2 and sense:
                arg1_sent_idx = arg1["sent_idx"]
                arg2_sent_idx = arg2["sent_idx"]
                if arg1_sent_idx == arg2_sent_idx:
                    relation_list_idx = arg1_sent_idx
                elif arg1_sent_idx+1 == arg2_sent_idx:
                    relation_list_idx = arg1_sent_idx + len_sentences
                else:
                    continue
                relations_in_sent = extracted_relations_in_order[relation_list_idx]
                sent_parsed_result1, eventualities1 = sentences[arg1_sent_idx]
                sent_parsed_result2, eventualities2 = sentences[arg2_sent_idx]
                for e1 in eventualities1:
                    if self._match_argument_eventuality_by_Jaccard(sent_parsed_result1, arg1, e1):
                        heid = e1.eid
                        for e2 in eventualities2:
                            if e1 == e2:
                                continue
                            if self._match_argument_eventuality_by_Jaccard(sent_parsed_result2, arg2, e2):
                                teid = e2.eid
                                if output_format == "triple":
                                    relations_in_sent.append((heid, sense, teid))
                                elif output_format == "relation":
                                    existed_relation = False
                                    for relation in relations_in_sent:
                                        if relation.hid == heid and relation.tid == teid:
                                            relation.update_relations([sense])
                                            existed_relation = True
                                            break
                                    if not existed_relation:
                                        relations_in_sent.append(Relation(heid, teid, [sense]))

        if in_order:
            return extracted_relations_in_order
        else:
            if output_format == "triple":
                return sorted(chain(*extracted_relations_in_order))
            elif output_format == "relation":
                rid2relation = dict()
                for relation in chain(*extracted_relations_in_order):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deepcopy(relation)
                    else:
                        rid2relation[relation.rid].update_relations(relation)
                return sorted(rid2relation.values(), key=lambda x: x.rid)
    
    def _match_argument_eventuality_by_Jaccard(self, sent_parsed_result, argument, eventuality, threshold=0.6):
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

    def _match_argument_eventuality_by_dependencies(self, sent_parsed_result, conn_indices, argument, eventuality):
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