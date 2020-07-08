import bisect
from copy import copy, deepcopy
from itertools import chain, permutations
from aser.eventuality import Eventuality
from aser.extract.rule import ALL_EVENTUALITY_RULES
from aser.extract.utils import parse_sentense_with_stanford, get_corenlp_client, get_clauses, powerset
from aser.extract.utils import ANNOTATORS
from aser.extract.discourse_parser import ConnectiveExtractor, ArgumentPositionClassifier, \
    SSArgumentExtractor, PSArgumentExtractor, ExplicitSenseClassifier
from aser.extract.discourse_parser import SyntaxTree

class BaseEventualityExtractor(object):
    def __init__(self, **kw):
        self.corenlp_path = kw.get("corenlp_path", "")
        self.corenlp_port = kw.get("corenlp_port", 0)
        self.annotators = kw.get("annotators", list(ANNOTATORS))

        _, self.is_externel_corenlp = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)

    def close(self):
        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()

    def __del__(self):
        self.close()

    def parse_text(self, text, annotators=None):
        if annotators is None:
            annotators = self.annotators

        corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, annotators=annotators)
        parsed_result = parse_sentense_with_stanford(text, corenlp_client, self.annotators)
        return parsed_result

    def extract_from_text(self, text, output_format="Eventuality", in_order=True, annotators=None, **kw):
        """ This method extracts all eventualities for each sentence.

        :type text: str
        :type output_format: str
        :type in_order: bool
        :type annotators: list or None
        :param text: input text
        :param output_format: the specific output format
        :param in_order: in order or out of order
        :param annotators: the annotators parameter for the stanford corenlp client
        :return: a list of lists of `Eventuality` object

        .. highlight:: python
        .. code-block:: python

            Input: 'The dog barks loudly because it is hungry. But we have no food for it.'

            Output:
                [
                    [
                        Eventuality(
                            {'dependencies': [((1, 'dog', 'NN'), 'det', (0, 'the', 'DT')),
                                              ((2, 'bark', 'VBZ'), 'nsubj', (1, 'dog', 'NN')),
                                              ((2, 'bark', 'VBZ'), 'advmod', (3, 'loudly', 'RB'))],
                            'eid': 'b51425727182a0d25734a92ae16a456cb5e6351f',
                            'frequency': 1.0,
                            'mentions': {},
                            'ners': ['O', 'O', 'O', 'O'],
                            'pattern': 's-v',
                            'pos_tags': ['DT', 'NN', 'VBZ', 'RB'],
                            'skeleton_dependencies': [((2, 'bark', 'VBZ'), 'nsubj', (1, 'dog', 'NN'))],
                            'skeleton_words': ['dog', 'bark'],
                            'verbs': ['bark'],
                            'words': ['the', 'dog', 'bark', 'loudly']}),
                        Eventuality(
                            {'dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'it', 'PRP')),
                                              ((2, 'hungry', 'JJ'), 'cop', (1, 'be', 'VBZ'))],
                            'eid': '8fbd35fcb293f526b54c5989969251d6a31e4893',
                            'frequency': 1.0,
                            'mentions': {},
                            'ners': ['O', 'O', 'O'],
                            'pattern': 's-be-a',
                            'pos_tags': ['PRP', 'VBZ', 'JJ'],
                            'skeleton_dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'it', 'PRP')),
                                                    ((2, 'hungry', 'JJ'), 'cop', (1, 'be', 'VBZ'))],
                            'skeleton_words': ['it', 'be', 'hungry'],
                            'verbs': ['be'],
                            'words': ['it', 'be', 'hungry']})],
                    [
                        Eventuality(
                            {'dependencies': [((3, 'have', 'VB'), 'nsubj', (0, 'we', 'PRP')),
                                              ((3, 'have', 'VB'), 'aux', (1, 'do', 'VBP')),
                                              ((3, 'have', 'VB'), 'neg', (2, 'not', 'RB')),
                                              ((3, 'have', 'VB'), 'dobj', (5, 'left', 'NN')),
                                              ((5, 'left', 'NN'), 'compound', (4, 'food', 'NN'))],
                            'eid': '32bd10b7e116f7656b7424d3f3a47dab230d52de',
                            'frequency': 1.0,
                            'mentions': {},
                            'ners': ['O', 'O', 'O', 'O', 'O', 'O'],
                            'pattern': 's-v-o',
                            'pos_tags': ['PRP', 'VBP', 'RB', 'VB', 'NN', 'NN'],
                            'skeleton_dependencies': [((3, 'have', 'VB'), 'nsubj', (0, 'we', 'PRP')),
                                                    ((3, 'have', 'VB'), 'dobj', (5, 'left', 'NN'))],
                            'skeleton_words': ['we', 'have', 'left'],
                            'verbs': ['do', 'have'],
                            'words': ['we', 'do', 'not', 'have', 'food', 'left']})]]
        """
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_text only supports Eventuality or json.")
        parsed_result = self.parse_text(text, annotators)
        return self.extract_from_parsed_result(parsed_result, output_format, in_order, **kw)

    def extract_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        """ This method extracts eventualities from parsed_result of one paragraph.

        :type parsed_result: dict, or a list of dicts
        :type output_format: str
        :type in_order: bool
        :param parsed_result: a list of dicts generated by `aser.extract.utils.parse_sentense_with_stanford` or a dict
        :param output_format: the specific output format
        :param in_order: in order or out of order
        :return: a list of lists of `Eventuality` objects or a list of lists of json

        .. highlight:: python
        .. code-block:: python

            Input:
                [
                    {'text': 'The dog barks loudly because it is hungry.',
                    'dependencies': [(1, 'det', 0),
                                     (2, 'nsubj', 1),
                                     (2, 'advmod', 3),
                                     (2, 'punct', 8),
                                     (3, 'dep', 7),
                                     (7, 'mark', 4),
                                     (7, 'nsubj', 5),
                                     (7, 'cop', 6)],
                    'tokens': ['The', 'dog', 'barks', 'loudly', 'because', 'it', 'is', 'hungry', '.'],
                    'pos_tags': ['DT', 'NN', 'VBZ', 'RB', 'IN', 'PRP', 'VBZ', 'JJ', '.'],
                    'lemmas': ['the', 'dog', 'bark', 'loudly', 'because', 'it', 'be', 'hungry', '.'],
                    'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                    'mentions': [],
                    'parse': '(ROOT (S (NP (DT The) (NN dog)) (VP (VBZ barks) (ADVP (RB loudly) (SBAR (IN because) (S (NP (PRP it)) (VP (VBZ is) (ADJP (JJ hungry))))))) (. .)))'},
                    {'text': 'The dog barks loudly because',
                    'dependencies': [(4, 'cc', 0),
                                     (4, 'nsubj', 1),
                                     (4, 'aux', 2),
                                     (4, 'neg', 3),
                                     (4, 'dobj', 6),
                                     (4, 'punct', 7),
                                     (6, 'compound', 5)],
                    'tokens': ['But', 'we', 'do', "n't", 'have', 'food', 'left', '.'],
                    'pos_tags': ['CC', 'PRP', 'VBP', 'RB', 'VB', 'NN', 'NN', '.'],
                    'lemmas': ['but', 'we', 'do', 'not', 'have', 'food', 'left', '.'],
                    'ners': ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'],
                    'mentions': [],
                    'parse': "(ROOT (S (CC But) (NP (PRP we)) (VP (VBP do) (RB n't) (VP (VB have) (NP (NN food) (NN left)))) (. .)))"}]
            Output:
                [
                    [
                        Eventuality(
                            {'dependencies': [((1, 'dog', 'NN'), 'det', (0, 'the', 'DT')),
                                              ((2, 'bark', 'VBZ'), 'nsubj', (1, 'dog', 'NN')),
                                              ((2, 'bark', 'VBZ'), 'advmod', (3, 'loudly', 'RB'))],
                            'eid': 'b51425727182a0d25734a92ae16a456cb5e6351f',
                            'frequency': 1.0,
                            'mentions': {},
                            'ners': ['O', 'O', 'O', 'O'],
                            'pattern': 's-v',
                            'pos_tags': ['DT', 'NN', 'VBZ', 'RB'],
                            'skeleton_dependencies': [((2, 'bark', 'VBZ'), 'nsubj', (1, 'dog', 'NN'))],
                            'skeleton_words': ['dog', 'bark'],
                            'verbs': ['bark'],
                            'words': ['the', 'dog', 'bark', 'loudly']}),
                        Eventuality(
                            {'dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'it', 'PRP')),
                                              ((2, 'hungry', 'JJ'), 'cop', (1, 'be', 'VBZ'))],
                            'eid': '8fbd35fcb293f526b54c5989969251d6a31e4893',
                            'frequency': 1.0,
                            'mentions': {},
                            'ners': ['O', 'O', 'O'],
                            'pattern': 's-be-a',
                            'pos_tags': ['PRP', 'VBZ', 'JJ'],
                            'skeleton_dependencies': [((2, 'hungry', 'JJ'), 'nsubj', (0, 'it', 'PRP')),
                                                    ((2, 'hungry', 'JJ'), 'cop', (1, 'be', 'VBZ'))],
                            'skeleton_words': ['it', 'be', 'hungry'],
                            'verbs': ['be'],
                            'words': ['it', 'be', 'hungry']})],
                    [
                        Eventuality(
                            {'dependencies': [((3, 'have', 'VB'), 'nsubj', (0, 'we', 'PRP')),
                                              ((3, 'have', 'VB'), 'aux', (1, 'do', 'VBP')),
                                              ((3, 'have', 'VB'), 'neg', (2, 'not', 'RB')),
                                              ((3, 'have', 'VB'), 'dobj', (5, 'left', 'NN')),
                                              ((5, 'left', 'NN'), 'compound', (4, 'food', 'NN'))],
                            'eid': '32bd10b7e116f7656b7424d3f3a47dab230d52de',
                            'frequency': 1.0,
                            'mentions': {},
                            'ners': ['O', 'O', 'O', 'O', 'O', 'O'],
                            'pattern': 's-v-o',
                            'pos_tags': ['PRP', 'VBP', 'RB', 'VB', 'NN', 'NN'],
                            'skeleton_dependencies': [((3, 'have', 'VB'), 'nsubj', (0, 'we', 'PRP')),
                                                    ((3, 'have', 'VB'), 'dobj', (5, 'left', 'NN'))],
                            'skeleton_words': ['we', 'have', 'left'],
                            'verbs': ['do', 'have'],
                            'words': ['we', 'do', 'not', 'have', 'food', 'left']})]]
        """
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")
        raise NotImplementedError


class SeedRuleEventualityExtractor(BaseEventualityExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.skip_words = kw.get("skip_words", set())
        if not isinstance(self.skip_words, set):
            self.skip_words = set(self.skip_words)

    def extract_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")

        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False
        
        eventuality_rules = kw.get("eventuality_rules", None)
        if eventuality_rules is None:
            eventuality_rules = ALL_EVENTUALITY_RULES
        
        para_eventualities = [list() for _ in range(len(parsed_result))]
        for sent_parsed_result, sent_eventualities in zip(parsed_result, para_eventualities):
            if self.skip_words and set(sent_parsed_result["tokens"]) & self.skip_words:
                continue
            seed_rule_eventualities = dict()
            # print(sent_parsed_result["tokens"])
            for rule_name in eventuality_rules:
                tmp_eventualities = self._extract_eventualities_from_dependencies_with_single_rule(
                    sent_parsed_result, eventuality_rules[rule_name], rule_name)
                seed_rule_eventualities[rule_name] = tmp_eventualities
                # print("rule", rule_name, tmp_eventualities)
            seed_rule_eventualities = self._filter_special_case(seed_rule_eventualities)
            # print("-------------")
            for eventualities in seed_rule_eventualities.values():
                sent_eventualities.extend(eventualities)
        
        if in_order:
            if output_format == "json":
                para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
                    for sent_eventualities in para_eventualities]
            if is_single_sent:
                return para_eventualities[0]
            else:
                return para_eventualities
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif output_format == "json":
                eventualities = sorted([eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()], key=lambda e: e["eid"])
            return eventualities

    def _extract_eventualities_from_dependencies_with_single_rule(self, sent_parsed_result, eventuality_rule, rule_name):
        local_eventualities = list()
        verb_positions = [i for i, tag in enumerate(sent_parsed_result["pos_tags"])
                          if tag.startswith("VB")]
        for verb_position in verb_positions:
            tmp_e = self._extract_eventuality_with_fixed_target(
                sent_parsed_result, eventuality_rule, verb_position, rule_name)
            if tmp_e is not None:
                local_eventualities.append(tmp_e)
        return local_eventualities

    def _extract_eventuality_with_fixed_target(self, sent_parsed_result, eventuality_rule, verb_position, rule_name):
        selected_edges = list()
        selected_skeleton_edges = list()
        local_dict = {'V1': verb_position}
        for tmp_rule_r in eventuality_rule.positive_rules:
            foundmatch = False
            for dep_r in sent_parsed_result["dependencies"]:
                decision, local_dict = self._match_rule_r_and_dep_r(tmp_rule_r, dep_r, local_dict)
                if decision:
                    selected_edges.append(dep_r)
                    selected_skeleton_edges.append(dep_r)
                    foundmatch = True
                    break
            if not foundmatch:
                # print('Miss one positive relation')
                return None

        for tmp_rule_r in eventuality_rule.possible_rules:
            for dep_r in sent_parsed_result["dependencies"]:
                decision, local_dict = self._match_rule_r_and_dep_r(tmp_rule_r, dep_r, local_dict)
                if decision:
                    selected_edges.append(dep_r)

        for tmp_rule_r in eventuality_rule.negative_rules:
            for dep_r in sent_parsed_result["dependencies"]:
                if dep_r in selected_edges:
                    # print('This edge is selected by the positive example, so we will skip it')
                    continue
                decision, local_dict = self._match_rule_r_and_dep_r(tmp_rule_r, dep_r, local_dict)
                if decision:
                    # print('found one negative relation')
                    return None

        if len(selected_edges) > 0:
            event = Eventuality(pattern=rule_name,
                                dependencies=selected_edges,
                                skeleton_dependencies=selected_skeleton_edges,
                                sent_parsed_result=sent_parsed_result)
            if len(event) > 0:
                return event
            else:
                return event
        else:
            return None

    @staticmethod
    def _match_rule_r_and_dep_r(rule_r, dep_r, current_dict):
        tmp_dict = {key: val for key, val in current_dict.items()}
        if rule_r[1][0] == '-':
            tmp_relations = rule_r[1][1:].split('/')
            if rule_r[0] in current_dict and dep_r[0] == current_dict[rule_r[0]]:
                if dep_r[1] in tmp_relations:
                    return False, current_dict
                else:
                    # print(dep_r[1])
                    return True, tmp_dict
        if rule_r[1][0] == '+':
            tmp_relations = rule_r[1][1:].split('/')
            if rule_r[0] in current_dict and dep_r[0] == current_dict[rule_r[0]]:
                if dep_r[1] in tmp_relations:
                    tmp_dict[rule_r[2]] = dep_r[2]
                    return True, tmp_dict
                else:
                    # print(dep_r[1])
                    return False, current_dict
        if rule_r[1][0] == '^':
            tmp_dep_r = list()
            tmp_dep_r.append(dep_r[2])
            tmp_dep_r.append(dep_r[1])
            tmp_dep_r.append(dep_r[0])
            tmp_rule_r = list()
            tmp_rule_r.append(rule_r[2])
            tmp_rule_r.append(rule_r[1][1:])
            tmp_rule_r.append(rule_r[0])
            if tmp_rule_r[1] == tmp_dep_r[1]:
                if tmp_rule_r[0] in current_dict and tmp_dep_r[0] == current_dict[tmp_rule_r[0]]:
                    if tmp_rule_r[2] not in tmp_dict:
                        tmp_dict[tmp_rule_r[2]] = tmp_dep_r[2]
                        return True, tmp_dict
        else:
            tmp_dep_r = dep_r
            tmp_rule_r = rule_r
            if tmp_rule_r[1] == tmp_dep_r[1]:
                if tmp_rule_r[0] in current_dict and tmp_dep_r[0] == current_dict[tmp_rule_r[0]]:
                    if tmp_rule_r[2] not in tmp_dict:
                        tmp_dict[tmp_rule_r[2]] = tmp_dep_r[2]
                        return True, tmp_dict
        return False, current_dict

    @staticmethod
    def _filter_special_case(extracted_eventualities):
        for k, v in extracted_eventualities.items():
            extracted_eventualities[k] = [e for e in v if "|" not in e.words]

        extracted_eventualities['s-v-a'] = []
        extracted_eventualities['s-be-o'] = []
        extracted_eventualities['s-v-be-o'] = []
        extracted_eventualities['s-v-o-be-o'] = []

        if len(extracted_eventualities['s-v-v']) > 0:
            tmp_s_v_v = list()
            tmp_s_v_a = list()
            for e in extracted_eventualities['s-v-v']:
                for edge in e.dependencies:
                    if edge[1] == 'xcomp':
                        if 'VB' in edge[2][2]:
                            tmp_s_v_v.append(e)
                        if 'JJ' in edge[2][2]:
                            e.pattern = 's-v-a'
                            tmp_s_v_a.append(e)
                        break
            extracted_eventualities['s-v-v'] = tmp_s_v_v
            extracted_eventualities['s-v-a'] = tmp_s_v_a

        if len(extracted_eventualities['s-v-be-a']) > 0:
            tmp_s_v_be_a = list()
            tmp_s_v_be_o = list()
            for e in extracted_eventualities['s-v-be-a']:
                for edge in e.dependencies:
                    if edge[1] == 'xcomp':
                        if 'JJ' in edge[2][2]:
                            tmp_s_v_be_a.append(e)
                        if 'NN' in edge[2][2]:
                            e.pattern = 's-v-be-o'
                            tmp_s_v_be_o.append(e)
                        break
            extracted_eventualities['s-v-be-a'] = tmp_s_v_be_a
            extracted_eventualities['s-v-be-o'] = tmp_s_v_be_o

        if len(extracted_eventualities['s-be-a']) > 0:
            tmp_s_be_a = list()
            tmp_s_be_o = list()
            for e in extracted_eventualities['s-be-a']:
                for edge in e.dependencies:
                    if edge[1] == 'cop':
                        if 'JJ' in edge[0][2]:
                            tmp_s_be_a.append(e)
                        if 'NN' in edge[0][2]:
                            e.pattern = 's-be-o'
                            tmp_s_be_o.append(e)
                        break
            extracted_eventualities['s-be-a'] = tmp_s_be_a
            extracted_eventualities['s-be-o'] = tmp_s_be_o

        if len(extracted_eventualities['s-v-o-be-a']) > 0:
            tmp_s_v_o_be_a = list()
            tmp_s_v_o_be_o = list()
            for e in extracted_eventualities['s-v-o-be-a']:
                for edge in e.dependencies:
                    if edge[1] == 'xcomp':
                        if 'JJ' in edge[2][2]:
                            tmp_s_v_o_be_a.append(e)
                        if 'NN' in edge[2][2]:
                            e.pattern = 's-v-o-be-o'
                            tmp_s_v_o_be_o.append(e)
                        break
            extracted_eventualities['s-v-o-be-a'] = tmp_s_v_o_be_a
            extracted_eventualities['s-v-o-be-o'] = tmp_s_v_o_be_o

        if len(extracted_eventualities['s-v']) > 0:
            tmp_s_v = list()
            for e in extracted_eventualities['s-v']:
                for edge in e.dependencies:
                    if edge[1] == 'nsubj':
                        if edge[0][0] > edge[2][0] or edge[0][1] == 'be':
                            tmp_s_v.append(e)
            extracted_eventualities['s-v'] = tmp_s_v

        return extracted_eventualities

class DiscourseEventualityExtractor(BaseEventualityExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.seed_rule_eventuality_extractor = SeedRuleEventualityExtractor(**kw)
        self.conn_extractor = ConnectiveExtractor(**kw)
        # self.argpos_classifier = ArgumentPositionClassifier(**kw)
        # self.ss_extractor = SSArgumentExtractor(**kw)
        # self.ps_extractor = PSArgumentExtractor(**kw)

    def extract_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")
        
        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False

        syntax_tree_cache = kw.get("syntax_tree_cache", dict())
        
        para_eventualities = [list() for _ in range(len(parsed_result))]
        para_clauses = self._extract_clauses(parsed_result, syntax_tree_cache)
        for sent_parsed_result, sent_clauses, sent_eventualities in zip(parsed_result, para_clauses, para_eventualities):
            for clause in sent_clauses:
                len_clause = len(clause)
                idx_mapping = {j: i for i, j in enumerate(clause)}
                indices_set = set(clause)
                clause_parsed_result = {
                    "text": "",
                    "dependencies": [(idx_mapping[dep[0]], dep[1], idx_mapping[dep[2]]) for dep in sent_parsed_result["dependencies"] \
                        if dep[0] in indices_set and dep[2] in indices_set],
                    "tokens": [sent_parsed_result["tokens"][idx] for idx in clause],
                    "pos_tags": [sent_parsed_result["pos_tags"][idx] for idx in clause],
                    "lemmas": [sent_parsed_result["lemmas"][idx] for idx in clause]}
                if "ners" in sent_parsed_result:
                    clause_parsed_result["ners"] = [sent_parsed_result["ners"][idx] for idx in clause]
                if "mentions" in sent_parsed_result:
                    clause_parsed_result["mentions"] = list()
                    for mention in sent_parsed_result["mentions"]:
                        start_idx = bisect.bisect_left(clause, mention["start"])
                        if not (start_idx < len_clause and clause[start_idx] == mention["start"]):
                            continue
                        end_idx = bisect.bisect_left(clause, mention["end"]-1)
                        if not (end_idx < len_clause and clause[end_idx] == mention["end"]-1):
                            continue
                        mention = copy(mention)
                        mention["start"] = start_idx
                        mention["end"] = end_idx+1
                        clause_parsed_result["mentions"].append(mention)
                eventualities = self.seed_rule_eventuality_extractor.extract_from_parsed_result(
                    clause_parsed_result, output_format="Eventuality", in_order=True)
                len_existed_eventualities = len(sent_eventualities)
                for e in eventualities:
                    for k, v in e.raw_sent_mapping.items():
                        e.raw_sent_mapping[k] = clause[v]
                    e.eid = Eventuality.generate_eid(e)
                    existed_eventuality = False
                    for e_idx in range(len_existed_eventualities):
                        if sent_eventualities[e_idx].eid == e.eid and \
                            sent_eventualities[e_idx].raw_sent_mapping == e.raw_sent_mapping:
                            existed_eventuality = True
                            break
                    if not existed_eventuality:
                        sent_eventualities.append(e)
        
        if in_order:
            if output_format == "json":
                para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
                    for sent_eventualities in para_eventualities]
            if is_single_sent:
                return para_eventualities[0]
            else:
                return para_eventualities
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif output_format == "json":
                eventualities = sorted([eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()], key=lambda e: e["eid"])
            return eventualities

    def _extract_clauses(self, parsed_result, syntax_tree_cache):
        para_arguments = [set() for _ in range(len(parsed_result))]
        connectives = self.conn_extractor.extract(parsed_result, syntax_tree_cache)
        para_connectives = [set() for _ in range(len(parsed_result))]
        for connective in connectives:
            sent_idx, indices = connective["sent_idx"], tuple(connective["indices"])
            para_connectives[sent_idx].add(indices)
        for sent_idx, sent_parsed_result in enumerate(parsed_result):
            sent_connectives = para_connectives[sent_idx]
            sent_arguments = para_arguments[sent_idx]

            if sent_idx in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx]
            else:
                syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])
            
            # more but slower
            # for indices in powerset(sent_connectives):
            #     indices = set(chain.from_iterable(indices))
            #     sent_arguments.update(get_clauses(sent_parsed_result, syntax_tree, index_seps=indices))
            sent_arguments.update(get_clauses(sent_parsed_result, syntax_tree, index_seps=set(chain.from_iterable(sent_connectives))))
        # print("'clause indices':", para_arguments)
        return para_arguments