import bisect
from copy import copy, deepcopy
from itertools import chain, permutations
from .discourse_parser import ConnectiveExtractor
from .discourse_parser import SyntaxTree
from .rule import ALL_EVENTUALITY_RULES
from .utils import parse_sentense_with_stanford, get_corenlp_client, get_clauses, powerset
from .utils import ANNOTATORS
from ..eventuality import Eventuality


class BaseEventualityExtractor(object):
    """ Base ASER eventuality extractor to extract eventualities

    """
    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        """

        :param corenlp_path: corenlp path, e.g., /home/xliucr/stanford-corenlp-3.9.2
        :type corenlp_path: str (default = "")
        :param corenlp_port: corenlp port, e.g., 9000
        :type corenlp_port: int (default = 0)
        :param kw: other parameters
        :type kw: Dict[str, object]
        """

        self.corenlp_path = corenlp_path
        self.corenlp_port = corenlp_port
        self.annotators = kw.get("annotators", list(ANNOTATORS))

        _, self.is_externel_corenlp = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)

    def close(self):
        """ Close the extractor safely
        """

        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()

    def __del__(self):
        self.close()

    def parse_text(self, text, annotators=None):
        """ Parse a raw text by corenlp

        :param text: a raw text
        :type text: str
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :return: the parsed result
        :rtype: List[Dict[str, object]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

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
                         '.']}]
        """

        if annotators is None:
            annotators = self.annotators

        corenlp_client, _ = get_corenlp_client(
            corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, annotators=annotators
        )
        parsed_result = parse_sentense_with_stanford(text, corenlp_client, self.annotators)
        return parsed_result

    def extract_from_text(self, text, output_format="Eventuality", in_order=True, annotators=None, **kw):
        """ Extract eventualities from a raw text

        :param text: a raw text
        :type text: str
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities
        :rtype: Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            [[my army will find you boat],
             [i be sure, we could find you suitable accommodation]]
        """

        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_text only supports Eventuality or json.")
        parsed_result = self.parse_text(text, annotators)
        return self.extract_from_parsed_result(parsed_result, output_format, in_order, **kw)

    def extract_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        """ Extract eventualities from the parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities
        :rtype: Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]]

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
                         '.']}]

            Output:

            [[my army will find you boat],
             [i be sure, we could find you suitable accommodation]]

        """

        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")
        raise NotImplementedError


class SeedRuleEventualityExtractor(BaseEventualityExtractor):
    """ ASER eventuality extractor based on rules to extract eventualities  (for ASER v1.0)

    """
    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        """

        :param corenlp_path: corenlp path, e.g., /home/xliucr/stanford-corenlp-3.9.2
        :type corenlp_path: str (default = "")
        :param corenlp_port: corenlp port, e.g., 9000
        :type corenlp_port: int (default = 0)
        :param kw: other parameters, e.g., "skip_words" to drop sentences that contain such words
        :type kw: Dict[str, object]
        """
        super().__init__(corenlp_path, corenlp_port, **kw)
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
                    sent_parsed_result, eventuality_rules[rule_name], rule_name
                )
                seed_rule_eventualities[rule_name] = tmp_eventualities
                # print("rule", rule_name, tmp_eventualities)
            seed_rule_eventualities = self._filter_special_case(seed_rule_eventualities)
            # print("-------------")
            for eventualities in seed_rule_eventualities.values():
                sent_eventualities.extend(eventualities)

        if in_order:
            para_eventualities = [
                sorted(sent_eventualities, key=lambda e: e.position) for sent_eventualities in para_eventualities
            ]
            if output_format == "json":
                para_eventualities = [
                    [eventuality.encode(encoding=None) for eventuality in sent_eventualities]
                    for sent_eventualities in para_eventualities
                ]
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
                eventualities = sorted(
                    [eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()],
                    key=lambda e: e["eid"]
                )
            return eventualities

    def _extract_eventualities_from_dependencies_with_single_rule(
        self, sent_parsed_result, eventuality_rule, rule_name
    ):
        local_eventualities = list()
        verb_positions = [i for i, tag in enumerate(sent_parsed_result["pos_tags"]) if tag.startswith("VB")]
        for verb_position in verb_positions:
            tmp_e = self._extract_eventuality_with_fixed_target(
                sent_parsed_result, eventuality_rule, verb_position, rule_name
            )
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
            event = Eventuality(
                pattern=rule_name,
                dependencies=selected_edges,
                skeleton_dependencies=selected_skeleton_edges,
                parsed_result=sent_parsed_result
            )
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
    """ ASER eventuality extractor based on constituency analysis to extract eventualities  (for ASER v2.0)

    """
    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        super().__init__(corenlp_path, corenlp_port, **kw)
        self.seed_rule_eventuality_extractor = SeedRuleEventualityExtractor(**kw)
        self.conn_extractor = ConnectiveExtractor(**kw)

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
        for sent_parsed_result, sent_clauses, sent_eventualities in zip(
            parsed_result, para_clauses, para_eventualities
        ):
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
                        end_idx = bisect.bisect_left(clause, mention["end"] - 1)
                        if not (end_idx < len_clause and clause[end_idx] == mention["end"] - 1):
                            continue
                        mention = copy(mention)
                        mention["start"] = start_idx
                        mention["end"] = end_idx + 1
                        clause_parsed_result["mentions"].append(mention)
                eventualities = self.seed_rule_eventuality_extractor.extract_from_parsed_result(
                    clause_parsed_result, output_format="Eventuality", in_order=True
                )
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
            para_eventualities = [
                sorted(sent_eventualities, key=lambda e: e.position) for sent_eventualities in para_eventualities
            ]
            if output_format == "json":
                para_eventualities = [
                    [eventuality.encode(encoding=None) for eventuality in sent_eventualities]
                    for sent_eventualities in para_eventualities
                ]
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
                eventualities = sorted(
                    [eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()],
                    key=lambda e: e["eid"]
                )
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
            #     sent_arguments.update(get_clauses(sent_parsed_result, syntax_tree, sep_indices=indices))
            sent_arguments.update(
                get_clauses(sent_parsed_result, syntax_tree, sep_indices=set(chain.from_iterable(sent_connectives)))
            )
        # print("'clause indices':", para_arguments)
        return para_arguments
