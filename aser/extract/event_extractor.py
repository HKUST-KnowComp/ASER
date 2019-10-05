from aser.eventuality import Eventuality
from aser.extract.rule import ALL_EVENTUALITY_RULES, CONNECTIVE_LIST, CLAUSE_WORDS
from aser.extract.utils import get_corenlp_client, parse_sentense_with_stanford


class EventualityExtractor(object):
    def __init__(self, corenlp_path, corenlp_port):
        self.corenlp_client = get_corenlp_client(
            corenlp_path=corenlp_path, port=corenlp_port)

    def close(self):
        self.corenlp_client.stop()

    def extract_eventualities(self, text, only_events=True,
                              output_format="eventuality"):
        """ This method would firstly split text into sentences and extract
            all eventualities for each sentence.

            :type text: str
            :type only_events: bool
            :type output_format: "eventuality" or "json"
            :param text: input text
            :param only_events: only output eventualities if True, otherwise output eventualities with sentences parsed results
            :param output_format: output eventualities of `eventuality` object or a `json` dict
            :return: a list of eventualities with/without sentence information

            .. highlight:: python
            .. code-block:: python

                Input: 'The dog barks loudly. Because he is hungry.'

                Output:
                [{'eventuality_list': [{'dependencies': [[[2, 'dog', 'NN'],
                                                          'det',
                                                          [1, 'the', 'DT']],
                                                         [[3, 'bark', 'VBZ'],
                                                          'nsubj',
                                                          [2, 'dog', 'NN']],
                                                         [[3, 'bark', 'VBZ'],
                                                          'advmod',
                                                          [4, 'loudly', 'RB']]],
                                        'eid': 'b47ba21a77206552509f2cb0c751b959aaa3a625',
                                        'frequency': 0.0,
                                        'pattern': 's-v',
                                        'skeleton_dependencies': [[[3, 'bark', 'VBZ'],
                                                                   'nsubj',
                                                                   [2, 'dog', 'NN']]],
                                        'skeleton_words': [['dog', 'NN'], ['bark', 'VBZ']],
                                        'verbs': 'bark',
                                        'words': [['the', 'DT'],
                                                  ['dog', 'NN'],
                                                  ['bark', 'VBZ'],
                                                  ['loudly', 'RB']]}],
                  'sentence_dependencies': [[[2, 'dog', 'NN'], 'det', [1, 'the', 'DT']],
                                            [[3, 'bark', 'VBZ'], 'nsubj', [2, 'dog', 'NN']],
                                            [[3, 'bark', 'VBZ'], 'advmod', [4, 'loudly', 'RB']],
                                            [[3, 'bark', 'VBZ'], 'punct', [5, '.', '.']]],
                  'sentence_tokens': ['The', 'dog', 'barks', 'loudly', '.']},
                 {'eventuality_list': [{'dependencies': [[[4, 'hungry', 'JJ'],
                                                          'nsubj',
                                                          [2, 'he', 'PRP']],
                                                         [[4, 'hungry', 'JJ'],
                                                          'cop',
                                                          [3, 'be', 'VBZ']]],
                                        'eid': 'f2a6b813bdfbad5354da5514e85ca97b666ef8d1',
                                        'frequency': 0.0,
                                        'pattern': 's-be-a',
                                        'skeleton_dependencies': [[[4, 'hungry', 'JJ'],
                                                                   'nsubj',
                                                                   [2, 'he', 'PRP']],
                                                                  [[4, 'hungry', 'JJ'],
                                                                   'cop',
                                                                   [3, 'be', 'VBZ']]],
                                        'skeleton_words': [['he', 'PRP'],
                                                           ['be', 'VBZ'],
                                                           ['hungry', 'JJ']],
                                        'verbs': 'be',
                                        'words': [['he', 'PRP'],
                                                  ['be', 'VBZ'],
                                                  ['hungry', 'JJ']]}],
                  'sentence_dependencies': [[[4, 'hungry', 'JJ'], 'mark', [1, 'because', 'IN']],
                                            [[4, 'hungry', 'JJ'], 'nsubj', [2, 'he', 'PRP']],
                                            [[4, 'hungry', 'JJ'], 'cop', [3, 'be', 'VBZ']],
                                            [[4, 'hungry', 'JJ'], 'punct', [5, '.', '.']]],
                  'sentence_tokens': ['Because', 'he', 'is', 'hungry', '.']}]
        """
        rst_list = []
        parsed_results = parse_sentense_with_stanford(text, self.corenlp_client)
        for parsed_result in parsed_results:
            rst = self.extract_eventualities_from_parsed_result(parsed_result, only_events, output_format)
            rst_list.append(rst)
        return rst_list

    def extract_eventualities_from_parsed_result(
            self, parsed_result, only_events=True, output_format="eventuality"):
        """ This method would extract eventualities from parsed_result of one sentence

        :type parsed_result: dict
        :type only_events: bool
        :type output_format: "eventuality" or "json"
        :param parsed_result: a dict generated by `aser.extract.utils.parse_sentense_with_stanford`
        :param only_events: only output eventualities if True, otherwise output eventualities with sentences parsed results
        :param output_format: output eventualities of `eventuality` object or a `json` dict
        :return: a list of eventuality with/without sentence information

        .. highlight:: python
        .. code-block:: python

            Input:
                {'dependencies': [[[2, 'dog', 'NN'], 'det', [1, 'the', 'DT']],
                                  [[3, 'bark', 'VBZ'], 'nsubj', [2, 'dog', 'NN']],
                                  [[3, 'bark', 'VBZ'], 'advmod', [4, 'loudly', 'RB']],
                                  [[3, 'bark', 'VBZ'], 'punct', [5, '.', '.']]],
                 'tokens': ['The', 'dog', 'barks', 'loudly', '.']}

            Output:
                {'eventuality_list': [{'dependencies': [[[2, 'dog', 'NN'],
                                                         'det',
                                                         [1, 'the', 'DT']],
                                                        [[3, 'bark', 'VBZ'],
                                                         'nsubj',
                                                         [2, 'dog', 'NN']],
                                                        [[3, 'bark', 'VBZ'],
                                                         'advmod',
                                                         [4, 'loudly', 'RB']]],
                                       'eid': 'b47ba21a77206552509f2cb0c751b959aaa3a625',
                                       'frequency': 0.0,
                                       'pattern': 's-v',
                                       'skeleton_dependencies': [[[3, 'bark', 'VBZ'],
                                                                  'nsubj',
                                                                  [2, 'dog', 'NN']]],
                                       'skeleton_words': [['dog', 'NN'], ['bark', 'VBZ']],
                                       'verbs': 'bark',
                                       'words': [['the', 'DT'],
                                                 ['dog', 'NN'],
                                                 ['bark', 'VBZ'],
                                                 ['loudly', 'RB']]}],
                 'sentence_dependencies': [[[2, 'dog', 'NN'], 'det', [1, 'the', 'DT']],
                                           [[3, 'bark', 'VBZ'], 'nsubj', [2, 'dog', 'NN']],
                                           [[3, 'bark', 'VBZ'], 'advmod', [4, 'loudly', 'RB']],
                                           [[3, 'bark', 'VBZ'], 'punct', [5, '.', '.']]],
                 'sentence_tokens': ['The', 'dog', 'barks', 'loudly', '.']}
        """
        eventualities = self._extract_eventualities_from_parsed_result(
            parsed_result, ALL_EVENTUALITY_RULES)
        if output_format == "json":
            eventualities = [e.to_dict() for e in eventualities]
        else:
            assert output_format == "eventuality", \
                "Output format `{}` is not support".format(output_format)
        if only_events:
            return eventualities
        else:
            rst = {
                    "sentence_dependencies": parsed_result["dependencies"],
                    "sentence_tokens": parsed_result["tokens"],
                    "eventuality_list": eventualities
                  }
            return rst

    def _extract_eventualities_from_parsed_result(self, parsed_result, eventuality_rules):
        # If it is a sentence that has clause word, just skip it
        if set(parsed_result["tokens"]) & CLAUSE_WORDS:
            return []
        all_eventualities = dict()
        for rule_name in eventuality_rules:
            tmp_eventualities = self._extract_eventualities_from_dependencies_with_single_rule(
                parsed_result["dependencies"], eventuality_rules[rule_name], rule_name)
            all_eventualities[rule_name] = tmp_eventualities
        all_eventualities = self._filter_special_case(all_eventualities)
        eventuality_list = [e for elist in all_eventualities.values() for e in elist]
        return eventuality_list

    def _extract_eventualities_from_dependencies_with_single_rule(self, parsed_result, eventuality_rule,
                                                               rule_name):
        local_eventualities = list()
        verb_positions = list()
        for relation in parsed_result:
            if 'VB' in relation[0][2]:
                verb_positions.append(relation[0][0])
            if 'VB' in relation[2][2]:
                verb_positions.append(relation[2][0])

        verb_positions = list(set(verb_positions))
        for verb_position in verb_positions:
            tmp_a = self._extract_eventuality_with_fixed_target(
                parsed_result, eventuality_rule, verb_position, rule_name)
            if tmp_a is not None:
                local_eventualities.append(tmp_a)
        return local_eventualities


    def _extract_eventuality_with_fixed_target(self, parsed_result, eventuality_rule, verb_position, rule_name):
        selected_edges = list()
        selected_skeleton_edges = list()
        local_dict = {'V1': verb_position}
        for tmp_rule_r in eventuality_rule.positive_rules:
            foundmatch = False
            for dep_r in parsed_result:
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
            for dep_r in parsed_result:
                decision, local_dict = self._match_rule_r_and_dep_r(tmp_rule_r, dep_r, local_dict)
                if decision:
                    selected_edges.append(dep_r)
        for tmp_rule_r in eventuality_rule.negative_rules:
            for dep_r in parsed_result:
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
                                skeleton_dependencies=selected_skeleton_edges)
            return event
        else:
            return None

    @staticmethod
    def _match_rule_r_and_dep_r(rule_r, dep_r, current_dict):
        tmp_dict = current_dict
        if rule_r[1][0] == '-':
            tmp_relations = rule_r[1][1:].split('/')
            if rule_r[0] in current_dict and dep_r[0][0] == current_dict[rule_r[0]]:
                if dep_r[1] in tmp_relations:
                    return False, current_dict
                else:
                    # print(dep_r[1])
                    return True, tmp_dict
        if rule_r[1][0] == '+':
            tmp_relations = rule_r[1][1:].split('/')
            if rule_r[0] in current_dict and dep_r[0][0] == current_dict[rule_r[0]]:
                if dep_r[1] in tmp_relations:
                    tmp_dict[rule_r[2]] = dep_r[2][0]
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
                if tmp_rule_r[0] in current_dict and tmp_dep_r[0][0] == current_dict[tmp_rule_r[0]]:
                    if tmp_rule_r[2] not in tmp_dict:
                        tmp_dict[tmp_rule_r[2]] = tmp_dep_r[2][0]
                        return True, tmp_dict
        else:
            tmp_dep_r = dep_r
            tmp_rule_r = rule_r
            if tmp_rule_r[1] == tmp_dep_r[1]:
                if tmp_rule_r[0] in current_dict and tmp_dep_r[0][0] == current_dict[tmp_rule_r[0]]:
                    if tmp_rule_r[2] not in tmp_dict:
                        tmp_dict[tmp_rule_r[2]] = tmp_dep_r[2][0]
                        return True, tmp_dict
        return False, current_dict

    @staticmethod
    def _filter_special_case(extracted_eventualities):
        extracted_eventualities['s-v-a'] = []
        extracted_eventualities['s-v-be-o'] = []
        if len(extracted_eventualities['s-v-v']) > 0:
            tmp_s_v_v = list()
            tmp_s_v_a = list()
            for e in extracted_eventualities['s-v-v']:
                for edge in e.dependencies:
                    if edge[1] == 'xcomp':
                        if 'VB' in edge[2][2]:
                            tmp_s_v_v.append(e)
                        if 'JJ' in edge[2][2]:
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
                            tmp_s_v_be_o.append(e)
                        break
            extracted_eventualities['s-v-be-a'] = tmp_s_v_be_a
            extracted_eventualities['s-v-be-o'] = tmp_s_v_be_o
        if len(extracted_eventualities['s-v']) > 0:
            tmp_s_v = list()
            for e in extracted_eventualities['s-v']:
                for edge in e.dependencies:
                    if edge[1] == 'nsubj':
                        if edge[0][0] > edge[2][0] or edge[0][1] == 'be':
                            tmp_s_v.append(e)
            extracted_eventualities['s-v'] = tmp_s_v
        for relation in extracted_eventualities:
            new_eventualities = list()
            for tmp_e in extracted_eventualities[relation]:
                found_connective = False
                for edge in tmp_e.dependencies:
                    if edge[2][1] in CONNECTIVE_LIST:
                        found_connective = True
                        break
                if found_connective:
                    new_edges = list()
                    for edge in tmp_e.dependencies:
                        if edge[2][1] in CONNECTIVE_LIST:
                            continue
                        new_edges.append(edge)

                    new_e = Eventuality(pattern=relation,
                                        dependencies=new_edges,
                                        skeleton_dependencies=tmp_e.skeleton_dependencies)
                    tmp_e = new_e
                if len(tmp_e.dependencies) > 0:
                    new_eventualities.append(tmp_e)
            extracted_eventualities[relation] = new_eventualities

        return extracted_eventualities

