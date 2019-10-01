from stanfordnlp.server import CoreNLPClient
from aser.eventuality import Eventuality
from aser.extract.activity import Activity
from aser.extract.rule import All_activity_rules
from aser.extract.utils import parse_sentense_with_stanford

connective_list = {'before', 'afterward', 'next', 'then', 'till', 'until', 'after', 'earlier', 'once', 'previously',
                   'meantime', 'meanwhile', 'simultaneously', 'because', 'for', 'accordingly', 'consequently', 'so',
                   'thus', 'therefore', 'if', 'but', 'conversely', 'however', 'nonetheless', 'although', 'and',
                   'additionally', 'also', 'besides', 'further', 'furthermore', 'similarly', 'likewise', 'moreover',
                   'plus', 'specifically', 'alternatively', 'or', 'otherwise', 'unless', 'instead', 'except'}

clause_words = ['when', 'who', 'what', 'where', 'how', 'When', 'Who', 'What', 'Where', 'How', 'why', 'Why', 'which',
                'Which', '?']

def filter_special_case(extracted_eventualities):
    extracted_eventualities['s-v-a'] = []
    extracted_eventualities['s-v-be-o'] = []
    if len(extracted_eventualities['s-v-v']) > 0:
        tmp_s_v_v = list()
        tmp_s_v_a = list()
        for e in extracted_eventualities['s-v-v']:
            for edge in e.parsed_relations:
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
            for edge in e.parsed_relations:
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
            for edge in e.parsed_relations:
                if edge[1] == 'nsubj':
                    if edge[0][0] > edge[2][0] or edge[0][1] == 'be':
                        tmp_s_v.append(e)
        extracted_eventualities['s-v'] = tmp_s_v
    for relation in extracted_eventualities:
        new_eventualities = list()
        for tmp_e in extracted_eventualities[relation]:
            found_connective = False
            for edge in tmp_e.parsed_relations:
                if edge[2][1] in connective_list:
                    found_connective = True
                    break
            if found_connective:
                new_e = Activity(None)
                new_e.skeleton_parsed_relations = tmp_e.skeleton_parsed_relations
                new_edges = list()
                for edge in tmp_e.parsed_relations:
                    if edge[2][1] in connective_list:
                        continue
                    new_edges.append(edge)
                new_e.parsed_relations = new_edges
                new_e.find_skeleton_words()
                tmp_e = new_e
            if len(tmp_e.parsed_relations) > 0:
                new_eventualities.append(tmp_e)
        extracted_eventualities[relation] = new_eventualities

    return extracted_eventualities


def match_rule_r_and_dep_r(rule_r, dep_r, current_dict):
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


def extract_activity_with_fixed_target(parsed_result, activity_rule, verb_position, rule_name):
    selected_edges = list()
    selected_skeleton_edges = list()
    local_dict = {'V1': verb_position}
    for tmp_rule_r in activity_rule.positive_rules:
        foundmatch = False
        for dep_r in parsed_result:
            decision, local_dict = match_rule_r_and_dep_r(tmp_rule_r, dep_r, local_dict)
            if decision:
                selected_edges.append(dep_r)
                selected_skeleton_edges.append(dep_r)
                foundmatch = True
                break
        if not foundmatch:
            # print('Miss one positive relation')
            return None
    for tmp_rule_r in activity_rule.possible_rules:
        for dep_r in parsed_result:
            decision, local_dict = match_rule_r_and_dep_r(tmp_rule_r, dep_r, local_dict)
            if decision:
                selected_edges.append(dep_r)
    for tmp_rule_r in activity_rule.negative_rules:
        for dep_r in parsed_result:
            if dep_r in selected_edges:
                # print('This edge is selected by the positive example, so we will skip it')
                continue
            decision, local_dict = match_rule_r_and_dep_r(tmp_rule_r, dep_r, local_dict)
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


def extract_activities_from_parsed_result_with_single_rule(parsed_result, activity_rule,
                                                           rule_name):
    local_activities = list()
    verb_positions = list()
    for relation in parsed_result:
        if 'VB' in relation[0][2]:
            verb_positions.append(relation[0][0])
        if 'VB' in relation[2][2]:
            verb_positions.append(relation[2][0])

    verb_positions = list(set(verb_positions))
    for verb_position in verb_positions:
        tmp_a = extract_activity_with_fixed_target(
            parsed_result, activity_rule, verb_position, rule_name)
        if tmp_a is not None:
            local_activities.append(tmp_a)
    return local_activities


def extract_activities_from_parsed_result(parsed_result, activity_rules):
    all_activities = []
    for rule_name in activity_rules:
        tmp_activities = extract_activities_from_parsed_result_with_single_rule(parsed_result,
                                                                                activity_rules[rule_name],
                                                                                rule_name)
        all_activities.extend(tmp_activities)
    return all_activities


class EventualityExtractor(object):
    def __init__(self, port):
        self.corenlp_client = CoreNLPClient(
            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'], timeout=60000,
            memory='5G', endpoint="http://localhost:%d" % port,
            start_server=True, be_quiet=False)
        self.corenlp_client.annotate("hello world")
        pass

    def close(self):
        self.corenlp_client.stop()

    def extract(self, sentence):
        """ extract all the eventualities from an input sentence
                Args:
                    s: string, the input sentence.

                Returns:
                    activities: list, [
                    {"sentence_dependencies": ...,
                     "sentence_tokens": ...,
                     "eventuality_list": ...}, ...]
                """
        rst_list = []
        sentences = parse_sentense_with_stanford(sentence, self.corenlp_client)
        for sentence in sentences:
            eventualities = extract_activities_from_parsed_result(
                sentence["dependencies"], All_activity_rules)
            tmp = [e.to_dict() for e in eventualities]
            rst_list.append(
                {"sentence_dependencies": sentence["dependencies"],
                 "sentence_tokens": sentence["tokens"],
                 "eventuality_list": tmp})
        return rst_list

    def extract_from_dependencies(self, dependencies):
        pass