import re

import numpy as np
from tqdm import trange

from glucose_utils import Unusable, glucose_group_list, ATOMIC_group_list, \
    ATOMIC_subject_list
from atomic_utils import PP_SINGLE


def trim(s):
    """
    This function get rid of the empty space at the beginning and end of the string
    :param s: A string to be trimmed
    :return: The string after trimming.
    """
    if s.startswith(' ') or s.endswith(' '):
        return re.sub(r"^(\s+)|(\s+)$", "", s)
    return s


def extract_a_structure(stru: str):
    """
    This function extracts the knowledge in GLUCOSE dataset, please check the structure in GLUCOSE general structure
    columns for detail.
    :param stru: We take a structure string from GLUCOSE (int)_GeneralStructure column, and we split if by different
    character to make it a valid dict for parsing.
    :return: A dict of separated structure. The key should be 'subject', 'object', 'preposition', 'object', etc.
    That appeared in the general structure.
    """
    structure = {}
    for part in trim(stru).split(']'):
        if part.split('[')[-1] == '':
            continue
        elif '||' in part:
            structure[part.split('[')[-1]] = trim(part.split('||')[0].replace('{', ''))
        else:
            structure[part.split('[')[-1]] = trim(part.split('[')[0].replace('{', '').replace('}_', '').split('||')[0])
    return structure


def structure_to_pure_text(stru: dict):
    """
    This function change our extracted structure back to pure text
    :param stru: The dict structure we just got using the function extract_a_structure(str)
    :return: Return a string text. For example {'subject':'I', 'verb':'love', 'preposition':'you'} will output
    'I love you'
    """
    parts = []
    for key in stru.keys():
        parts.append(stru[key])
    return " ".join(parts)


def get_substitute_single_subject(rules: dict):
    """
    This function will choose a single subject word that hasn't exist in the replacement rule dict we already have.
    :param rules: Current replacement dict.
    :return: A single subject word that is not in the current replacement dict.
    """
    for i in ATOMIC_subject_list:
        if i not in rules.values():
            return i


def get_substitute_group_subject(rules: dict):
    """
    This function will choose a group subject word that hasn't exist in the replacement rule dict we already have.
    :param rules: Current replacement dict.
    :return: A group subject word that is not in the current replacement dict.
    """
    for i in ATOMIC_group_list:
        if i not in rules.values():
            return i


def replace_by_dict(rule: dict, head: str, tail: str):
    """
    This function replaces the head and tail string by the given replacement rule dict.
    :param rule: The replacement rule dictionary we have built.
    :param head: The head string
    :param tail: The tail string
    :return: The replaced head string and replaced tail string, return two strings.
    """
    head_result, tail_result = head.lower(), tail.lower()
    for i in rule.keys():
        head_result = " ".join([rule[i] if j == i else j for j in head_result.split(' ')])
        tail_result = " ".join([rule[i] if j == i else j for j in tail_result.split(' ')])
    return head_result, tail_result


def generate_aser_to_glucose_dict(head: str, tail: str, full_return=True):
    """
    This function generates the replacement dictionary from ASER form to CKGP form. For example, 'I love you' will be
    replaced to 'PersonX love PersonY'.
    :param head: The head string in ASER form. e.g. I love you
    :param tail: The tail string in ASER form. e.g. You love I
    :param full_return: Whether return the replaced head, replaced tail, replaced head + tail as not.
    If true, return one dict and three strings. If false, only a dict.
    :return: return a dict of replacement rules from ASER to CKGP. If full_return is true, then also return the
    replaced head, replaced tail, replaced head + replaced tail, joined by a comma ', '

    Note that For object, 'it' and 'that' are not covered. They will not be reversed
    For place, 'there' is not taken care of, as it's widely used in many ways.
    """
    rule = {}
    # deal with "my". In stanford parser the lemma of `my' is still `my'
    head = " ".join(["i" if token == "my" else token for token in head.split()])
    tail = " ".join(["i" if token == "my" else token for token in tail.split()])
    
    # Make the head subject
    head_subj = head.split()[0]
    tail_subj = tail.split()[0]
    if head_subj in PP_SINGLE and tail_subj in PP_SINGLE and head_subj!=tail_subj:
        rule[head_subj] = get_substitute_single_subject(rule)
        rule[tail_subj] = get_substitute_single_subject(rule)

    tokens = (head.lower() + ' ' + tail.lower()).split(' ')

    for t in tokens:
        if t in PP_SINGLE:
            if t in rule.keys():
                continue
            else:
                rule[t] = get_substitute_single_subject(rule)
        if t in glucose_group_list:
            if t in rule.keys():
                continue
            else:
                rule[t] = get_substitute_group_subject(rule)
    replaced_head, replaced_tail = replace_by_dict(rule, head, tail)
    if full_return:
        return rule, replaced_head, replaced_tail, replaced_head + ', ' + replaced_tail
    else:
        return rule


def generate_unusable_knowledge_index_list():
    """
    This is a function that generates the unusable knowledge index for every list. In our parsing process, there are
    several special characters that we can not take care of, so we choose to ignore these knowledge and don't use it.
    :return: The failing index dict for every list. the key is ['head','tail'][range(1,11)], and the value is a list
    of all the unusable index.
    """
    glucose = np.load('../../dataset/Glucose_parsed_stru_dict.npy', allow_pickle=True).item()
    fail_index = {'head': {}, 'tail': {}}
    for i in range(1, 11):
        for j in ['head', 'tail']:
            fail_index[j][i] = []
    for part in ['head', 'tail']:
        for i in trange(1, 11):
            for ind, k in enumerate(glucose[part][i]):
                tokens = structure_to_pure_text(extract_a_structure(k)).split(' ')
                if any(item in tokens for item in Unusable):
                    fail_index[part][i].append(ind)
    print(fail_index)
    print("There are {} knowledges filtered out".format(
        sum(len(part_dict[j]) for j in range(1, 11) for part_dict in [fail_index['head'], fail_index['tail']])))
    np.save('../build_graph/unusable_index.npy', fail_index)
    return fail_index
