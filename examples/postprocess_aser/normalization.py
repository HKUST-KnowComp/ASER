# coding: utf-8
# author: Jiayang Cheng
from collections import defaultdict
from dataclasses import replace
from email.policy import default
from tokenize import group

PP_SINGULAR = [
    'i', 'you', 'he', 'she', 'someone', 'guy', 'man', 'woman', 'somebody'
]
PP_PLURAL = ['they', 'we']
ATOMIC_SINGULAR = ['PersonX', 'PersonY', 'PersonZ', 'PersonA', 'PersonB']
ATOMIC_PLURAL = ['PeopleX', 'PeopleY']

def get_substitution(rules, from_list):
    """
    This function will choose a single subject word that hasn't exist in the replacement rule dict we already have.
    :param rules: Current replacement dict.
    :param from_list: The substitution alternatives.
    :return: A single subject word that is not in the current replacement dict.
    """
    for i in from_list:
        if i not in rules.values():
            return i

def replace_by_dict(mapping, node_str):
    """
    This function replaces the string by the given replacement rule dict.
    :param mapping: The replacement rule dictionary we have built.
    :param node_str: The target node string to be replaced
    :return: The replaced string
    """

    return ' '.join([(mapping[i] if i in mapping else i)
                     for i in node_str.split(' ')])

class BasicPronounNormalizer:
    ''' Basic pronoun normalizer used in CKGP. '''

    def __init__(self):
        self.get_substitute_singular = lambda rules: get_substitution(
            rules, ATOMIC_SINGULAR)
        self.get_substitute_plural = lambda rules: get_substitution(
            rules, ATOMIC_PLURAL)

    def find_pronoun_subject(self, node_str):
        """
        Find pronoun subject within a given node_str. 
        Return the subject str if found.
        """
        return node_str.split()[0]

    def pronoun_normalize(self, head: str, tail: str, full_return=True):
        """
        Adapted from method ``generate_aser_to_glucose_dict''.

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
        head = replace_by_dict({'my': 'i'}, head)
        tail = replace_by_dict({'my': 'i'}, tail)

        # find subject str in node (naive implementation, use first word)
        head_subj = self.find_pronoun_subject(head)
        tail_subj = self.find_pronoun_subject(tail)

        # if the subjects are all singular person-pronouns, assign mapping to PersonX, PersonY, etc.
        if head_subj in PP_SINGULAR and tail_subj in PP_SINGULAR and head_subj != tail_subj:
            rule[head_subj] = self.get_substitute_singular(rule)
            rule[tail_subj] = self.get_substitute_singular(rule)

        tokens = (head + ' ' + tail).split(' ')

        # find and normalize other pronouns (either singular or plural)
        for t in tokens:
            if t in PP_SINGULAR:
                if t in rule: continue
                else: rule[t] = self.get_substitute_singular(rule)

            if t in PP_PLURAL:
                if t in rule: continue
                else: rule[t] = self.get_substitute_plural(rule)

        replaced_head = replace_by_dict(rule, head)
        replaced_tail = replace_by_dict(rule, tail)

        if full_return:
            return rule, replaced_head, replaced_tail, replaced_head + ', ' + replaced_tail
        else:
            return rule

class ParsingBasedNormalizer:
    '''
    Using parsing information to normalize nodes
    '''

    def __init__(self):
        self.WP_WORDS = {
                'who', 'whom', 'whose', 'whoever', 'whomever'
        }
        # self.INDEFINITE_WORDS = {'woman', 'men', 'women', \
        #                     'anybody', 'everybody', 'somebody', 'nobody', 'anyone', 'everyone', 'someone'}
        self.POSSESSIVE_PRP = {'my', 'mine', 'your', 'yours', 'his', 'hers', 'ours', 'theirs'}
        # self.INDEFINITE_WORDS = {'woman', 'men', 'women', \
        #                         'i', 'me', 'myself', 'my', 'mine', \
        #                         'you', 'yourself', 'your', 'yours', \
        #                         'he', 'him', 'himself', 'his', \
        #                         'she', 'her', 'herself', 'hers', \
        #                         'we', 'us', 'ourself', 'ourselves', 'ours', \
        #                         'they', 'them', 'themself', 'themselves', 'their', 'theirs'}
        # self.INDEFINITE_WORDS_SP = {'one', 'all', 'man', 'person', 'guy',\
        #     'anybody', 'everybody', 'somebody', 'nobody', 'anyone', 'everyone', 'someone'}   # effective only when occurs as subject
        self.INDEFINITE_WORDS = {'woman', 'men', 'women', \
                                'i', 'me', 'myself', 'my', 'mine', \
                                'you', 'yourself', 'your', 'yours', \
                                'he', 'him', 'himself', 'his', \
                                'she', 'her', 'herself', 'hers', \
                                'we', 'us', 'ourself', 'ourselves', 'ours', \
                                'they', 'them', 'themself', 'themselves', 'their', 'theirs',\
                                'anybody', 'everybody', 'somebody', 'nobody', 'anyone', 'everyone', 'someone'}
        self.INDEFINITE_WORDS_SP = {'one', 'all', 'man', 'person', 'guy'}   # effective only when occurs as subject
        self.INDEFINITE_WORDS_ALL = self.INDEFINITE_WORDS_SP | self.INDEFINITE_WORDS

        self.PRP_COREF = [['i', 'me', 'myself', 'my', 'mine'],\
                           ['you', 'yourself', 'your', 'yours'],\
                            ['he', 'him', 'himself', 'his', 'man'], \
                            ['she', 'her', 'herself', 'hers', 'woman'], \
                            ['we', 'us', 'ourself', 'ourselves', 'ours'], \
                            ['they', 'them', 'themself', 'themselves', 'their', 'theirs'],\
                            ['anybody', 'everybody', 'anyone', 'everyone', 'all']]
        self.PRP_COREF = {item:i for i, items in enumerate(self.PRP_COREF) for item in items }
        self.PRP_SUBSET = {0:{4}}

    def _restore_span(self,
                      root: tuple,
                      words: list = None,
                      compounds=None,
                      modifiers=None,
                      cases=None):
        """
        Restore the subject/object span based on root and compounds.

        Inputs:
            <root:tuple> the target (core) word of the span. The span will be recovered centered on this word. 
                        The input should be a tuple in the dependencies.
            <words:list> if provided, an additional word list of the recovered span will be provided as the last return parameter.

            The inputs below should be like the outputs of self._get_data_from_dep.
            <compounds> if provided, the compound info will be considered to recover the span.
            <modifiers> if provided, the modifier info will be considered to recover the span.
            <cases> if provided, the case info will be considered (e.g., someone ['s]) to recover the span.
        
        Outputs:
            <span:tuple> a span (tuple of two indices) indicating the start and end index of the recovered span.
            <details:dict> other details (e.g., the target word index).
            
            (if "words" is input)
            <word span: list of str> list of words inside the span.
        """

        root_index = root[0]
        leftmost, rightmost = root_index, root_index + 1

        details = defaultdict(list)
        details['target'].append(root_index)
        details['target_word'].append(words[root_index])

        if compounds and (root in compounds):
            comp = compounds[root]
            for i in comp:
                details['compounds'].append(i[0])

        # if modifiers:
        #     # possessives, nmodof
        #     for nmod_str in ('nmod:poss', 'nmod:of'):
        #     # for nmod_str in ('nmod:of',):
        #         if nmod_str in modifiers and (root in modifiers[nmod_str]):
        #             nmod_list = modifiers[nmod_str][root]

        #             details[nmod_str].append(nmod_list[0][0])

        if cases and (root in cases):
            case_n = cases[root]
            for i in case_n:
                if i[1] == "'s":
                    details['case'].append(i[0])

        for rel_type in details:
            for i in details[rel_type]:
                if isinstance(i, int):
                    leftmost = min(i, leftmost)
                    rightmost = max(i + 1, rightmost)

        details = dict(details)
        if words:
            return (leftmost, rightmost), details, words[leftmost:rightmost]
        else:
            return (leftmost, rightmost), details

    def _get_data_from_dep(self, dependency):
        # collect compounds, modifiers, cases from dependency

        modifiers = dict()
        compounds = defaultdict(list)
        cases = defaultdict(list)
        for h, rel, t in dependency:
            if rel == 'compound': compounds[h].append(t)  # compound
            if 'mod' in rel:  # modifiers
                if rel not in modifiers:
                    modifiers[rel] = defaultdict(list)
                modifiers[rel][h].append(t)
            if rel == 'case':  # case
                cases[h].append(t)
        return compounds, modifiers, cases

    def _get_subj_obj_from_dep(self, skeleton_dependency):
        # collect subject and object data from skeleton dependency
        subj = None
        objs = []
        for h, rel, t in skeleton_dependency:
            if rel.startswith('nsubj'): subj = t  # nsubj, nsubjpass
            if rel[1:].startswith('obj'): objs.append(t)  # iobj, dobj ...

            # pattern specific, e.g., s-be-o
            if rel == 'cop' and h[2] in {
                    'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'WP', 'WP$', 'JJ'
            }:
                objs.append(h)
        return subj, objs

    def classify_personal(self,
                          index: int,
                          info: dict,
                          is_subj: bool = False):
        # classify whether the input word in eventuality is personal words
        # [1] personal pronouns [PoS in {PRP, PRP$, WP, WP$}]
        # (PRP, PRP$, e.g., i you he she, oneself...)
        # (WP & == who, whoever, whomever)
        #
        # [2] other words [PoS in NN, NNS, NNP, NNPS]
        # ('one', 'man', 'woman', 'people', 'person', 'guy' as subj/obj)
        # indefinite pronouns (e.g., someone, somebody, nobody, anybody, ..., ...)
        #
        # NER results ? e.g. Dave, Marry
        #
        # others? e.g. economist, dentist, official, mother, father, child, ...
        root = (index, info['words'][index], info['pos_tags'][index])
        ners = info['ners']
        mentions = info['mentions']
        personal_spans = []
        for span in mentions:
            if mentions[span]['ner'] == 'PERSON':
                personal_spans.append(span)

        pos_tag = 'PRP$' if root[1] in self.POSSESSIVE_PRP else root[2]
        true_returns = {'is_person': True, 'word': root[1], 'pos_tag': pos_tag, 'index': index}
        false_returns = {'is_person': False, 'word': root[1], 'pos_tag': pos_tag, 'index': index}

        # pronouns
        if root[2] in {'PRP', 'PRP$'} and (root[1] != 'it'):
            return true_returns
        if root[2] in {'WP', 'WP$'} and (root[1] in self.WP_WORDS):
            return true_returns

        # other PoS
        if root[2] in {'NN', 'NNS', 'NNP', 'NNPS', 'PRP', 'PRP$', 'FW', 'LS'}:
            # definitely human
            if root[1] in self.INDEFINITE_WORDS:
                return true_returns
            # is human only when in certain position
            if (root[1] in self.INDEFINITE_WORDS_SP) and is_subj:
                return true_returns

            # NER results?
            if (ners and ners[index] == 'PERSON') or (personal_spans and any([l <= index and index < r for l,r in personal_spans])):
                return true_returns

        return false_returns

    def process_info(self, info: dict):
        # process a single info dict for aser nodes

        dependency = info['dependencies']
        skeleton_dep = info['skeleton_dependencies']

        subj, objs = self._get_subj_obj_from_dep(skeleton_dep)
        compounds, modifiers, cases = self._get_data_from_dep(dependency)
        if subj is None: return None

        returns = defaultdict(list)
        # restore subject
        span, details, words = self._restore_span(subj, info['words'],
                                                  compounds, modifiers)
        returns['subj'].append((span, details, words))

        # restore object
        for obj in objs:
            span, details, words = self._restore_span(obj, info['words'],
                                                      compounds, modifiers)
            returns['objs'].append((span, details, words))
        
        # restore nmod:poss (possessitve modifier), e.g [someone's] hat
        if 'nmod:poss' in modifiers:
            for nmod_poss_list in modifiers['nmod:poss'].values():
                for poss in nmod_poss_list:
                    span, details, words = self._restore_span(poss, info['words'],
                                                        compounds, modifiers,
                                                        cases)
                    returns['possessive'].append((span, details, words))
        
        # restore other nominal words (other htan nmod:poss)
        for nmod in modifiers:
            if nmod != 'nmod:poss':
                for nmod_list in modifiers[nmod].values():
                    for modifier in nmod_list:
                        span, details, words = self._restore_span(modifier, info['words'],
                                                        compounds, modifiers)
                        returns['nmod'].append((span, details, words))

        return returns

    def get_subj_objs(self, info, return_root=False):
        data = self.process_info(info)

        if data:
            if return_root:
                return data['subj'][0], data['objs']
            else:
                subj = ' '.join(data['subj'][0][2])
                objs = [' '.join(i[2]) for i in data['objs']]
                return subj, objs
    
    def get_personal_words(self, info: dict):
        """
        Find all personal words in a given node (info dict)

        Return all the spans that are classified as personal words.
            <defaultdict>: {'default': [list of personal (span, details, words)], 'possessive': [list of personal (span, details, words)]}
        """
        processed_info = self.process_info(info)
        if processed_info is None: return {}
        
        subj, obj_list, poss_list, nmod_list = processed_info.get('subj', []), processed_info.get('objs', []),\
                                                processed_info.get('possessive', []), processed_info.get('nmod', [])

        all_personal_spans = []
        def _proc_item(item, is_subj=False, label='default'):
            item_index = item[1]['target'][0]
            personal_info = self.classify_personal(item_index, info, is_subj=is_subj)
            if personal_info['pos_tag'] in {'PRP$', 'WP$'}:
                label = 'possessive'
            if personal_info['is_person']:
                all_personal_spans.append((item, label))

        # subjects or objects
        for item in subj:
            _proc_item(item, is_subj=True)
        for item in obj_list:
            _proc_item(item)
        
        # possessive
        for item in poss_list:
            _proc_item(item, label='possessive')
        
        # nmod
        for item in nmod_list:
            _proc_item(item)
        
        return all_personal_spans

    def node_person_coref(self, personal_spans: list, info: dict):
        """
        Identify the coreference among people within node.

        Inputs:
            <personal_spans: list> The output from self.get_personal_words
            <info: dict> The info of node stored in ASER
        Outputs:
            <dict> 'persons': grouped personal info, with each entry being a list of a given person's mentions
                    ...(TODO)

        TODO: 
        1- coreference relationship between NER results and Pronouns
        2- improve implementing subset relationship 
        """
        coref_list = []
        for compared_people, comp_label in personal_spans:
            for i in range(len(coref_list)):
                group = coref_list[i]
                for people, _ in group:
                    
                    # if ' '.join(people[2]) not in self.INDEFINITE_WORDS_SP: # indefinite pronouns cannot be regard as the same , e.g. somebody v.s. somebody
                    # words in span exact match
                    if (len(compared_people[2]) == len(people[2])) and all([compared_people[2][i]==people[2][i] for i in range(len(people[2]))]):
                        group.append((compared_people, comp_label))
                        break
                    # target word (core) exact match
                    compared_target_word = info['words'][compared_people[1]['target'][0]]
                    people_target_word = info['words'][people[1]['target'][0]]
                    if compared_target_word == people_target_word:
                        group.append((compared_people, comp_label))
                        break
                    # special cases, e.g. {i me myself my mine}, {you yourself yours}, {he him himself his}, {she her herself hers}, {they them themself their theirs}, {we us ourself our ours}, {one oneself ones}
                    prp_coref_comp, prp_coref_peop = self.PRP_COREF.get(compared_target_word, None), self.PRP_COREF.get(people_target_word, None)
                    if  (prp_coref_comp is not None) and (prp_coref_peop is not None) and (prp_coref_comp == prp_coref_peop):
                        group.append((compared_people, comp_label))
                        break
                    # TODO: complex cases
                else:
                    continue
                break
            else:
                group = [(compared_people, comp_label)]
                coref_list.append(group)
        persons = {'P{}'.format(i):group_info for i, group_info in enumerate(coref_list)}

        subset = [] # TODO subset relation between person/people
        for p_h in persons:
            for p_t in persons:
                if p_h == p_t: continue
                # group level compare
                for p_h_mention, _ in persons[p_h]:
                    for p_t_mention, _ in persons[p_t]:
                        # is subset?
                        
                        # special case: i(me, ...) -subset-> we (us, our, ...)
                        p_h_target, p_t_target = p_h_mention[1]['target_word'][0], p_t_mention[1]['target_word'][0]
                        prp_coref_h, prp_coref_t = self.PRP_COREF.get(p_h_target, None), self.PRP_COREF.get(p_t_target, None)
                        if (prp_coref_h is not None) and (prp_coref_t is not None) and\
                             (prp_coref_h in self.PRP_SUBSET) and \
                                (prp_coref_t in self.PRP_SUBSET[prp_coref_h]):
                            subset.append((p_h, p_t))
                            break
                    else:
                        continue
                    break

        return {'persons': persons, 'subset': subset}

    def pair_person_coref(self, head_persons:str, tail_persons:str, head_info:dict, tail_info:dict):
        """
        Identify the coreference based on a pair of (head, _, tail) personal info.

        Inputs:
            <head_persons, tail_persons: dict> The outputs from self.node_person_coref
            <head_info, tail_info> The info dicts stored in ASER

        TODO: 
        1- subset relationship among personal words between nodes?
        """

        def _group_align(head_p, tail_p):
            ''' judge whether two person groups are "same", "diff" or "h_subset_t", "t_subset_h"
            TODO: 
            subset judgement?
            improve 'same' case detection
            '''
            for h_mention in head_persons['persons'][head_p]:
                for t_mention in tail_persons['persons'][tail_p]:
                    # all words are the same
                    head_all_words = h_mention[0][2]
                    tail_all_words = t_mention[0][2]
                    if all(i==j for i, j in zip(head_all_words, tail_all_words)):
                        return 'same'
                    
                    # all target words are the same
                    head_target_words = h_mention[0][1]['target_word']
                    tail_target_words = t_mention[0][1]['target_word']
                    if all(i==j for i, j in zip(head_target_words, tail_target_words)):
                        return 'same'
                    
                    # in the same coref group, e.g. I, me, my, mine, myself...
                    for h_wrd, t_wrd in zip(head_target_words, tail_target_words):

                        h_prp_coref, t_prp_coref = self.PRP_COREF.get(h_wrd, None), self.PRP_COREF.get(t_wrd, None)
                        # print(h_prp_coref, t_prp_coref)
                        if (h_prp_coref is not None) and (t_prp_coref is not None):
                            if (h_prp_coref == t_prp_coref):
                                return 'same'

                            # subset relation
                            if h_prp_coref in self.PRP_SUBSET and \
                                t_prp_coref in self.PRP_SUBSET[h_prp_coref]:
                                return 'h_subset_t'
                            
                            if t_prp_coref in self.PRP_SUBSET and \
                                h_prp_coref in self.PRP_SUBSET[t_prp_coref]:
                                return 't_subset_h'
                            
                    # TODO: other coreference?
            return 'diff'

        pair_coref = defaultdict(set)
        for head_p in head_persons['persons']:
            # print(head_group_list)
            for tail_p in tail_persons['persons']:
                label = _group_align(head_p, tail_p) # label could be "same", "diff", "h_subset_t", "t_subset_h"
                
                head = 'H_'+head_p
                tail = 'T_'+tail_p
                if label == "same":
                    pair_coref[label].add((head, tail))
                if label == 'h_subset_t':
                    pair_coref['subset'].add((head, tail))
                if label == 't_subset_h':
                    pair_coref['subset'].add((tail, head))
        # subset relatinship expanded by "same" 
        for head, tail in pair_coref['same']:
            for h_p1, h_p2 in head_persons['subset']:
                if head[2:] == h_p1:
                    pair_coref['subset'].add((tail, 'H_'+h_p2))
                if head[2:] == h_p2:
                    pair_coref['subset'].add(('H_'+h_p1, tail))
            for t_p1, t_p2 in tail_persons['subset']:
                if tail[2:] == t_p1:
                    pair_coref['subset'].add((head, 'T_'+t_p2))
                if tail[2:] == t_p2:
                    pair_coref['subset'].add(('T_'+t_p1, head))
        pair_coref = dict(pair_coref)
        if 'same' not in pair_coref: pair_coref['smae'] = []
        if 'subset' not in pair_coref: pair_coref['subset'] = []
        return pair_coref

    def get_norm_node(self, node, personal_coref, show_possessive:bool=True):
        """ Get the normalized node name
        Input:
            <node:str> the original node name (before normalization)
            <personal_coref> the output from self.node_person_coref
        Output:
            <str> the normalized node's name
            <dict> mapping from person (e.g. P0) to list of indices
        """
        if len(personal_coref['persons']) == 0:
            return {'norm_node': node, 'p2i': {}}
        replace_dict = {}
        span2pers = {}
        for p in personal_coref['persons']:
            for p_mention, label in personal_coref['persons'][p]:
                span = p_mention[0]
                span2pers[span] = p
                pers_tok = '['+(p+"'s" if (show_possessive and label=='possessive') else p)+']'
                replace_dict[span] = pers_tok
        
        spans = list(replace_dict.keys())
        spans.sort(key=lambda x: x[0])

        try:
            assert all([spans[i][1]<=spans[i+1][0] for i in range(len(spans)-1)])# no overlap
        except:
            # remove the latter one to prevent overlap
            i = 0
            while i < len(spans)-1:
                if spans[i][1] > spans[i+1][0]:
                    spans.pop(i+1)
                else:
                    i += 1
            
        node_wrds = node.split(' ')
        norm_node_wrds = []
        p2indices = defaultdict(list)
        last_j = 0
        for i, j in spans:
            norm_node_wrds.extend(node_wrds[last_j:i])
            p2indices[span2pers[(i,j)]].append(len(norm_node_wrds))
            norm_node_wrds.append(replace_dict[(i,j)])
            last_j = j
        norm_node_wrds.extend(node_wrds[last_j:])
        return {'norm_node': ' '.join(norm_node_wrds), 'p2i': dict(p2indices)}

    def proc_and_print(self, info_list):
        for info in info_list:
            info = eval(info)

            data = self.process_info(info)

            print(' '.join(info['words']))
            if data:
                subj = ' '.join(data['subj'][0][2])
                objs = [' '.join(i[2]) for i in data['objs']]
                print('[SUBJ]:{}\t[OBJS]: {}'.format(subj, ', '.join(objs)))