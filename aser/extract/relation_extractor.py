from aser.eventuality import Eventuality

RELATION_TYPES = [
    'Precedence', 'Succession', 'Synchronous',
    'Reason', 'Result',
    'Condition', 'Contrast', 'Concession',
    'Conjunction', 'Instantiation', 'Restatement', 'ChosenAlternative', 'Alternative', 'Exception',
    'Co_Occurrence']


RELATION_SEED_CONNECTIVES = {
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
    'Co_Occurrence': []
}


class BaseRelationExtractor(object):
    def __init__(self):
        pass

    def extract(self, eventuality_pair):
        raise NotImplementedError


class SeedRuleRelationExtractor(BaseRelationExtractor):
    def __init__(self):
        super().__init__()
        pass

    def extract(self, sentences):
        """ This methods extract relations among extracted eventualities based on seed rule

            :type sentences: list
            :param sentences: list of (sentence_parsed_result, EventualityList)
            :return: list of triples of format (eid1, relation, eid2)

            .. highlight:: python
            .. code-block:: python

                Input Example:


                Output Example:
                    [('a53fd728f8a4dd955e7ed2bd72ff07ffabb8e7f5',
                      'Co_Occurrence',
                      'c08b06c1b3a3e9ada88dd7034618d0969ae2b244'),
                     ('a53fd728f8a4dd955e7ed2bd72ff07ffabb8e7f5',
                      'Reason',
                      'c08b06c1b3a3e9ada88dd7034618d0969ae2b244')
        """
        rst_extracted_relations = []
        for sent_parsed_result, eventualities in sentences:
            for head_eventuality in eventualities:
                for tail_eventuality in eventualities:
                    if head_eventuality.position < tail_eventuality.position:
                        extracted_relations = self._extract_from_eventuality_pair_in_one_sentence(
                            head_eventuality, tail_eventuality, sent_parsed_result)
                        for rel in extracted_relations:
                            heid = head_eventuality.eid
                            teid = tail_eventuality.eid
                            rst_extracted_relations.append((heid, rel, teid))

        for i in range(len(sentences) - 1):
            s1_eventuality_list = sentences[i][1]
            s2_eventuality_list = sentences[i + 1][1]
            if len(s1_eventuality_list) > 1 or len(s2_eventuality_list) > 1:
                continue
            s1_sentence_tokens = sentences[i][0]["tokens"]
            s2_sentence_tokens = sentences[i + 1][0]["tokens"]
            for head_eventuality in s1_eventuality_list:
                for tail_eventuality in s2_eventuality_list:
                    if not isinstance(head_eventuality, Eventuality):
                        tmp = head_eventuality
                        head_eventuality = Eventuality()
                        head_eventuality = head_eventuality.from_dict(tmp)
                    if not isinstance(tail_eventuality, Eventuality):
                        tmp = tail_eventuality
                        tail_eventuality = Eventuality()
                        tail_eventuality = tail_eventuality.from_dict(tmp)
                    extracted_relations = self._extract_from_eventuality_pair_in_two_sentence(
                        head_eventuality, tail_eventuality,
                        s1_sentence_tokens,
                        s2_sentence_tokens)
                    for rel in extracted_relations:
                        heid = head_eventuality.eid
                        teid = tail_eventuality.eid
                        rst_extracted_relations.append((heid, rel, teid))

        return rst_extracted_relations


    def _extract_from_eventuality_pair_in_one_sentence(self,
                                                       head_eventuality,
                                                       tail_eventuality,
                                                       sent_parsed_result):
        extracted_relations = ['Co_Occurrence']
        for relation_type in RELATION_TYPES:
            for connective_words in RELATION_SEED_CONNECTIVES[relation_type]:
                if self._verify_connective_in_one_sentence(
                    connective_words, head_eventuality, tail_eventuality,
                    sent_parsed_result["dependencies"],
                    sent_parsed_result["tokens"]):
                    extracted_relations.append(relation_type)
                    break
        return extracted_relations

    def _extract_from_eventuality_pair_in_two_sentence(self,
                                                      head_eventuality,
                                                      tail_eventuality,
                                                      s1_sentence_tokens,
                                                      s2_sentence_tokens):
        extracted_relations = list()
        for relation_type in RELATION_TYPES:
            for connective_words in RELATION_SEED_CONNECTIVES[relation_type]:
                if self._verify_connective_in_two_sentence(
                        connective_words,
                        head_eventuality, tail_eventuality,
                        s1_sentence_tokens, s2_sentence_tokens):
                    extracted_relations.append(relation_type)
                    break

        return extracted_relations

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
        e1_position, e2_position = head_eventuality.position + 1, tail_eventuality.position + 1
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


class NeuralRelationExtractor(BaseRelationExtractor):
    def __init__(self):
        super().__init__()
        raise NotImplementedError

    def extract(self, eventualtity_pair):
        raise NotImplementedError