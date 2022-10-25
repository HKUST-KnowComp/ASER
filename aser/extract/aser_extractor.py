from copy import copy, deepcopy
from itertools import chain
from .eventuality_extractor import SeedRuleEventualityExtractor, DiscourseEventualityExtractor
from .relation_extractor import SeedRuleRelationExtractor, DiscourseRelationExtractor
from .utils import parse_sentense_with_stanford, get_corenlp_client
from .utils import ANNOTATORS


class BaseASERExtractor(object):
    """ Base ASER Extractor to extract both eventualities and relations.
    It includes an instance of `BaseEventualityExtractor` and an instance of `BaseRelationExtractor`.

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

        self.eventuality_extractor = None
        self.relation_extractor = None

    def close(self):
        """ Close the extractor safely
        """

        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()
        if self.eventuality_extractor:
            self.eventuality_extractor.close()
        if self.relation_extractor:
            self.relation_extractor.close()

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

    def extract_eventualities_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, use_lemma=True, **kw):
        """ Extract eventualities from the parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param use_lemma: whether the returned eventuality uses lemma
        :type use_lemma: bool (default = True)
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
            raise ValueError(
                "Error: extract_eventualities_from_parsed_result only supports Eventuality or json."
            )

        return self.eventuality_extractor.extract_from_parsed_result(
            parsed_result, output_format=output_format, in_order=in_order, use_lemma=use_lemma, **kw
        )

    def extract_eventualities_from_text(self, text, output_format="Eventuality", in_order=True, use_lemma=True, annotators=None, **kw):
        """ Extract eventualities from a raw text

        :param text: a raw text
        :type text: str
        :param output_format: which format to return, "Eventuality" or "json"
        :type output_format: str (default = "Eventuality")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param use_lemma: whether the returned eventuality uses lemma
        :type use_lemma: bool (default = True)
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
            raise NotImplementedError("Error: extract_eventualities_from_text only supports Eventuality or json.")

        parsed_result = self.parse_text(text, annotators=annotators)
        return self.extract_eventualities_from_parsed_result(
            parsed_result, output_format=output_format, in_order=in_order, use_lemma=use_lemma, **kw
        )

    def extract_relations_from_parsed_result(
        self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
    ):
        """ Extract relations from a parsed result (of a paragraph) and extracted eventualities

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

        if output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations_from_parsed_result only supports Relation or triplet.")

        return self.relation_extractor.extract_from_parsed_result(
            parsed_result, para_eventualities, output_format=output_format, in_order=in_order, **kw
        )

    def extract_relations_from_text(self, text, output_format="Relation", in_order=True, annotators=None, **kw):
        """ Extract relations from a raw text and extracted eventualities

        :param text: a raw text
        :type text: str
        :param output_format: which format to return, "Relation" or "triplet"
        :type output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted relations
        :rtype: Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            [[],
             [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
             [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]]
        """

        if output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations_from_text only supports Relation or triplet.")

        parsed_result = self.parse_text(text, annotators=annotators)
        para_eventualities = self.extract_eventualities_from_parsed_result(parsed_result)
        return self.extract_relations_from_parsed_result(
            parsed_result, para_eventualities, output_format=output_format, in_order=in_order, **kw
        )

    def extract_from_parsed_result(
        self,
        parsed_result,
        eventuality_output_format="Eventuality",
        relation_output_format="Relation",
        in_order=True,
        use_lemma=True,
        **kw
    ):
        """ Extract both eventualities and relations from a parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param eventuality_output_format: which format to return eventualities, "Eventuality" or "json"
        :type eventuality_output_format: str (default = "Eventuality")
        :param relation_output_format: which format to return relations, "Relation" or "triplet"
        :type relation_output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param use_lemma: whether the returned eventuality uses lemma
        :type use_lemma: bool (default = True)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities and relations
        :rtype: Tuple[Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]], Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]]

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

            ([[my army will find you boat],
              [i be sure, we could find you suitable accommodation]],
             [[],
              [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
              [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]])
        """

        if eventuality_output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
        if relation_output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations only supports Relation or triplet.")

        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False

        para_eventualities = self.extract_eventualities_from_parsed_result(
            parsed_result, output_format="Eventuality", in_order=True, use_lemma=use_lemma, **kw
        )
        para_relations = self.extract_relations_from_parsed_result(
            parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw
        )

        if in_order:
            if eventuality_output_format == "json":
                para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
                                      for sent_eventualities in para_eventualities]
            if relation_output_format == "triplet":
                para_relations = [list(chain.from_iterable([relation.to_triplet() for relation in sent_relations])) \
                             for sent_relations in para_relations]
            if is_single_sent:
                return para_eventualities[0], para_relations[0]
            else:
                return para_eventualities, para_relations
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if eventuality_output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif eventuality_output_format == "json":
                eventualities = sorted(
                    [eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()],
                    key=lambda e: e["eid"]
                )

            rid2relation = dict()
            for relation in chain.from_iterable(para_relations):
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update(relation)
            if relation_output_format == "Relation":
                para_relations = sorted(rid2relation.values(), key=lambda r: r.rid)
            elif relation_output_format == "triplet":
                para_relations = sorted(chain.from_iterable([relation.to_triplets() for relation in rid2relation.values()]))
            return eventualities, para_relations

    def extract_from_text(
        self,
        text,
        eventuality_output_format="Eventuality",
        relation_output_format="Relation",
        in_order=True,
        use_lemma=True,
        annotators=None,
        **kw
    ):
        """ Extract both eventualities and relations from a raw text

        :param text: a raw text
        :type text: str
        :param eventuality_output_format: which format to return eventualities, "Eventuality" or "json"
        :type eventuality_output_format: str (default = "Eventuality")
        :param relation_output_format: which format to return relations, "Relation" or "triplet"
        :type relation_output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param use_lemma: whether the returned eventuality uses lemma
        :type use_lemma: bool (default = True)
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param kw: other parameters
        :type kw: Dict[str, object]
        :return: the extracted eventualities and relations
        :rtype: :rtype: Tuple[Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]], Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "My army will find your boat. In the meantime, I'm sure we could find you suitable accommodations."

            Output:

            ([[my army will find you boat],
              [i be sure, we could find you suitable accommodation]],
             [[],
              [(7d9ea9023b66a0ebc167f0dbb6ea8cd75d7b46f9, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Co_Occurrence': 1.0})],
              [(8540897b645962964fd644242d4cc0032f024e86, 25edad6781577dcb3ba715c8230416fb0d4c45c4, {'Synchronous': 1.0})]])
        """
        if eventuality_output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
        if relation_output_format not in ["Relation", "triplet"]:
            raise NotImplementedError("Error: extract_relations only supports Relation or triplet.")

        parsed_result = self.parse_text(text, annotators=annotators)
        return self.extract_from_parsed_result(
            parsed_result,
            eventuality_output_format=eventuality_output_format,
            relation_output_format=relation_output_format,
            in_order=in_order,
            use_lemma=use_lemma,
            **kw
        )


class SeedRuleASERExtractor(BaseASERExtractor):
    """ ASER Extractor based on rules to extract both eventualities and relations (for ASER v1.0)

    """
    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        if "annotators" not in kw:
            kw["annotators"] = list(ANNOTATORS)
            if "parse" in kw["annotators"]:
                kw["annotators"].remove("parse")
            if "depparse" not in kw["annotators"]:
                kw["annotators"].append("depparse")
        super().__init__(corenlp_path, corenlp_port, **kw)
        from .rule import CLAUSE_WORDS
        self.eventuality_extractor = SeedRuleEventualityExtractor(
            corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, skip_words=CLAUSE_WORDS, **kw
        )
        self.relation_extractor = SeedRuleRelationExtractor(**kw)


class DiscourseASERExtractor(BaseASERExtractor):
    """ ASER Extractor based on discourse parsing to extract both eventualities and relations (for ASER v2.0)

    """
    def __init__(self, corenlp_path="", corenlp_port=0, **kw):
        if "annotators" not in kw:
            kw["annotators"] = list(ANNOTATORS)
            if "depparse" in kw["annotators"]:
                kw["annotators"].remove("depparse")
            if "parse" not in kw["annotators"]:
                kw["annotators"].append("parse")
        super().__init__(corenlp_path, corenlp_port, **kw)
        self.eventuality_extractor = DiscourseEventualityExtractor(
            corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, **kw
        )
        self.relation_extractor = DiscourseRelationExtractor(**kw)

    def extract_from_parsed_result(
        self,
        parsed_result,
        eventuality_output_format="Eventuality",
        relation_output_format="Relation",
        in_order=True,
        use_lemma=True,
        **kw
    ):
        """ Extract both eventualities and relations from a parsed result

        :param parsed_result: the parsed result returned by corenlp
        :type parsed_result: List[Dict[str, object]]
        :param eventuality_output_format: which format to return eventualities, "Eventuality" or "json"
        :type eventuality_output_format: str (default = "Eventuality")
        :param relation_output_format: which format to return relations, "Relation" or "triplet"
        :type relation_output_format: str (default = "Relation")
        :param in_order: whether the returned order follows the input token order
        :type in_order: bool (default = True)
        :param use_lemma: whether the returned eventuality uses lemma
        :type use_lemma: bool (default = True)
        :param kw: other parameters (e.g., syntax_tree_cache)
        :type kw: Dict[str, object]
        :return: the extracted eventualities and relations
        :rtype: :rtype: Tuple[Union[List[List[aser.eventuality.Eventuality]], List[List[Dict[str, object]]], List[aser.eventuality.Eventuality], List[Dict[str, object]]], Union[List[List[aser.relation.Relation]], List[List[Dict[str, object]]], List[aser.relation.Relation], List[Dict[str, object]]]]
        """

        if "syntax_tree_cache" not in kw:
            kw["syntax_tree_cache"] = dict()
        return super().extract_from_parsed_result(
            parsed_result,
            eventuality_output_format=eventuality_output_format,
            relation_output_format=relation_output_format,
            in_order=in_order,
            use_lemma=use_lemma,
            **kw
        )

# The following extractor can cover more eventualities but the semantic meaning may be incomplete.
# class DiscourseASERExtractor(BaseASERExtractor):
#     def __init__(self, corenlp_path="", corenlp_port=0, **kw):
#         super().__init__(corenlp_path, corenlp_port, **kw)
#         self.eventuality_extractor = SeedRuleEventualityExtractor(**kw)
#         self.conn_extractor = ConnectiveExtractor(**kw)
#         self.argpos_classifier = ArgumentPositionClassifier(**kw)
#         self.ss_extractor = SSArgumentExtractor(**kw)
#         self.ps_extractor = PSArgumentExtractor(**kw)
#         self.explicit_classifier = ExplicitSenseClassifier(**kw)

#     def _extract_eventualities_from_clause(self, sent_parsed_result, clause, use_lemma):
#         len_clause = len(clause)
#         idx_mapping = {j: i for i, j in enumerate(clause)}
#         indices_set = set(clause)
#         clause_parsed_result = {
#             "text": "",
#             "dependencies": [(idx_mapping[dep[0]], dep[1], idx_mapping[dep[2]]) for dep in sent_parsed_result["dependencies"] \
#                 if dep[0] in indices_set and dep[2] in indices_set],
#             "tokens": [sent_parsed_result["tokens"][idx] for idx in clause],
#             "pos_tags": [sent_parsed_result["pos_tags"][idx] for idx in clause],
#             "lemmas": [sent_parsed_result["lemmas"][idx] for idx in clause]}
#         if "ners" in sent_parsed_result:
#             clause_parsed_result["ners"] = [sent_parsed_result["ners"][idx] for idx in clause]
#         if "mentions" in sent_parsed_result:
#             clause_parsed_result["mentions"] = list()
#             for mention in sent_parsed_result["mentions"]:
#                 start_idx = bisect.bisect_left(clause, mention["start"])
#                 if not (start_idx < len_clause and clause[start_idx] == mention["start"]):
#                     continue
#                 end_idx = bisect.bisect_left(clause, mention["end"]-1)
#                 if not (end_idx < len_clause and clause[end_idx] == mention["end"]-1):
#                     continue
#                 mention = copy(mention)
#                 mention["start"] = start_idx
#                 mention["end"] = end_idx+1
#                 clause_parsed_result["mentions"].append(mention)
#         eventualities = self.eventuality_extractor.extract_from_parsed_result(
#             clause_parsed_result, output_format="Eventuality", in_order=True, use_lemma=use_lemma)
#         for eventuality in eventualities:
#             for k, v in eventuality.raw_sent_mapping.items():
#                 eventuality.raw_sent_mapping[k] = clause[v]
#             eventuality.eid = Eventuality.generate_eid(eventuality)
#         return eventualities

#     def _append_new_eventuaities_to_list(self, existed_eventualities, new_eventualities):
#         len_existed_eventualities = len(existed_eventualities)
#         for new_e in new_eventualities:
#             is_existed = False
#             for old_idx in range(len_existed_eventualities):
#                 old_e = existed_eventualities[old_idx]
#                 if old_e.eid == new_e.eid and old_e.raw_sent_mapping == new_e.raw_sent_mapping:
#                     is_existed = True
#                     break
#             if not is_existed:
#                 existed_eventualities.append(new_e)

#     def extract_eventualities_from_parsed_result(self, parsed_result,
#                                                  output_format="Eventuality", in_order=True, use_lemma=True, **kw):
#         if output_format not in ["Eventuality", "json"]:
#             raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")

#         if not isinstance(parsed_result, (list, tuple, dict)):
#             raise NotImplementedError
#         if isinstance(parsed_result, dict):
#             is_single_sent = True
#             parsed_result = [parsed_result]
#         else:
#             is_single_sent = False

#         syntax_tree_cache = kw.get("syntax_tree_cache", dict())

#         para_eventualities = [list() for _ in range(len(parsed_result))]
#         para_clauses = self._extract_clauses(parsed_result, syntax_tree_cache)
#         for sent_parsed_result, sent_clauses, sent_eventualities in zip(parsed_result, para_clauses, para_eventualities):
#             for clause in sent_clauses:
#                 sent_eventualities.extend(self._extract_eventualities_from_clause(sent_parsed_result, clause, use_lemma))

#         if in_order:
#             if output_format == "json":
#                 para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
#                     for sent_eventualities in para_eventualities]
#             if is_single_sent:
#                 return para_eventualities[0]
#             else:
#                 return para_eventualities
#         else:
#             eid2eventuality = dict()
#             for eventuality in chain.from_iterable(para_eventualities):
#                 eid = eventuality.eid
#                 if eid not in eid2eventuality:
#                     eid2eventuality[eid] = deepcopy(eventuality)
#                 else:
#                     eid2eventuality[eid].update(eventuality)
#             if output_format == "Eventuality":
#                 eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
#             elif output_format == "json":
#                 eventualities = sorted([eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()], key=lambda e: e["eid"])
#             return eventualities

#     def extract_relations_from_parsed_result(self, parsed_result, para_eventualities,
#                                              output_format="Relation",
#                                              in_order=True, **kw):
#         if output_format not in ["Relation", "triplet"]:
#             raise NotImplementedError("Error: extract_relations_from_parsed_result only supports Relation or triplet.")

#         len_sentences = len(parsed_result)
#         if len_sentences == 0:
#             if in_order:
#                 return [list()]
#             else:
#                 return list()

#         similarity = kw.get("similarity", "simpson").lower()
#         threshold = kw.get("threshold", 0.8)
#         if threshold < 0.0 or threshold > 1.0:
#             raise ValueError("Error: threshold should be between 0.0 and 1.0.")
#         if similarity == "simpson":
#             similarity_func = self._match_argument_eventuality_by_Simpson
#         elif similarity == "jaccard":
#             similarity_func = self._match_argument_eventuality_by_Jaccard
#         elif similarity == "discourse":
#             similarity_func = self._match_argument_eventuality_by_dependencies
#         else:
#             raise NotImplementedError("Error: extract_from_parsed_result only supports Simpson or Jaccard.")

#         syntax_tree_cache = kw.get("syntax_tree_cache", dict())

#         para_relations = [list() for _ in range(2*len_sentences-1)]

#         # replace sentences that contains no eventuality with empty sentences
#         filtered_parsed_result = list()
#         for sent_idx, (sent_parsed_result, sent_eventualities) in enumerate(zip(parsed_result, para_eventualities)):
#             if len(sent_eventualities) > 0:
#                 relations_in_sent = para_relations[sent_idx]
#                 for e1_idx in range(len(sent_eventualities)-1):
#                     heid = sent_eventualities[e1_idx].eid
#                     for e2_idx in range(e1_idx+1, len(sent_eventualities)):
#                         teid = sent_eventualities[e2_idx].eid
#                         relations_in_sent.append(Relation(heid, teid, ["Co_Occurrence"]))
#                 filtered_parsed_result.append(sent_parsed_result)
#             else:
#                 filtered_parsed_result.append(EMPTY_SENT_PARSED_RESULT) # empty sentence
#                 # filtered_parsed_result.append(sent_parsed_result)

#         connectives = self.conn_extractor.extract(filtered_parsed_result, syntax_tree_cache)
#         SS_connectives, PS_connectives = self.argpos_classifier.classify(filtered_parsed_result, connectives, syntax_tree_cache)
#         SS_connectives = self.ss_extractor.extract(filtered_parsed_result, SS_connectives, syntax_tree_cache)
#         PS_connectives = self.ps_extractor.extract(filtered_parsed_result, PS_connectives, syntax_tree_cache)
#         connectives = self.explicit_classifier.classify(filtered_parsed_result, SS_connectives+PS_connectives, syntax_tree_cache)
#         connectives.sort(key=lambda x: (x["sent_idx"], x["indices"][0] if len(x["indices"]) > 0 else -1))

#         for connective in connectives:
#             conn_indices = connective.get("indices", None)
#             arg1 = connective.get("arg1", None)
#             arg2 = connective.get("arg2", None)
#             sense = connective.get("sense", None)
#             if conn_indices and arg1 and arg2 and (sense and sense != "None"):
#                 arg1_sent_idx = arg1["sent_idx"]
#                 arg2_sent_idx = arg2["sent_idx"]
#                 relation_list_idx = arg1_sent_idx if arg1_sent_idx == arg2_sent_idx else arg1_sent_idx + len_sentences
#                 relations = para_relations[relation_list_idx]
#                 sent_parsed_result1, sent_eventualities1 = parsed_result[arg1_sent_idx], para_eventualities[arg1_sent_idx]
#                 sent_parsed_result2, sent_eventualities2 = parsed_result[arg2_sent_idx], para_eventualities[arg2_sent_idx]
#                 arg1_eventualities = [e for e in sent_eventualities1 if \
#                     similarity_func(sent_parsed_result1, arg1, e, threshold=threshold, conn_indices=conn_indices)]
#                 arg2_eventualities = [e for e in sent_eventualities2 if \
#                     similarity_func(sent_parsed_result2, arg2, e, threshold=threshold, conn_indices=conn_indices)]
#                 cnt = 0.0
#                 if len(arg1_eventualities) > 0 and len(arg2_eventualities) > 0:
#                     cnt = 1.0 / (len(arg1_eventualities) * len(arg2_eventualities))
#                 for e1 in arg1_eventualities:
#                     heid = e1.eid
#                     for e2 in arg2_eventualities:
#                         teid = e2.eid
#                         existed_relation = False
#                         for relation in relations:
#                             if relation.hid == heid and relation.tid == teid:
#                                 relation.update({sense: cnt})
#                                 existed_relation = True
#                                 break
#                         if not existed_relation:
#                             relations.append(Relation(heid, teid, {sense: cnt}))

#         if in_order:
#             if output_format == "Relation":
#                 return para_relations
#             elif output_format == "triplet":
#                 return [sorted(chain.from_iterable([r.to_triplets() for r in relations])) \
#                     for relations in para_relations]
#         else:
#             if output_format == "Relation":
#                 rid2relation = dict()
#                 for relation in chain(*para_relations):
#                     if relation.rid not in rid2relation:
#                         rid2relation[relation.rid] = deepcopy(relation)
#                     else:
#                         rid2relation[relation.rid].update(relation)
#                 return sorted(rid2relation.values(), key=lambda r: r.rid)
#             if output_format == "triplet":
#                 return sorted([r.to_triplets() for relations in para_relations for r in relations])

#     def extract_from_parsed_result(self, parsed_result,
#                                    eventuality_output_format="Eventuality",
#                                    relation_output_format="Relation",
#                                    in_order=True, **kw):
#         if eventuality_output_format not in ["Eventuality", "json"]:
#             raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
#         if relation_output_format not in ["Relation", "triplet"]:
#             raise NotImplementedError("Error: extract_relations only supports Relation or triplet.")

#         if not isinstance(parsed_result, (list, tuple, dict)):
#             raise NotImplementedError
#         if isinstance(parsed_result, dict):
#             is_single_sent = True
#             parsed_result = [parsed_result]
#         else:
#             is_single_sent = False

#         syntax_tree_cache = kw.get("syntax_tree_cache", dict())

#         len_sentences = len(parsed_result)
#         para_eventualities = [list() for _ in range(len_sentences)]
#         para_relations = [list() for _ in range(2*len_sentences-1)]

#         connectives = self.conn_extractor.extract(parsed_result, syntax_tree_cache)
#         SS_connectives, PS_connectives = self.argpos_classifier.classify(parsed_result, connectives, syntax_tree_cache)
#         SS_connectives = self.ss_extractor.extract(parsed_result, SS_connectives, syntax_tree_cache)
#         PS_connectives = self.ps_extractor.extract(parsed_result, PS_connectives, syntax_tree_cache)
#         connectives = self.explicit_classifier.classify(parsed_result, SS_connectives+PS_connectives, syntax_tree_cache)
#         connectives.sort(key=lambda x: (x["sent_idx"], x["indices"][0] if len(x["indices"]) > 0 else -1))

#         for connective in connectives:
#             conn_indices = connective.get("indices", None)
#             arg1 = connective.get("arg1", None)
#             arg2 = connective.get("arg2", None)
#             sense = connective.get("sense", None)
#             if conn_indices and arg1 and arg2:
#                 arg1_sent_idx = arg1["sent_idx"]
#                 arg2_sent_idx = arg2["sent_idx"]
#                 senses = []
#                 if arg1_sent_idx == arg2_sent_idx:
#                     senses.append("Co_Occurrence")
#                 if sense and sense != "None":
#                     senses.append(sense)
#                 if len(senses) == 0:
#                     continue
#                 relation_list_idx = arg1_sent_idx if arg1_sent_idx == arg2_sent_idx else arg1_sent_idx + len_sentences
#                 relations = para_relations[relation_list_idx]
#                 sent_parsed_result1, sent_eventualities1 = parsed_result[arg1_sent_idx], para_eventualities[arg1_sent_idx]
#                 sent_parsed_result2, sent_eventualities2 = parsed_result[arg2_sent_idx], para_eventualities[arg2_sent_idx]
#                 arg1_eventualities = self._extract_eventualities_from_clause(sent_parsed_result1, arg1["indices"])
#                 arg2_eventualities = self._extract_eventualities_from_clause(sent_parsed_result2, arg2["indices"])
#                 self._append_new_eventuaities_to_list(sent_eventualities1, arg1_eventualities)
#                 self._append_new_eventuaities_to_list(sent_eventualities2, arg2_eventualities)

#                 cnt = 0.0
#                 if len(arg1_eventualities) > 0 and len(arg2_eventualities) > 0:
#                     cnt = 1.0 / (len(arg1_eventualities) * len(arg2_eventualities))
#                 for e1 in arg1_eventualities:
#                     heid = e1.eid
#                     for e2 in arg2_eventualities:
#                         teid = e2.eid
#                         is_existed = False
#                         for relation in relations:
#                             if relation.hid == heid and relation.tid == teid:
#                                 relation.update({sense: cnt for sense in senses})
#                                 is_existed = True
#                                 break
#                         if not is_existed:
#                             relations.append(Relation(heid, teid, {sense: cnt for sense in senses}))

#         if in_order:
#             if eventuality_output_format == "json":
#                 para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
#                     for sent_eventualities in para_eventualities]
#             if relation_output_format == "triplet":
#                 relations = [list(chain.from_iterable([relation.to_triplet() for relation in sent_relations])) \
#                     for sent_relations in para_relations]
#             if is_single_sent:
#                 return para_eventualities[0], para_relations[0]
#             else:
#                 return para_eventualities, para_relations
#         else:
#             eid2eventuality = dict()
#             for eventuality in chain.from_iterable(para_eventualities):
#                 eid = eventuality.eid
#                 if eid not in eid2eventuality:
#                     eid2eventuality[eid] = deepcopy(eventuality)
#                 else:
#                     eid2eventuality[eid].update(eventuality)
#             if eventuality_output_format == "Eventuality":
#                 eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
#             elif eventuality_output_format == "json":
#                 eventualities = sorted([eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()], key=lambda e: e["eid"])

#             rid2relation = dict()
#             for relation in chain.from_iterable(para_relations):
#                 if relation.rid not in rid2relation:
#                     rid2relation[relation.rid] = deepcopy(relation)
#                 else:
#                     rid2relation[relation.rid].update(relation)
#             if relation_output_format == "Relation":
#                 relations = sorted(rid2relation.values(), key=lambda r: r.rid)
#             elif relation_output_format == "triplet":
#                 relations = sorted(chain.from_iterable([relation.to_triplets() for relation in rid2relation.values()]))
#             return eventualities, relations

#     def _extract_clauses(self, parsed_result, syntax_tree_cache):
#         para_arguments = [set() for _ in range(len(parsed_result))]
#         connectives = self.conn_extractor.extract(parsed_result, syntax_tree_cache)
#         para_connectives = [set() for _ in range(len(parsed_result))]
#         for connective in connectives:
#             sent_idx, indices = connective["sent_idx"], tuple(connective["indices"])
#             para_connectives[sent_idx].add(indices)
#         for sent_idx, sent_parsed_result in enumerate(parsed_result):
#             sent_connectives = para_connectives[sent_idx]
#             sent_arguments = para_arguments[sent_idx]

#             if sent_idx in syntax_tree_cache:
#                 syntax_tree = syntax_tree_cache[sent_idx]
#             else:
#                 syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])

#             # more but slower
#             # for indices in powerset(sent_connectives):
#             #     indices = set(chain.from_iterable(indices))
#             #     sent_arguments.update(get_clauses(sent_parsed_result, syntax_tree, sep_indices=indices))
#             sent_arguments.update(get_clauses(sent_parsed_result, syntax_tree, sep_indices=set(chain.from_iterable(sent_connectives))))
#         return para_arguments
