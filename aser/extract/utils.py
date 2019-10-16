import os
import re
import socket
from stanfordnlp.server import CoreNLPClient


_VALID_CHARS = """qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890`~!@#$%^&*/?., ;:"'"""
_ANNOTATORS = ('tokenize', 'ssplit', 'pos', 'lemma', 'depparse')

def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False

def get_corenlp_client(corenlp_path, port, annotators=_ANNOTATORS):
    os.environ["CORENLP_HOME"] = corenlp_path
    print("Starting corenlp client at port {}".format(port))
    if is_port_occupied(port=port):
        try:
            corenlp_client = CoreNLPClient(
                annotators=annotators, timeout=60000,
                memory='5G', endpoint="http://localhost:%d" % port,
                start_server=False, be_quiet=False)
            corenlp_client.annotate("hello world",
                                    annotators=list(annotators),
                                    output_format="json")
            return corenlp_client, True
        except Exception as err:
            raise err
    else:
        corenlp_client = CoreNLPClient(
            annotators=annotators, timeout=60000,
            memory='5G', endpoint="http://localhost:%d" % port,
            start_server=True, be_quiet=False)
        corenlp_client.annotate("hello world",
                                annotators=list(annotators),
                                output_format="json")
        return corenlp_client, False


def parse_sentense_with_stanford(input_sentence, corenlp_client, annotators=_ANNOTATORS):
    def clean_sentence_for_parsing(input_sentence):
        new_sentence = ''
        for char in input_sentence:
            if char in _VALID_CHARS:
                new_sentence += char
            else:
                new_sentence += '\n'
        return new_sentence
    cleaned_sentence = clean_sentence_for_parsing(input_sentence)
    tmp_output = corenlp_client.annotate(cleaned_sentence,
                                         annotators=list(annotators),
                                         output_format="json")
    parsed_rst_list = list()
    for s in tmp_output['sentences']:
        enhanced_dependency_list = s['enhancedPlusPlusDependencies']
        dependencies = set()
        for relation in enhanced_dependency_list:
            if relation['dep'] == 'ROOT':
                continue
            governor_pos = relation['governor']
            dependent_pos = relation['dependent']
            dependencies.add(
                (governor_pos - 1,
                 relation['dep'],
                 dependent_pos - 1))
        dependencies = list(dependencies)
        dependencies.sort(key=lambda x: (x[0], x[2]))

        if s['tokens']:
            char_st = s['tokens'][0]['characterOffsetBegin']
            char_end = s['tokens'][-1]['characterOffsetEnd']
        else:
            char_st, char_end = 0, 0
        parsed_rst_list.append({
            "text": cleaned_sentence[char_st:char_end],
            "dependencies": dependencies,
            "tokens": [t['word'] for t in s['tokens']],
            "lemmas": [t['lemma'] for t in s['tokens']],
            "pos_tags": [t['pos'] for t in s['tokens']]
        })
    return parsed_rst_list


def sort_dependencies_position(dependencies, reset_position=True):
    """ Fix absolute position into relevant position and sort.

        Input example:
        [[8, 'cop', 7], [8, 'nsubj', 6]]

        Output example if fix_position:
        [[2, 'nsubj', 0], [2, 'cop', 1]], {0: 6, 1: 7, 2: 8}

        Output example if not fix_position:
        [, [8, 'nsubj', 6], [8, 'cop', 7]]

    """

    tmp_dependencies = set()
    for triple in dependencies:
        tmp_dependencies.add(tuple(triple))
    new_dependencies = list()
    if reset_position:
        positions = set()
        for governor, _, dependent in tmp_dependencies:
            positions.add(governor)
            positions.add(dependent)
        positions = list(sorted(positions))
        position_map = dict(zip(positions, range(len(positions))))

        for governor, dep, dependent in tmp_dependencies:
            new_dependencies.append(
                (position_map[governor], dep, position_map[dependent]))
        new_dependencies.sort(key=lambda x: (x[0], x[2]))
        return new_dependencies, position_map, {val: key for key, val in position_map.items()}
    else:
        new_dependencies = list([t for t in
                                 sorted(tmp_dependencies, key=lambda x: (x[0], x[2]))])
        return new_dependencies, None, None


def extract_indices_from_dependencies(dependencies):
    """ Extract all tokens from dependencies

        Input example:
        [[8, 'cop', 7], [8, 'nsubj', 6]]

        Output example:
        [6, 7, 8]
    """
    word_positions = set()
    for governor_pos, _, dependent_pos in dependencies:
        word_positions.add(governor_pos)
        word_positions.add(dependent_pos)

    return list(sorted(word_positions))

def iter_files(path):
    """Walk through all files located under a root path."""
    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)