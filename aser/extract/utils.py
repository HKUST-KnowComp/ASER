import os
import socket
from stanfordnlp.server import CoreNLPClient


_VALID_CHARS = frozenset("""qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890`~!@#$%^&*/?., ;:"'""")


def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False

def get_corenlp_client(corenlp_path, port):
    os.environ["CORENLP_HOME"] = corenlp_path

    assert not is_port_occupied(port), "Port {} is occupied by other process".format(port)
    corenlp_client = CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'], timeout=60000,
        memory='5G', endpoint="http://localhost:%d" % port,
        start_server=True, be_quiet=True)
    corenlp_client.annotate("hello world",
                            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'],
                            output_format="json")
    return corenlp_client


def parse_sentense_with_stanford(input_sentence, corenlp_client):
    def clean_sentence_for_parsing(input_sentence):
        new_sentence = ''
        for char in input_sentence:
            if char in _VALID_CHARS:
                new_sentence += char
            else:
                new_sentence += '\n'
        return new_sentence
    cleaned_sentence = clean_sentence_for_parsing(input_sentence)
    tmp_output = corenlp_client.annotate(cleaned_sentence, output_format="json")

    dependencies_list = list()
    for s in tmp_output['sentences']:
        enhanced_dependency_list = s['enhancedPlusPlusDependencies']
        dependencies = list()
        for relation in enhanced_dependency_list:
            if relation['dep'] == 'ROOT':
                continue
            governor_position = relation['governor']
            dependent_position = relation['dependent']
            dependencies.append(
                [
                    [governor_position,
                     s['tokens'][governor_position - 1]['lemma'],
                     s['tokens'][governor_position - 1]['pos']],

                    relation['dep'],

                    [dependent_position,
                     s['tokens'][dependent_position - 1]['lemma'],
                     s['tokens'][dependent_position - 1]['pos']]
                ]
            )
        tokens = list()
        for token in s['tokens']:
            tokens.append(token['word'])
        dependencies_list.append({
            "dependencies": dependencies,
            "tokens": tokens
        })
    return dependencies_list


def sort_dependencies_position(dependencies, fix_position=True):
    """ Fix absolute position into relevant position and sort.

        Input example:
        [[[8, 'hungry', 'JJ'], 'cop', [7, 'be', 'VBP']],
         [[8, 'hungry', 'JJ'], 'nsubj', [6, 'I', 'PRP']]]

        Output example if fix_position:
        [[[2, 'hungry', 'JJ'], 'nsubj', [0, 'I', 'PRP']],
         [[2, 'hungry', 'JJ'], 'cop', [1, 'be', 'VBP']]]

        Output example if not fix_position:
        [[[8, 'hungry', 'JJ'], 'nsubj', [7, 'I', 'PRP']],
         [[8, 'hungry', 'JJ'], 'cop', [6, 'be', 'VBP']]]

    """
    if fix_position:
        positions = set()
        for head, _, tail in dependencies:
            positions.add(head[0])
            positions.add(tail[0])
        positions = list(sorted(positions))
        position_map = dict(zip(positions, range(len(positions))))

        for i in range(len(dependencies)):
            head, _, tail = dependencies[i]
            head[0] = position_map[head[0]]
            tail[0] = position_map[tail[0]]
    dependencies.sort(key=lambda x: (x[0][0], x[2][0]))


def extract_tokens_from_dependencies(dependencies, only_words=False):
    """ Extract all tokens from dependencies

        Input example:
        [[[8, 'hungry', 'JJ'], 'cop', [7, 'be', 'VBP']],
         [[8, 'hungry', 'JJ'], 'nsubj', [6, 'I', 'PRP']]]

        Output example:
        [['I', 'PRP'], ['be', 'VBP'], ['hungry', 'JJ']]
    """
    pos_and_tokens = set()
    for governor, _, dependent in dependencies:
        pos_and_tokens.add(tuple(governor))
        pos_and_tokens.add(tuple(dependent))
    if only_words:
        tokens = [t[1] for t in
                  sorted(pos_and_tokens, key=lambda x: x[0])]
    else:
        tokens = [[t[1], t[2]] for t in
                  sorted(pos_and_tokens, key=lambda x: x[0])]
    return tokens