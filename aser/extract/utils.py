import os
import re
import socket
from itertools import chain, combinations
from stanfordnlp.server import CoreNLPClient

ANNOTATORS = ("tokenize", "ssplit", "pos", "lemma", "parse", "ner")
TYPE_SET = frozenset(["CITY", "ORGANIZATION", "COUNTRY",
            "STATE_OR_PROVINCE", "LOCATION", "NATIONALITY", "PERSON"])
PRONOUN_SET = frozenset(["i", "I", "me", "my", "mine", "myself", "we", "us", "our", "ours", "ourselves", "you", "your", "yours",
                "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it",
                "its", "itself", "they", "them", "their", "theirs", "themself", "themselves"])
PUNCTUATION_SET = frozenset(list("""!"#&'*+,-..../:;<=>?@[\]^_`|~""") + ["``", "''"])
CLAUSE_SEPARATOR_SET = frozenset(list(".,:;?!~-") + ["..", "...", "--", "---"])

EMPTY_SENT_PARSED_RESULT = {
    "text": ".",
    "dependencies": [],
    "tokens": ["."],
    "lemmas": ["."],
    "pos_tags": ["."],
    "parse": "(ROOT (NP (. .)))",
    "ners": ["O"],
    "mentions": []}

MAX_ATTEMPT=10

def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False


def get_corenlp_client(corenlp_path="", corenlp_port=0, annotators=None):
    if not (corenlp_path and corenlp_port):
        return None, True

    if not annotators:
        annotators = list(ANNOTATORS)

    if is_port_occupied(port=corenlp_port):
        try:
            os.environ["CORENLP_HOME"] = corenlp_path
            corenlp_client = CoreNLPClient(
                annotators=annotators, timeout=99999,
                memory='4G', endpoint="http://localhost:%d" % corenlp_port,
                start_server=False, be_quiet=False)
            corenlp_client.annotate("hello world",
                                    annotators=list(annotators),
                                    output_format="json")
            return corenlp_client, True
        except Exception as err:
            raise err
    else:
        print("Starting corenlp client at port {}".format(corenlp_port))
        corenlp_client = CoreNLPClient(
            annotators=annotators, timeout=99999,
            memory='4G', endpoint="http://localhost:%d" % corenlp_port,
            start_server=True, be_quiet=False)
        corenlp_client.annotate("hello world",
                                annotators=list(annotators),
                                output_format="json")
        return corenlp_client, False


def clean_sentence_for_parsing(text):
    return re.sub(r'[^\x00-\x7F]+', ' ', text)

def parse_sentense_with_stanford(input_sentence, corenlp_client: CoreNLPClient,
                                 annotators=None, max_len=1000):
    if not annotators:
        annotators = list(ANNOTATORS)

    cleaned_para = clean_sentence_for_parsing(input_sentence)

    need_to_split = len(cleaned_para) > max_len
    if not need_to_split:
        try:
            parsed_sentences = corenlp_client.annotate(cleaned_para, annotators=annotators, output_format='json')['sentences']
            raw_texts = list()
            for sent in parsed_sentences:
                if sent['tokens']:
                    char_st = sent['tokens'][0]['characterOffsetBegin']
                    char_end = sent['tokens'][-1]['characterOffsetEnd']
                else:
                    char_st, char_end = 0, 0
                raw_text = cleaned_para[char_st:char_end]
                raw_texts.append(raw_text)
        except Exception:
            need_to_split = True

    if need_to_split:
        parsed_sentences = list()
        raw_texts = list()
        temp = corenlp_client.annotate(cleaned_para, annotators=["ssplit"], output_format='json')['sentences']
        for sent in temp:            
            if sent['tokens']:
                char_st = sent['tokens'][0]['characterOffsetBegin']
                char_end = sent['tokens'][-1]['characterOffsetEnd']
            else:
                char_st, char_end = 0, 0
            if char_st == char_end:
                continue
            text = cleaned_para[char_st:char_end]
            parsed_sentences.extend(corenlp_client.annotate(text, annotators=annotators, output_format='json')['sentences'])
            raw_texts.append(text)

    parsed_rst_list = list()
    for sent, text in zip(parsed_sentences, raw_texts):
        enhanced_dependency_list = sent['enhancedPlusPlusDependencies']
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

        x = {
            "text": text,
            "dependencies": dependencies,
            "tokens": [t['word'] for t in sent['tokens']],
        }
        if 'pos' in annotators:
            x['pos_tags'] = [t['pos'] for t in sent['tokens']]
        if 'lemma' in annotators:
            x["lemmas"] = [t['lemma'] for t in sent['tokens']]
        if 'ner' in annotators:
            mentions = []
            for m in sent['entitymentions']:
                if m['ner'] in TYPE_SET and m['text'].lower().strip() not in PRONOUN_SET:
                    mentions.append({'start': m['tokenBegin'], 'end': m['tokenEnd'], 'text': m['text'], 'ner': m['ner'],
                                     'link': None, 'entity': None})

            x['ners'] = [t['ner'] for t in sent['tokens']]
            x['mentions'] = mentions
        if 'parse' in annotators:
            x['parse'] = re.sub(r"\s+", " ", sent['parse'])

        parsed_rst_list.append(x)
    return parsed_rst_list


def sort_dependencies_position(dependencies, reset_position=True):
    """ Fix absolute position into relevant position and sort.
        Input example:
        [[8, 'cop', 7], [8, 'nsubj', 6]]
        Output example if fix_position:
        [[2, 'nsubj', 0], [2, 'cop', 1]], {0: 6, 1: 7, 2: 8}
        Output example if not fix_position:
        [[8, 'nsubj', 6], [8, 'cop', 7]]
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

def index_from(sequence, x, start_from=0):
    indices = []
    for idx in range(start_from, len(sequence)):
        if x == sequence[idx]:
            indices.append(idx)
    return indices

def powerset(iterable, min_size=0, max_size=-1):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = sorted(iterable)
    if max_size == -1:
        max_size = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(min_size, max_size+1))

def get_clauses(sent_parsed_result, syntax_tree, index_seps=None):
    def find_clauses(clause):
        # split by the SBAR tag
        clause_tree = syntax_tree.get_subtree_by_token_indices(clause)
        find_SBAR = False
        if clause_tree.tree:
            for node in clause_tree.tree.traverse():
                if node.name == "SBAR":
                    leaves = set([node.index for node in node.get_leaves()])
                    if len(leaves) == len(clause):
                        continue
                    clause1, clause2 = list(), list()
                    for idx in clause:
                        if idx in leaves:
                            clause1.append(idx)
                        else:
                            clause2.append(idx)
                    if clause1[0] < clause2[0]:
                        return tuple(clause1), tuple(clause2)
                        # return find_clauses(clause1) + find_clauses(clause2)
                    else:
                        return tuple(clause2), tuple(clause1)
                        # return find_clauses(clause2) + find_clauses(clause1)
        return [tuple(clause)]

    if index_seps is None:
        index_seps = set()
    elif isinstance(index_seps, (list, tuple)):
        index_seps = set(index_seps)
    sent_len = len(sent_parsed_result["tokens"])
    
    clauses = list() # (parent, indices)
    clause = list()
    for t_idx, token in enumerate(sent_parsed_result["tokens"]):
        # split the sentence by seps
        valid = token not in CLAUSE_SEPARATOR_SET and t_idx not in index_seps
        if valid:
            clause.append(t_idx)
        if t_idx == sent_len-1 or not valid:
            # strip_punctuation
            clause = strip_punctuation(sent_parsed_result, clause)
            if len(clause) > 0:
                clauses.extend(find_clauses(clause))
            clause = list()
    return clauses

def get_prev_token_index(doc_parsed_result, sent_idx, idx, skip_tokens=None):
    if skip_tokens is None:
        skip_tokens = set()
    curr_sent_idx, curr_idx = sent_idx, idx

    for i in range(MAX_ATTEMPT):
        if curr_idx-1 >= 0:
            curr_idx = curr_idx - 1
        elif curr_sent_idx-1 >= 0:
            curr_sent_idx = curr_sent_idx - 1
            curr_idx = len(doc_parsed_result[curr_sent_idx]["tokens"]) - 1
        else:
            return -1, -1
        curr_token = doc_parsed_result[curr_sent_idx]["tokens"][curr_idx]
        if curr_token not in skip_tokens:
            return curr_sent_idx, curr_idx
    return -1, -1

def get_next_token_index(doc_parsed_result, sent_idx, idx, skip_tokens=None):
    if skip_tokens is None:
        skip_tokens = set()
    curr_sent_idx, curr_idx = sent_idx, idx

    for i in range(MAX_ATTEMPT):
        if curr_idx+1 < len(doc_parsed_result[curr_sent_idx]["tokens"]):
            curr_idx = curr_idx + 1
        elif curr_sent_idx+1 < len(doc_parsed_result):
            curr_sent_idx = curr_sent_idx + 1
            curr_idx = 0
        else:
            return -1, -1
        curr_token = doc_parsed_result[curr_sent_idx]["tokens"][curr_idx]
        if curr_token not in skip_tokens:
            return curr_sent_idx, curr_idx
    return -1, -1

def strip_punctuation(sent_parsed_result, indices):
    valid_idx1, valid_idx2 = 0, len(indices)
    while valid_idx1 < valid_idx2:
        if indices[valid_idx1] >= len(sent_parsed_result["tokens"]):
            break
        token = sent_parsed_result["tokens"][indices[valid_idx1]]
        if token in PUNCTUATION_SET or token == "-LCB-" or token == "-LRB-":
            valid_idx1 += 1
        else:
            break
    while valid_idx1 < valid_idx2:
        if indices[valid_idx2-1] >= len(sent_parsed_result["tokens"]):
            valid_idx2 -= 1
            continue
        token = sent_parsed_result["tokens"][indices[valid_idx2-1]]
        if token in PUNCTUATION_SET or token == "-LCB-" or token == "-LRB-":
            valid_idx2 -= 1
        else:
            break
    if valid_idx1 == 0 and valid_idx2 == len(indices):
        return indices
    else:
        return indices[valid_idx1:valid_idx2]
