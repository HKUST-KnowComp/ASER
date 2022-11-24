import os
import re
import socket
from collections import defaultdict
from copy import copy, deepcopy
from itertools import chain, combinations
# from stanfordnlp.server import CoreNLPClient, TimeoutException
from stanza.server import CoreNLPClient, TimeoutException

ANNOTATORS = ("tokenize", "ssplit", "pos", "lemma", "parse", "ner")

TYPE_SET = frozenset(["CITY", "ORGANIZATION", "COUNTRY", "STATE_OR_PROVINCE", "LOCATION", "NATIONALITY", "PERSON"])

PRONOUN_SET = frozenset(
    [
        "i", "I", "me", "my", "mine", "myself",
        "we", "us", "our", "ours", "ourselves",
        "you", "your", "yours",
        "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "they", "them", "their", "theirs", "themself", "themselves"
    ]
)

PUNCTUATION_SET = frozenset(list("""!"#&'*+,-..../:;<=>?@[\]^_`|~""") + ["``", "''"])

CLAUSE_SEPARATOR_SET = frozenset(list(".,:;?!~-") + ["..", "...", "--", "---"])

URL_REGEX = re.compile(
    r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))',
    re.IGNORECASE
)

EMPTY_SENT_PARSED_RESULT = {
    "text": ".",
    "dependencies": [],
    "tokens": ["."],
    "lemmas": ["."],
    "pos_tags": ["."],
    "parse": "(ROOT (NP (. .)))",
    "ners": ["O"],
    "mentions": []
}

MAX_LEN = 1024
MAX_ATTEMPT = 10


def is_port_occupied(ip="127.0.0.1", port=80):
    """ Check whether the ip:port is occupied

    :param ip: the ip address
    :type ip: str (default = "127.0.0.1")
    :param port: the port
    :type port: int (default = 80)
    :return: whether is occupied
    :rtype: bool
    """

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False


def get_corenlp_client(corenlp_path="", corenlp_port=0, annotators=None):
    """

    :param corenlp_path: corenlp path, e.g., /home/xliucr/stanford-corenlp-3.9.2
    :type corenlp_path: str (default = "")
    :param corenlp_port: corenlp port, e.g., 9000
    :type corenlp_port: int (default = 0)
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :return: the corenlp client and whether the client is external
    :rtype: Tuple[stanfordnlp.server.CoreNLPClient, bool]
    """

    if corenlp_port == 0:
        return None, True

    if not annotators:
        annotators = list(ANNOTATORS)

    os.environ["CORENLP_HOME"] = corenlp_path

    if is_port_occupied(port=corenlp_port):
        try:
            corenlp_client = CoreNLPClient(
                annotators=annotators,
                timeout=99999,
                memory='4G',
                endpoint="http://localhost:%d" % corenlp_port,
                start_server=False,
                be_quiet=False
            )
            # corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
            return corenlp_client, True
        except BaseException as err:
            raise err
    elif corenlp_path != "":
        print("Starting corenlp client at port {}".format(corenlp_port))
        corenlp_client = CoreNLPClient(
            annotators=annotators,
            timeout=99999,
            memory='4G',
            endpoint="http://localhost:%d" % corenlp_port,
            start_server=True,
            be_quiet=False
        )
        corenlp_client.annotate("hello world", annotators=list(annotators), output_format="json")
        return corenlp_client, False
    else:
        return None, True


def split_sentence_for_parsing(text, corenlp_client, annotators=None, max_len=MAX_LEN):
    """ Split a long sentence (paragraph) into a list of shorter sentences

    :param text: a raw text
    :type text: str
    :param corenlp_client: the given corenlp client
    :type corenlp_client: stanfordnlp.server.CoreNLPClient
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
    :type max_len: int (default = 1024)
    :return: a list of sentences that satisfy the maximum length requirement
    :rtype: List[str]
    """

    if len(text) <= max_len:
        return [text]

    texts = text.split("\n\n")
    if len(texts) > 1:
        return list(
            chain.from_iterable(
                map(lambda sent: split_sentence_for_parsing(sent, corenlp_client, annotators, max_len), texts)
            )
        )

    texts = text.split("\n")
    if len(texts) > 1:
        return list(
            chain.from_iterable(
                map(lambda sent: split_sentence_for_parsing(sent, corenlp_client, annotators, max_len), texts)
            )
        )

    texts = list()
    temp = corenlp_client.annotate(text, annotators=["ssplit"], output_format='json')['sentences']
    for sent in temp:
        if sent['tokens']:
            char_st = sent['tokens'][0]['characterOffsetBegin']
            char_end = sent['tokens'][-1]['characterOffsetEnd']
        else:
            char_st, char_end = 0, 0
        if char_st == char_end:
            continue
        if char_end - char_st <= max_len:
            texts.append(text[char_st:char_end])
        else:
            texts.extend(re.split(PUNCTUATION_SET, text[char_st:char_end]))
    return texts


def clean_sentence_for_parsing(text):
    """ Clean the raw text

    :param text: a raw text
    :type text: str
    :return: the cleaned text
    :rtype: str
    """

    #  only consider the ascii
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # replace ref
    text = re.sub(r"<ref(.*?)>", "<ref>", text)

    # replace url
    text = re.sub(URL_REGEX, "<url>", text)
    text = re.sub(r"<url>[\(\)\[\]]*<url>", "<url>", text)

    return text.strip()


def parse_sentense_with_stanford(input_sentence, corenlp_client, annotators=None, max_len=MAX_LEN):
    """

    :param input_sentence: a raw sentence
    :type input_sentence: str
    :param corenlp_client: the given corenlp client
    :type corenlp_client: stanfordnlp.server.CoreNLPClient
    :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
    :type annotators: Union[List, None] (default = None)
    :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
    :type max_len: int (default = 1024)
    :return: the parsed result
    :rtype: List[Dict[str, object]]
    """

    if not annotators:
        annotators = list(ANNOTATORS)

    parsed_sentences = list()
    raw_texts = list()
    cleaned_sentence = clean_sentence_for_parsing(input_sentence)
    for sentence in split_sentence_for_parsing(cleaned_sentence, corenlp_client, annotators, max_len):
        while True:
            try:
                parsed_sentence = corenlp_client.annotate(sentence, annotators=annotators,
                                                          output_format="json")["sentences"]
                break
            except TimeoutException as e:
                continue
        for sent in parsed_sentence:
            if sent["tokens"]:
                char_st = sent["tokens"][0]["characterOffsetBegin"]
                char_end = sent["tokens"][-1]["characterOffsetEnd"]
            else:
                char_st, char_end = 0, 0
            raw_text = sentence[char_st:char_end]
            raw_texts.append(raw_text)
        parsed_sentences.extend(parsed_sentence)

    parsed_rst_list = list()
    for sent, text in zip(parsed_sentences, raw_texts):
        enhanced_dependency_list = sent["enhancedPlusPlusDependencies"]
        dependencies = set()
        for relation in enhanced_dependency_list:
            if relation["dep"] == "ROOT":
                continue
            governor_pos = relation["governor"]
            dependent_pos = relation["dependent"]
            dependencies.add((governor_pos - 1, relation["dep"], dependent_pos - 1))
        dependencies = list(dependencies)
        dependencies.sort(key=lambda x: (x[0], x[2]))

        x = {
            "text": text,
            "dependencies": dependencies,
            "tokens": [t["word"] for t in sent["tokens"]],
        }
        if "pos" in annotators:
            x["pos_tags"] = [t["pos"] for t in sent["tokens"]]
        if "lemma" in annotators:
            x["lemmas"] = [t["lemma"] for t in sent["tokens"]]
        if "ner" in annotators:
            mentions = []
            for m in sent["entitymentions"]:
                if m["ner"] in TYPE_SET and m["text"].lower().strip() not in PRONOUN_SET:
                    mentions.append(
                        {
                            "start": m["tokenBegin"],
                            "end": m["tokenEnd"],
                            "text": m["text"],
                            "ner": m["ner"],
                            "link": None,
                            "entity": None
                        }
                    )

            x["ners"] = [t["ner"] for t in sent["tokens"]]
            x["mentions"] = mentions
        if "parse" in annotators:
            x["parse"] = re.sub(r"\s+", " ", sent["parse"])

        parsed_rst_list.append(x)
    return parsed_rst_list


def iter_files(path):
    """ Walk through all files located under a root path

    :param path: the directory path
    :type path: str
    :return: all file paths in this directory
    :rtype: List[str]
    """

    if os.path.isfile(path):
        yield path
    elif os.path.isdir(path):
        for dirpath, _, file_names in os.walk(path):
            for f in file_names:
                yield os.path.join(dirpath, f)
    else:
        raise RuntimeError('Path %s is invalid' % path)


def index_from(sequence, x, start_from=0):
    """ Index from a specific start point

    :param sequence: a sequence
    :type sequence: List[object]
    :param x: an object to index
    :type x: object
    :param start_from: start point
    :type start_from: int
    :return: indices of the matched objects
    :rtype: List[int]
    """

    indices = []
    for idx in range(start_from, len(sequence)):
        if x == sequence[idx]:
            indices.append(idx)
    return indices


def powerset(iterable, min_size=0, max_size=-1):
    """ Generate all subsets

    :param iterable: a iterable container
    :type iterable: collections.Iterable
    :param min_size: minimum size of subsets
    :type min_size: int (default = 0)
    :param max_size: maximum size of subsets
    :type max_size: int (default = -1)
    :return: all subsets
    :rtype: collections.Iterator

    .. highlight:: python
    .. code-block:: python

        Input:

            [1, 2, 3]

        Output:

            iter([() (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)])
    """

    s = sorted(iterable)
    if max_size == -1:
        max_size = len(s)
    return chain.from_iterable(combinations(s, r) for r in range(min_size, max_size + 1))


def get_clauses(sent_parsed_result, syntax_tree, sep_indices=None):
    """ Split a sentence to subclauses based on constituency parsing

    :param sent_parsed_result: the parsed result of a sentence
    :type sent_parsed_result: Dict[str, object]
    :param syntax_tree: the constituency parsing result
    :type syntax_tree: aser.extract.discourse_parser.SyntaxTree
    :param sep_indices: the separator indices
    :type sep_indices: Union[None, List[int]]
    :return: a list of clauses
    :rtype: List[List[int]]
    """
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

    if sep_indices is None:
        sep_indices = set()
    elif isinstance(sep_indices, (list, tuple)):
        sep_indices = set(sep_indices)
    sent_len = len(sent_parsed_result["tokens"])

    clauses = list()  # (parent, indices)
    clause = list()
    for t_idx, token in enumerate(sent_parsed_result["tokens"]):
        # split the sentence by seps
        valid = token not in CLAUSE_SEPARATOR_SET and t_idx not in sep_indices
        if valid:
            clause.append(t_idx)
        if t_idx == sent_len - 1 or not valid:
            # strip_punctuations
            clause = strip_punctuations(sent_parsed_result, clause)
            if len(clause) > 0:
                clauses.extend(find_clauses(clause))
            clause = list()
    return clauses


def get_prev_token_index(doc_parsed_result, sent_idx, idx, skip_tokens=None):
    """ Get the sentence index and token index of the previous token

    :param doc_parsed_result: the parsed result of a document
    :type doc_parsed_result: List[Dict[str, object]]
    :param sent_idx: current sentence index
    :type sent_idx: int
    :param idx: current token index
    :type idx: int
    :param skip_tokens: the token set to skip
    :type skip_tokens: Union[None, set]
    :return: the sentence index and token index of the previous token
    :rtype: Tuple[int, int]
    """

    if skip_tokens is None:
        skip_tokens = set()
    curr_sent_idx, curr_idx = sent_idx, idx

    for i in range(MAX_ATTEMPT):
        if curr_idx - 1 >= 0:
            curr_idx = curr_idx - 1
        elif curr_sent_idx - 1 >= 0:
            curr_sent_idx = curr_sent_idx - 1
            curr_idx = len(doc_parsed_result[curr_sent_idx]["tokens"]) - 1
        else:
            return -1, -1
        curr_token = doc_parsed_result[curr_sent_idx]["tokens"][curr_idx]
        if curr_token not in skip_tokens:
            return curr_sent_idx, curr_idx
    return -1, -1


def get_next_token_index(doc_parsed_result, sent_idx, idx, skip_tokens=None):
    """ Get the sentence index and token index of the next token

    :param doc_parsed_result: the parsed result of a document
    :type doc_parsed_result: List[Dict[str, object]]
    :param sent_idx: current sentence index
    :type sent_idx: int
    :param idx: current token index
    :type idx: int
    :param skip_tokens: the token set to skip
    :type skip_tokens: Union[None, set]
    :return: the sentence index and token index of the next token
    :rtype: Tuple[int, int]
    """

    if skip_tokens is None:
        skip_tokens = set()
    curr_sent_idx, curr_idx = sent_idx, idx

    for i in range(MAX_ATTEMPT):
        if curr_idx + 1 < len(doc_parsed_result[curr_sent_idx]["tokens"]):
            curr_idx = curr_idx + 1
        elif curr_sent_idx + 1 < len(doc_parsed_result):
            curr_sent_idx = curr_sent_idx + 1
            curr_idx = 0
        else:
            return -1, -1
        curr_token = doc_parsed_result[curr_sent_idx]["tokens"][curr_idx]
        if curr_token not in skip_tokens:
            return curr_sent_idx, curr_idx
    return -1, -1


def strip_punctuations(sent_parsed_result, indices):
    """ Remove the leading and trailing punctuations

    :param sent_parsed_result: the parsed result of a sentence
    :type sent_parsed_result: Dict[str, object]
    :param indices: the token indices
    :type indices: List[int]
    :return: the
    :rtype: List[int]
    """

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
        if indices[valid_idx2 - 1] >= len(sent_parsed_result["tokens"]):
            valid_idx2 -= 1
            continue
        token = sent_parsed_result["tokens"][indices[valid_idx2 - 1]]
        if token in PUNCTUATION_SET or token == "-LCB-" or token == "-LRB-":
            valid_idx2 -= 1
        else:
            break
    if valid_idx1 == 0 and valid_idx2 == len(indices):
        return indices
    else:
        return indices[valid_idx1:valid_idx2]


def process_raw_file(raw_path, processed_path, sentence_parser):
    """ Process a file that contains raw texts

    :param raw_path: the file name to a file that contains raw texts
    :type raw_path: str
    :param processed_path: the file name to a file to store parsed results
    :type processed_path: str
    :param sentence_parser: the sentence parser to parse raw text
    :type sentence_parser: SentenceParser
    :return: the parsed results of the given file
    :rtype: List[List[Dict[str, object]]]
    """

    return sentence_parser.parse_raw_file(raw_path, processed_path, max_len=MAX_LEN)


def load_processed_data(processed_path, parsed_reader):
    """ Load parsed results from disk

    :param processed_path: the file name to a file that stores parsed results
    :type processed_path: str
    :param parsed_reader: the parsed reader to load parsed results
    :type parsed_reader: ParsedReader
    :return: the parsed result
    :rtype: List[List[Dict[str, object]]]
    """

    return parsed_reader.get_parsed_paragraphs_from_file(processed_path)


def extract_file(
    raw_path="",
    processed_path="",
    prefix_to_be_removed="",
    sentence_parser=None,
    parsed_reader=None,
    aser_extractor=None
):
    """ Extract eventualities and relations from a file (which contains raw texts or parsed results)

    :param raw_path: the file path that contains raw texts
    :type raw_path: str (optional)
    :param processed_path: the file path that stores the parsed result
    :type processed_path: str
    :param prefix_to_be_removed: the prefix in sids to remove
    :type prefix_to_be_removed: str
    :param sentence_parser: the sentence parser to parse raw text
    :type sentence_parser: SentenceParser
    :param parsed_reader: the parsed reader to load parsed results
    :type parsed_reader: ParsedReader
    :param aser_extractor: the ASER extractor to extract both eventualities and relations
    :type aser_extractor: BaseASERExtractor
    :return: a dictionary from eid to sids, a dictionary from rid to sids, a dictionary from eid to eventuality, and a dictionary from rid to relation
    :rtype: Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, aser.eventuality.Eventuality], Dict[str, aser.relation.Relation]]
    """

    # process raw data or load processed data
    if os.path.exists(processed_path):
        processed_data = load_processed_data(processed_path, parsed_reader)
    elif os.path.exists(raw_path):
        processed_data = process_raw_file(raw_path, processed_path, sentence_parser)
    else:
        raise ValueError("Error: at least one of raw_path and processed_path should not be None.")

    # remove prefix of sids
    document = list()
    for paragraph in processed_data:
        for sentence in paragraph:
            sentence["doc"] = os.path.splitext(os.path.basename(processed_path))[0]
            sentence["sid"] = sentence["sid"].replace(prefix_to_be_removed, "", 1)
            document.append(sentence)
        # document.append(EMPTY_SENT_PARSED_RESULT)

    eventuality_lists, relation_lists = aser_extractor.extract_from_parsed_result(document)

    # merge eventualities
    eid2sids = defaultdict(list)
    eid2eventuality = dict()
    for sentence, eventuality_list in zip(document, eventuality_lists):
        for eventuality in eventuality_list:
            eid2sids[eventuality.eid].append(sentence["sid"])
            if eventuality.eid not in eid2eventuality:
                eid2eventuality[eventuality.eid] = deepcopy(eventuality)
            else:
                eid2eventuality[eventuality.eid].update(eventuality)

    # merge relations
    rid2sids = defaultdict(list)
    rid2relation = dict()
    len_doc = len(document)

    # SS
    for idx in range(len_doc):
        relation_list = relation_lists[idx]
        for relation in relation_list:
            if sum(relation.relations.values()) > 0:
                rid2sids[relation.rid].append((document[idx]["sid"], document[idx]["sid"]))
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update(relation)
    # PS
    for idx in range(len_doc - 1):
        relation_list = relation_lists[len_doc + idx]
        for relation in relation_list:
            if sum(relation.relations.values()) > 0:
                rid2sids[relation.rid].append((document[idx]["sid"], document[idx + 1]["sid"]))
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update(relation)

    return eid2sids, rid2sids, eid2eventuality, rid2relation
