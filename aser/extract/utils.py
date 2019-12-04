import os
import socket, json, re, time
from stanfordnlp.server import CoreNLPClient
from pprint import pprint
from multiprocessing import Pool
from multiprocessing import Manager
from typing import List, Set
from tqdm import tqdm
from nltk.tokenize import sent_tokenize
import requests

_VALID_CHARS = frozenset("""qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890`~!@#$%^&*/?., ;:"'""")


def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False
        # try:
        #     return requests.get(f"http://localhost:{port}/ping").ok
        # except requests.exceptions.ConnectionError as e:
        #     return False


def get_corenlp_client(corenlp_path, port, annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'], props=None):
    os.environ["CORENLP_HOME"] = corenlp_path
    print("Starting corenlp client at port {}".format(port))

    if is_port_occupied(port=port):
        print('server started before..')
        try:
            corenlp_client = CoreNLPClient(
                annotators=annotators, timeout=60000,
                memory='5G', endpoint="http://localhost:%d" % port,
                start_server=False, be_quiet=True, properties=props)
            corenlp_client.annotate("hello world",
                                    annotators=annotators,
                                    output_format="json")
            return corenlp_client
        except Exception as err:
            raise err
    else:
        print('server down.. booting')
        corenlp_client = CoreNLPClient(
            annotators=annotators, timeout=60000,
            memory='5G', endpoint="http://localhost:%d" % port,
            start_server=True, be_quiet=True, properties=props)
        corenlp_client.annotate("hello world",
                                annotators=annotators,
                                output_format="json")
        return corenlp_client


def parse_sentense_with_stanford(input_sentence, corenlp_client,
                                 annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse']):
    type_set = {'CITY', 'ORGANIZATION', 'COUNTRY', 'STATE_OR_PROVINCE', 'LOCATION', 'NATIONALITY', 'PERSON'}
    pronoun_set = {'I', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                   'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it',
                   'its', 'itself', 'they', 'them', 'their', 'theirs', 'themself', 'themselves', }

    def clean_sentence_for_parsing(input_sentence):
        input_sentence = re.sub(r'[ \t]{2,}', ' ', input_sentence)
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

        if 'ner' in annotators:
            mentions = []
            for m in s['entitymentions']:
                if m['ner'] in type_set and m['text'].lower().strip() not in pronoun_set:
                    mentions.append({'start': m['tokenBegin'], 'end': m['tokenEnd'], 'text': m['text'], 'ner': m['ner'],
                                     'link': None, 'entity': None})

            parsed_rst_list[-1]['ners'] = [t['ner'] for t in s['tokens']]
            parsed_rst_list[-1]['mentions'] = mentions

        if 'parse' in annotators:
            parsed_rst_list[-1]['parse'] = s['parse']

    return parsed_rst_list


def parse_sentense_with_stanford_split(input_sentence, corenlp_client,
                                       annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse']):
    type_set = {'CITY', 'ORGANIZATION', 'COUNTRY', 'STATE_OR_PROVINCE', 'LOCATION', 'NATIONALITY', 'PERSON'}
    pronoun_set = {'I', 'me', 'my', 'mine', 'myself', 'we', 'us', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                   'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it',
                   'its', 'itself', 'they', 'them', 'their', 'theirs', 'themself', 'themselves', }
    EMPTY_SENTENCE_PARSED_RESULT = {'text': '.', 'dependencies': [], 'tokens': ['.']}
    if 'lemma' in annotators:
        EMPTY_SENTENCE_PARSED_RESULT['lemmas'] = ['.']
    if 'ner' in annotators:
        EMPTY_SENTENCE_PARSED_RESULT['ners'] = ['.']
        EMPTY_SENTENCE_PARSED_RESULT['mentions'] = ['.']
    if 'pos' in annotators:
        EMPTY_SENTENCE_PARSED_RESULT['pos_tags'] = ['.']
    if 'parse' in annotators:
        EMPTY_SENTENCE_PARSED_RESULT['parse'] = '(ROOT\r\n  (NP (. .)))'

    threshold = 230

    # release version
    # def clean_sentence_for_parsing(text):
    #     re.sub(r'[^\x00-\x7F]+', ' ', text)

    def clean_sentence_for_parsing(input_sentence):
        input_sentence = re.sub(r'[ \t]{2,}', ' ', input_sentence)
        new_sentence = ''
        for char in input_sentence:
            if char in _VALID_CHARS:
                new_sentence += char
            else:
                new_sentence += '\n'
        return new_sentence

    cleaned_para = clean_sentence_for_parsing(input_sentence)
    raw_sentences = corenlp_client.annotate(cleaned_para, annotators=['tokenize', 'ssplit', ],
                                            output_format='json')
    parsed_rst_list = list()
    # annotators_wo_tokenize = [a for a in annotators if a not in {'tokenize','ssplit'}]

    for sent in raw_sentences['sentences']:

        if sent['tokens']:
            char_st = sent['tokens'][0]['characterOffsetBegin']
            char_end = sent['tokens'][-1]['characterOffsetEnd']
        else:
            char_st, char_end = 0, 0

        raw_text = cleaned_para[char_st:char_end]
        if len(raw_text) > threshold:
            parsed_rst_list.append(EMPTY_SENTENCE_PARSED_RESULT)
            continue
        try:
            tmp_output = corenlp_client.annotate(raw_text, annotators=annotators, output_format='json'
                                                 # properties={
                                                 #     'annotators': annotators,
                                                 #     # 'tokenize.language': 'Whitespace',
                                                 #     # 'tokenize.whitespace': 'true',  # first property
                                                 #     # 'ssplit.eolonly': 'true',  # second property
                                                 #     'outputFormat': 'json'
                                                 # }
                                                 )
        except:
            parsed_rst_list.append(EMPTY_SENTENCE_PARSED_RESULT)
            continue

        s = tmp_output['sentences'][0]

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

        parsed_rst_list.append({
            "text": raw_text,
            "dependencies": dependencies,
            "tokens": [t['word'] for t in s['tokens']],
        })
        if 'pos' in annotators:
            parsed_rst_list[-1]['pos_tags'] = [t['pos'] for t in s['tokens']]
        if 'lemma' in annotators:
            parsed_rst_list[-1]["lemmas"] = [t['lemma'] for t in s['tokens']]
        if 'ner' in annotators:
            mentions = []
            for m in s['entitymentions']:
                if m['ner'] in type_set and m['text'].lower().strip() not in pronoun_set:
                    mentions.append({'start': m['tokenBegin'], 'end': m['tokenEnd'], 'text': m['text'], 'ner': m['ner'],
                                     'link': None, 'entity': None})

            parsed_rst_list[-1]['ners'] = [t['ner'] for t in s['tokens']]
            parsed_rst_list[-1]['mentions'] = mentions
        if 'parse' in annotators:
            parsed_rst_list[-1]['parse'] = s['parse']

    return parsed_rst_list


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


if __name__ == '__main__':
    # para = "Barack Hussein Obama II 奥巴马 (/bəˈrɑːk huːˈseɪn oʊˈbɑːmə/ (About this soundlisten);[1] born August 4, 1961) is an American attorney and politician who served as the 44th president 美国总统 of the United States from 2009 to 2017. A member of the Democratic Party 民主党, he was the first African 非裔美国人 American to be elected to the presidency. He previously served as a U.S. senator from Illinois from 2005 to 2008 and an Illinois 伊利诺伊州 state senator from 1997 to 2004."
    # para = '"Hello, I am Linda! And you are?" Linda said, "I am so glad to see you!" Tom is depressed? He want to talk: Aloha Linda!'
    para = "After winning re-election by defeating Republican opponent Mitt Romney, Obama was sworn in for a second term in 2013. During this term, he promoted inclusiveness for LGBT Americans. His administration filed briefs that urged the Supreme Court to strike down same-sex marriage bans as unconstitutional (United States v. Windsor and Obergefell v. Hodges); same-sex marriage was fully legalized in 2015 after the Court ruled that a same-sex marriage ban was unconstitutional in Obergefell. He advocated for gun control in response to the Sandy Hook Elementary School shooting, indicating support for a ban on assault weapons, and issued wide-ranging executive actions concerning global warming and immigration. In foreign policy, he ordered military intervention in Iraq in response to gains made by ISIL after the 2011 withdrawal from Iraq, continued the process of ending U.S. combat operations in Afghanistan in 2016, promoted discussions that led to the 2015 Paris Agreement on global climate change, initiated sanctions against Russia following the invasion in Ukraine and again after Russian interference in the 2016 United States elections, brokered a nuclear deal with Iran, and normalized U.S. relations with Cuba. Obama nominated three justices to the Supreme Court: Sonia Sotomayor and Elena Kagan were confirmed as justices, while Merrick Garland faced unprecedented partisan obstruction and was ultimately not confirmed. During his term in office, America's soft power and reputation abroad significantly improved."
    # print(f'raw: {para}')
    # split_counter,parse_counter = test_tokenize_ratio(para)
    # print('split time:{:.2f}s ratio:{:.4f}s'.format(split_counter,split_counter/(split_counter+parse_counter)))
    # print('parse time:{:.2f}s ratio:{:.4f}s'.format(parse_counter, parse_counter / (split_counter + parse_counter)))
    corenlp_path = '/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27'
    # anno = ['tokenize','ssplit','pos','lemma','depparse','ner']
    anno = ['tokenize', 'ssplit', 'pos', 'lemma', 'parse']
    client = get_corenlp_client(corenlp_path, 9110, annotators=anno)
    res = parse_sentense_with_stanford(para, client, annotators=anno)
    for r in res:
        print(r)
