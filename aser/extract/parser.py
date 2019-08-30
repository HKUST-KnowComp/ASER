import math
from pycorenlp import StanfordCoreNLP
from functools import partial
from multiprocessing import Pool

# corenlp ports
no_nlp_server = 15
nlp_list = [StanfordCoreNLP('http://localhost:900%d' % (i)) for i in range(no_nlp_server)]

valid_chars = set("""qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890`~!@#$%^&*/?., ;:"'""")

def clean_sentence_for_parsing(input_sentence):
    new_sentence = ''
    for char in input_sentence:
        if char in valid_chars:
            new_sentence += char
        else:
            new_sentence += '\n'
    return new_sentence

def parse_sentense_with_stanford(input_sentence, nlp_id=0):
    nlp = nlp_list[nlp_id]
    cleaned_sentence = clean_sentence_for_parsing(input_sentence)
    tmp_output = nlp.annotate(cleaned_sentence,
                              properties={'annotators': 'tokenize,depparse,lemma', 'outputFormat': 'json'})
    parsed_examples = list()
    for s in tmp_output['sentences']:
        enhanced_dependency_list = s['enhancedPlusPlusDependencies']
        stored_dependency_list = list()
        for relation in enhanced_dependency_list:
            if relation['dep'] == 'ROOT':
                continue
            governor_position = relation['governor']
            dependent_position = relation['dependent']
            stored_dependency_list.append(((governor_position, s['tokens'][governor_position - 1]['lemma'],
                                            s['tokens'][governor_position - 1]['pos']), relation['dep'], (
                                               dependent_position, s['tokens'][dependent_position - 1]['lemma'],
                                               s['tokens'][dependent_position - 1]['pos'])))
        tokens = list()
        for token in s['tokens']:
            tokens.append(token['word'])
        parsed_examples.append(
            {'parsed_relations': stored_dependency_list, 'sentence': input_sentence, 'tokens': tokens})
    return parsed_examples

def parse_sentenses_with_stanford(input_sentences, nlp_id=0):
    return [parse_sentense_with_stanford(s, nlp_id=nlp_id) for s in input_sentences]

def parse_sentences_with_stanford_multicore(input_sentences, n_workers=8):
    batch_size = math.ceil(len(input_sentences) / n_workers)
    pool = Pool(n_workers)
    pool_results= []
    for i in range(0, len(input_sentences), batch_size):
        j = min(len(input_sentences), i+batch_size)
        pool_results.append(pool.apply_async(partial(parse_sentenses_with_stanford, nlp_id=i//batch_size%no_nlp_server), input_sentences[i:j]))
    pool.close()
    pool.join()
    parsed_results = []
    for x in pool_results:
        parsed_results.extend(x.get())
    return parsed_results