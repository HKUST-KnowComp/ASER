import os
try:
    import ujson as json
except:
    import json

from aser.extract.utils import get_corenlp_client,parse_sentense_with_stanford

CORENLP_PATH = '/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27'

class SentenceParser:
    def __init__(self,port,annotators,corenlp_path=CORENLP_PATH,max_process_len=2000):
        self.client = get_corenlp_client(corenlp_path,port,annotators)[0]
        self.client_annotators = annotators
        self.threshold = max_process_len

    def __process_paragraph(self,para:str,annotators=None):
        anno = self.client_annotators if annotators is None else annotators
        sentences = []
        for i_p in range(0, len(para), self.threshold):
            content = para[i_p:i_p + self.threshold].strip()
            if content is not None and len(content) > 0:
                tmp = parse_sentense_with_stanford(content, self.client, annotators=anno)
                sentences.extend(tmp)
        return sentences

    def process_raw_file(self,raw_path, processed_path):
        para_lens, sentences = [], []

        for para in open(raw_path):
            tmp = self.__process_paragraph(para)
            para_lens.append(len(tmp))
            sentences.extend(tmp)

        if sentences == [] or para_lens == []:
            print('raw file {} is empty'.format(raw_path))
            return

        para_lens[0] += 1
        for i in range(1, len(para_lens)):
            para_lens[i] += para_lens[i - 1]

        os.makedirs(os.path.dirname(processed_path), exist_ok=True)

        with open(processed_path, 'w') as fw:
            fw.write(json.dumps({'sentence_lens': para_lens}) + '\n')
            for s in sentences:
                fw.write(json.dumps(s) + '\n')

if __name__ == '__main__':
    raw_path = '/home/data/corpora/aser/data/nyt/raw/1987/01/01/2698.txt'
    processed_path = '/home/hkeaa/test.jsonl'
    parser = SentenceParser(9109,annotators=['tokenize','ssplit','parse','ner'])
    parser.process_raw_file(raw_path,processed_path)