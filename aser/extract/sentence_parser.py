import os
try:
    import ujson as json
except:
    import json
from itertools import chain
from aser.extract.utils import get_corenlp_client, parse_sentense_with_stanford
from aser.extract.utils import ANNOTATORS

CORENLP_PATH = "/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27"


class SentenceParser:
    def __init__(self, **kw):
        self.corenlp_path = kw.get("corenlp_path", "")
        self.corenlp_port = kw.get("corenlp_port", 0)
        self.annotators = kw.get("annotators", list(ANNOTATORS))
        
        _, self.is_externel_corenlp = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)

    def close(self):
        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()

    def __del__(self):
        self.close()

    def __generate_sid(self, sent, file_name, sid):
        return file_name + "|" + str(sid)

    def parse_raw_file(self, raw_path, processed_path=None, annotators=None, max_len=1000):
        """ This method parses raw text of a raw file and save the processed data into a file.

        :type raw_path: str
        :type processed_path: str or None
        :type annotators: list or None
        :type max_len: int
        :param raw_path: the file path of the input file
        :param processed_path: the file path of the output file
        :param annotators: the annotators parameter for the stanford corenlp client
        :param max_len: the max length of paragraphs for parsing as a whole
        :return: a list of dicts
        """

        if annotators is None:
            annotators = self.annotators

        para_lens, paragraphs = [], []

        sid = 1
        try:
            with open(raw_path, "r", encoding="ascii", errors="ignore") as f:
                # for line in f:
                #     para = self.parse(line)
                #     para_lens.append(len(para)+sid)
                #     if processed_path:
                #         for sent in para:
                #             sent["sid"] = self.__generate_sid(sent, processed_path, sid)
                #             sid += 1
                #     paragraphs.append(para)
                paragraphs.append("")
                for line in f:
                    if line.startswith(".START") or line == "\n":
                        if len(paragraphs[-1]) != 0:
                            paragraphs.append("")
                    else:
                        paragraphs[-1] += line
                if len(paragraphs[-1]) == 0:
                    paragraphs.pop()
        except Exception as e:
            print(raw_path)
            raise e

        for i in range(len(paragraphs)):
            paragraphs[i] = self.parse(paragraphs[i], annotators=annotators, max_len=max_len)
            para_lens.append(len(paragraphs[i])+sid)
            if processed_path:
                for sent in paragraphs[i]:
                    sent["sid"] = self.__generate_sid(sent, processed_path, sid)
                    sid += 1

        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_lens": para_lens}))
            f.write("\n")
            for para in paragraphs:
                for sent in para:
                    f.write(json.dumps(sent))
                    f.write("\n")
        return paragraphs

    def parse(self, paragraph, annotators=None, max_len=1000):
        if annotators is None:
            annotators = self.annotators

        corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, annotators=annotators)
        parsed_para = parse_sentense_with_stanford(paragraph, corenlp_client, annotators, max_len)

        for sent_idx, sent in enumerate(parsed_para):
            sent["sid"] = self.__generate_sid(sent, "", sent_idx+1)
        
        return parsed_para

if __name__ == "__main__":
    raw_path = "/home/data/corpora/aser/data/nyt/raw/1987/01/01/2698.txt"
    processed_path = "/home/hkeaa/test.jsonl"
    parser = SentenceParser(
        9109, annotators=["tokenize", "ssplit", "parse", "ner"])
    parser.process_raw_file(raw_path, processed_path)
