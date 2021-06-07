import time
import json
from .utils import get_corenlp_client, parse_sentense_with_stanford
from .utils import ANNOTATORS, MAX_LEN


class SentenceParser:
    """ Sentence parser to process files that contain raw texts

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

    def close(self):
        """ Close the parser safely

        """
        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()

    def __del__(self):
        self.close()

    def generate_sid(self, sentence, file_name, sid):
        """

        :param sentence: the raw text
        :type sentence: str
        :param file_name: the file name
        :type file_name: str
        :param line_no: the line number
        :type line_no: int
        :return: the corresponding sentence id
        :rtype: str
        """

        return file_name + "|" + str(sid)

    def parse_raw_file(self, raw_path, processed_path=None, annotators=None, max_len=MAX_LEN):
        """ Parse all raw texts in the given file

        :param raw_path: the file path that contains raw texts
        :type raw_path: str
        :param processed_path: the file path that stores the parsed result
        :type processed_path: str
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
        :type max_len: int (default = 1024)
        :return: the parsed result
        :rtype: List[List[Dict[str, object]]]
        """

        if annotators is None:
            annotators = self.annotators

        paragraphs = []
        try:
            with open(raw_path, "r", encoding="utf-8", errors="ignore") as f:
                paragraphs.append("")
                for line in f:
                    if line.startswith(".START") or line == "\n":
                        if len(paragraphs[-1]) != 0:
                            paragraphs.append("")
                    else:
                        paragraphs[-1] += line
                if len(paragraphs[-1]) == 0:
                    paragraphs.pop()
        except BaseException as e:
            print(raw_path)
            print(e)
            raise e

        if len(paragraphs) == 1 and len(paragraphs[0]) > max_len:
            paragraphs = paragraphs[0].split("\n")
        sid = 1
        para_lens = []
        for i in range(len(paragraphs)):
            paragraphs[i] = self.parse(paragraphs[i], annotators=annotators, max_len=max_len)
            para_lens.append(len(paragraphs[i]) + sid)
            if processed_path:
                for sent in paragraphs[i]:
                    sent["sid"] = self.generate_sid(sent, processed_path, sid)
                    sid += 1

        with open(processed_path, "w", encoding="utf-8") as f:
            f.write(json.dumps({"sentence_lens": para_lens}))
            f.write("\n")
            for para in paragraphs:
                for sent in para:
                    f.write(json.dumps(sent))
                    f.write("\n")
        return paragraphs

    def parse(self, paragraph, annotators=None, max_len=MAX_LEN):
        """

        :param paragraph: a raw text
        :type paragraph: str
        :param annotators: annotators for corenlp, please refer to https://stanfordnlp.github.io/CoreNLP/annotators.html
        :type annotators: Union[List, None] (default = None)
        :param max_len: the max length of a paragraph (constituency parsing cannot handle super-long sentences)
        :type max_len: int (default = 1024)
        :return: the parsed result
        :rtype: List[Dict[str, object]]
        """
        if annotators is None:
            annotators = self.annotators

        corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, annotators=annotators)
        parsed_para = parse_sentense_with_stanford(paragraph, corenlp_client, annotators, max_len)

        for sent_idx, sent in enumerate(parsed_para):
            sent["sid"] = self.generate_sid(sent, "", sent_idx+1)
        
        return parsed_para

if __name__ == "__main__":
    raw_path = "/Users/sean/OneDrive - HKUST Connect/Documents/HKUST/Research/ASER/example_data/raw/yelp.txt"
    processed_path = "/Users/sean/OneDrive - HKUST Connect/Documents/HKUST/Research/ASER/example_data/processed/yelp.jsonl"
    parser = SentenceParser(corenlp_path="", corenlp_port=9000, annotators=list(ANNOTATORS))
    start_st = time.time()
    pared_para = parser.parse_raw_file(raw_path, processed_path)
    end_st = time.time()
    print("# Tokens: %d\tTime: %.4fs" % (sum([sum([len(sent["tokens"]) for sent in para]) for para in pared_para]), end_st-start_st))
