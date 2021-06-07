import json
from pprint import pprint


class ParsedReader(object):
    """ File reader to read parsed results from Disk

    """
    def __init__(self):
        pass

    def __del__(self):
        self.close()

    def close(self):
        pass

    def generate_sid(self, sentence, file_name, line_no):
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

        return file_name + "|" + str(line_no)

    def get_parsed_paragraphs_from_file(self, processed_path):
        """ This method retrieves all paragraphs from a processed file

        :type processed_path: str or None
        :param processed_path: the file path of the processed file
        :return: a list of lists of dicts
        """
        with open(processed_path, "r") as f:
            sent_len = json.loads(f.readline())["sentence_lens"]
            paragraphs = list()
            line_no = 1
            para_idx = 0
            while para_idx < len(sent_len):
                paragraph = list()
                end_no = sent_len[para_idx]
                while line_no < end_no:
                    sent = json.loads(f.readline())
                    sent["sid"] = self.generate_sid(sent, processed_path, line_no)
                    paragraph.append(sent)
                    line_no += 1
                para_idx += 1
                paragraphs.append(paragraph)
        return paragraphs

    def get_parsed_sentence_and_context(self, sid, context_window_size=0):
        """ Retrieve the parsed results of the corresponding sentence and its context

        :param sid: the sentence id
        :type sid: str
        :param context_window_size: the context window size
        :type context_window_size: int (default = 0)
        :return: a dictionary that contains the "sentence", "left_context", and "right_context"
        :rtype: Dict[str, object]
        """

        file_name, line_no = sid.rsplit("|", 1)
        line_no = int(line_no)
        sent, lctx, rctx = None, list(), list()
        with open(file_name, "r") as f:
            sent_len = json.loads(f.readline())["sentence_lens"]
            if len(sent_len) == 0:
                print("id:{} exceeds file limit.. file:{} is empty".format(sid, file_name))
            elif line_no >= sent_len[-1]:
                print("id:{} exceeds file limit.. file:{} only have {} lines".format(sid, file_name, sent_len[-1] - 1))
            else:
                for _ in range(line_no - 1 - context_window_size):
                    f.readline()

                # left context
                lctx_num = line_no - 1 if line_no - context_window_size < 1 else context_window_size
                for l_line_no in range(line_no - lctx_num, line_no):
                    l_sent = json.loads(f.readline())
                    l_sent["sid"] = self.generate_sid(l_sent, file_name, l_line_no)
                    lctx.append(l_sent)

                # sentence
                sent = json.loads(f.readline())
                sent["sid"] = self.generate_sid(sent, file_name, line_no)

                # right context
                rctx_num = sent_len[-1] - line_no - 1 if line_no + 1 + context_window_size > sent_len[
                    -1] else context_window_size
                for r_line_no in range(line_no + 1, line_no + 1 + rctx_num):
                    r_sent = json.loads(f.readline())
                    r_sent["sid"] = self.generate_sid(r_sent, file_name, r_line_no)
                    rctx.append(r_sent)
        return {"sentence": sent, "left_context": lctx, "right_context": rctx}


if __name__ == "__main__":
    sid = "/Users/sean/OneDrive - HKUST Connect/Documents/HKUST/Research/ASER/example_data/processed/yelp.jsonl|1"
    parse_reader = ParsedReader()
    res = parse_reader.get_parsed_sentence_and_context(sid, 2)
    for k, v in res.items():
        if k == "sentence":
            if v is None:
                pprint({k: v})
            else:
                pprint({k: v["text"]})
        else:
            if v is None:
                pprint({k: v})
            else:
                pprint({k: [i["text"] for i in v]})
