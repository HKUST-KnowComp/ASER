import os
import sys
try:
    import ujson as json
except:
    import json
from pprint import pprint

class ParsedReader:
    def __init__(self):
        pass

    def recover_dependencies(self, parsed_result):
        new_dependencies = list()
        for dep_r in parsed_result["dependencies"]:
            new_dependencies.append([dep_r[0]-1, dep_r[1], dep_r[2]-1])
        return new_dependencies

    def generate_sid(self, sent, file_name, line_no):
        return file_name + "|" + str(line_no)

    def get_parsed_paragraphs_from_file(self, file_name):
        with open(file_name, "r") as f:
            sent_len = json.loads(f.readline())['sentence_lens']
            paragraphs = list()
            line_no = 1
            para_idx = 0
            while para_idx < len(sent_len):
                paragraph = list()
                end_no = sent_len[para_idx]
                while line_no < end_no:
                    sent = json.loads(f.readline())
                    sent["sid"] = self.generate_sid(sent, file_name, line_no)
                    paragraph.append(sent)
                    line_no += 1
                para_idx += 1
                paragraphs.append(paragraph)
        return paragraphs

    def get_parsed_sent(self, sent_id, ctx_window=0):
        file_name, line_no = sent_id.rsplit("|", 1)
        line_no = int(line_no)
        sent, lctx, rctx = None, list(), list()
        with open(file_name, "r") as f:
            sent_len = json.loads(f.readline())['sentence_lens']
            # print('sent_len:{}'.format(sent_len))
            if len(sent_len) == 0:
                print('id:{} exceeds file limit.. file:{} is empty'.format(sent_id, file_name))
            elif line_no >= sent_len[-1]:
                print('id:{} exceeds file limit.. file:{} only have {} lines'.format(sent_id, file_name, sent_len[-1]-1))
            else:
                [f.readline() for _ in range(line_no-ctx_window)]

                # left ctx
                lctx_num = line_no if line_no-ctx_window < 0 else ctx_window
                for l_line_no in range(line_no-lctx_num, line_no):
                    l_sent = json.loads(f.readline())
                    l_sent["sid"] = self.generate_sid(l_sent, file_name, l_line_no)
                    lctx.append(l_sent)

                # sent
                sent = json.loads(f.readline())
                sent["sid"] = self.generate_sid(sent, file_name, line_no)

                # right ctx
                rctx_num = sent_len[-1]-line_no-1 if line_no+1+ctx_window > sent_len[-1] else ctx_window
                for r_line_no in range(line_no+1, line_no+1+rctx_num):
                    r_sent = json.loads(f.readline())
                    r_sent["sid"] = self.generate_sid(r_sent, file_name, r_line_no)
                    rctx.append(r_sent)
        return {'sent':sent,'lctx':lctx, 'rctx':rctx}


if __name__=='__main__':
    sent_id = '/home/data/corpora/nytimes/nyt_preprocess/parsed/2001/03/13/5.txt|1'
    parse_reader = ParsedReader()
    res = parse_reader.get_parsed_sent(sent_id, 2)
    for k,v in res.items():
        if k == 'sent':
            if v is None:
                pprint({k:v})
            else:
                pprint({k:v['text']})
        else:
            if v is None:
                pprint({k:v})
            else:
                pprint({k:[i['text'] for i in v]})
