import os,sys,json,re
from pprint import pprint

id = '/home/data/corpora/nytimes/nyt_preprocess/parsed/2001/03/13/5.txt@1'

class ParsedReader:
    def __init__(self):
        pass

    def get_parsed_sent(self,id:str,ctx_window=0):
        file_name,sent_id = id.split('@')
        sent_id = int(sent_id)
        sent,lctx,rctx=None,None,None
        with open(file_name) as f:

            sent_len = json.loads(f.readline())['sentence_lens']
            # print('sent_len:{}'.format(sent_len))
            if sent_len!=[] and sent_id < sent_len[-1]-1:

                [f.readline() for _ in range(sent_id-ctx_window)]
                # left ctx
                lctx_num = sent_id if sent_id-ctx_window<0 else ctx_window
                lctx = [json.loads(f.readline()) for _ in range(lctx_num)]

                sent = json.loads(f.readline().strip())
                # right ctx
                for i in range(ctx_window):
                    line = f.readline().strip()
                    if line != '':
                        if rctx is None:
                            rctx = []
                        rctx.append(json.loads(line))
            else:
                if sent_len == []:
                    print('id:{} exceeds file limit.. file:{} is empty'.format(sent_id,file_name))
                else:
                    print('id:{} exceeds file limit.. file:{} only have {} lines'.format(sent_id,file_name,sent_len[-1]-1))
        return {'sent':sent,'lctx':lctx,'rctx':rctx}

    def get_file(self,file_name):
        sents = []
        for i_l,line in enumerate(open(file_name)):
            line = line.strip()
            if i_l != 0:
                sent = json.loads(line)
                sents.append(sent)
            else:
                meta = json.loads(line)
                if meta['sentence_lens'] == []:
                    break
        return sents

# find mention locations in a eventuality
def find_mention_from_event(ment:str,event:str):
    comp = re.compile(ment)
    res = comp.finditer(event)
    return [m.span() for m in res]
if __name__=='__main__':
    parse_reader = ParsedReader()
    # res = parse_reader.get_parsed_sent(id,2)
    # for k,v in res.items():
    #     if k == 'sent':
    #         if v is None:
    #             pprint({k:v})
    #         else:
    #             pprint({k:v['raw']})
    #     else:
    #         if v is None:
    #             pprint({k:v})
    #         else:
    #             pprint({k:[i['raw'] for i in v]})

    res = parse_reader.get_file(id[:-2])
    print(res)
