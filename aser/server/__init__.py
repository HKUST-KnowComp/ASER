import json
import os
import time
from aser.database.db_API import KG_Connection
from aser.server.corenlp import StanfordCoreNLPServer
from aser.extract.extractor import extract_activity_struct_from_sentence
import zmq


class ASERServer(object):
    def __init__(self, *args, **kwargs):
        total_st = time.time()
        self.corenlp_servers = \
            [StanfordCoreNLPServer(
                corenlp_path=kwargs.get("corenlp_path", "./"),
                port=kwargs.get("base_corenlp_port", 9000) + i)
             for i in range(kwargs.get("corenlp_num", 5))]

        st = time.time()
        print("Connect to the KG...")
        db_dir = kwargs.get("db_dir", "./")
        self.kg_conn = KG_Connection(db_path=os.path.join(db_dir, "KG.db"), mode='cache')
        with open(os.path.join(db_dir, "inverted_table.json"), "r") as f:
            self.kg_inverted_table = json.load(f)
        print("Connect to the KG finished in {:.4f} s".format(time.time() - st))

        # Preloading coreNLP
        for i in range(len(self.corenlp_servers)):
            extract_activity_struct_from_sentence("I am hungry", i)

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:{}".format(kwargs.get("port", 8000)))
        print("Loading Server Finished in {:.4f} s".format(time.time() - total_st))
        self.run()

    def run(self):
        cnt = 0
        while True:
            try:
                data = self.socket.recv_multipart()
                mode = data[0].decode("ascii")
                if mode == "extract_event":
                    sentence = data[1].decode("ascii")
                    print("[EXTRACT_EVENT]         Received sentence: %s" % sentence)
                    rst = extract_activity_struct_from_sentence(sentence, cnt % len(self.corenlp_servers))
                    cnt += 1
                    self.socket.send_json(rst)
                elif mode == "exact_match_event":
                    eid = data[1].decode("ascii")
                    print("[EXACT_MATCH_EVENT]     Received eid: %s" % eid)
                    matched_event = self.kg_conn.get_exact_match_event(eid)
                    self.socket.send_json(matched_event)
                elif mode == "exact_match_relation":
                    eid1, eid2 = data[1].decode("ascii"), data[2].decode("ascii")
                    print("[EXACT_MATCH_RELATION]  Received eids: (%s, %s)" % (eid1, eid2))
                    matched_relation = self.kg_conn.get_exact_match_relation([eid1, eid2])
                    self.socket.send_json(matched_relation)
                elif mode == "get_related_events":
                    eid = data[1].decode("ascii")
                    print("[GET_RELATED_EVENTS]    Received eid: %s" % eid)
                    if eid in self.kg_inverted_table:
                        related_eids = self.kg_inverted_table[eid]
                        related_events = {}
                        for rel, rel_eids in related_eids.items():
                            related_events[rel] = self.kg_conn.get_exact_match_events(rel_eids)
                    else:
                        related_events = {}
                    self.socket.send_json(related_events)
            except:
                self.close()
                break


    def close(self):
        for corenlp in self.corenlp_servers:
            corenlp.close()
        self.socket.close()
        self.kg_conn.close()

    def __exit__(self):
        self.close()
