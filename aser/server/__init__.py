import time
from aser.database.db_API import KG_Connection
from aser.server.corenlp import StanfordCoreNLPServer
from aser.extract.extractor import extract_activity_struct_from_sentence
import zmq


class ASERServer(object):
    def __init__(self, *args, **kwargs):
        self.corenlp_servers = \
            [StanfordCoreNLPServer(
                corenlp_path=kwargs.get("corenlp_path", "./"),
                port=kwargs.get("base_corenlp_port", 9000) + i)
             for i in range(kwargs.get("corenlp_num", 5))]

        st = time.time()
        print("Connect to the KG...")
        self.kg_conn = KG_Connection(db_path=kwargs.get("db_path", "./KG.db"), mode='cache')
        print("Connect to the KG finished in {:.4f} s".format(time.time() - st))

        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:{}".format(kwargs.get("port", 8000)))
        self.run()
        print("Loading Server Done..")

    def run(self):
        cnt = 0
        while True:
            mode = self.socket.recv_string()
            self.socket.send(b'yes')
            if mode == "extract_event":
                sentence = self.socket.recv_string()
                print("Received sentence: %s" % sentence)
                rst = extract_activity_struct_from_sentence(sentence, cnt % len(self.corenlp_servers))
                cnt += 1
                self.socket.send_json(rst)
            elif mode == "exact_match_event":
                eid = self.socket.recv_string()
                matched_event = self.kg_conn.get_exact_match_event(eid)
                self.socket.send_json(matched_event)
            elif mode == "exact_match_relation":
                msg = self.socket.recv_string()
                eid1, eid2 = msg.split('$')
                matched_relation = self.kg_conn.get_exact_match_relation([eid1, eid2])
                self.socket.send_json(matched_relation)



    def close(self):
        for corenlp in self.corenlp_servers:
            corenlp.close()
        self.socket.close()
        self.kg_conn.close()

    def __exit__(self):
        self.close()
