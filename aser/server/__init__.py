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
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)
        self.socket.bind("tcp://*:{}".format(kwargs.get("port", 8000)))
        self.run()

    def run(self):
        cnt = 0
        while True:
            #  Wait for next request from client
            sentence = self.socket.recv_string()
            print("Received sentence: %s" % sentence)
            rst = extract_activity_struct_from_sentence(sentence, cnt % len(self.corenlp_servers))
            cnt += 1
            # print(rst)
            self.socket.send_json(rst)


    def close(self):
        for corenlp in self.corenlp_servers:
            corenlp.close()
        self.socket.close()

    def __exit__(self):
        self.close()
