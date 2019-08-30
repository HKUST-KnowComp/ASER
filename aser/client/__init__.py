import json
import zmq


class ASERClient(object):
    def __init__(self, port):
        context = zmq.Context()

        #  Socket to talk to server
        self.socket = context.socket(zmq.REQ)
        self.socket.connect("tcp://localhost:{}".format(port))


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.socket.close()


    def extract_eventualities(self, sentence):
        self.socket.send_string(sentence)
        msg = self.socket.recv_json()
        return msg