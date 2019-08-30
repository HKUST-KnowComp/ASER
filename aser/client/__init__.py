import zmq
from aser.database.db_API import preprocess_event

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
        self.socket.send_string("extract_event")
        self.socket.recv()
        self.socket.send_string(sentence)
        msg = self.socket.recv_json()
        pattern, event  = msg[0]['activity_list'][0]
        event['pattern'] = pattern
        return event

    def get_exact_match_event(self, event):
        self.socket.send_string("exact_match_event")
        self.socket.recv()
        pattern = event['pattern']
        del event['pattern']
        eid = preprocess_event(event, pattern)['_id']
        self.socket.send_string(eid)
        matched_event = self.socket.recv_json()
        return matched_event


    def get_exact_match_relation(self, event1, event2):
        eid1 = event1['_id']
        eid2 = event2['_id']
        self.socket.send_string("exact_match_relation")
        self.socket.recv()
        self.socket.send_string(eid1 + '$' + eid2)
        matched_relation = self.socket.recv_json()
        return matched_relation

