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

    def extract_eventualities_struct(self, sentence):
        self.socket.send_multipart([b"extract_event", sentence.encode("ascii")])
        msg = self.socket.recv_json()
        if msg and msg[0]['activity_list']:
            pattern, event  = msg[0]['activity_list'][0]
            event['pattern'] = pattern
        else:
            pattern, event = None, None
        return event

    def extract_eventualities(self, sentence):
        self.socket.send_multipart([b"extract_event", sentence.encode("ascii")])
        msg = self.socket.recv_json()
        if msg and msg[0]['activity_list']:
            pattern, event  = msg[0]['activity_list'][0]
            e = preprocess_event(event, pattern)
        else:
            e = None
        return e

    def get_exact_match_event(self, event):
        eid = event['_id'].encode("ascii")
        self.socket.send_multipart([b"exact_match_event", eid])
        matched_event = self.socket.recv_json()
        return matched_event


    def get_exact_match_relation(self, event1, event2):
        eid1 = event1['_id'].encode("ascii")
        eid2 = event2['_id'].encode("ascii")
        self.socket.send_multipart([b"exact_match_relation", eid1, eid2])
        matched_relation = self.socket.recv_json()
        return matched_relation

    def get_related_events(self, event):
        eid = event['_id'].encode("ascii")
        self.socket.send_multipart([b"get_related_events", eid])
        related_events = self.socket.recv_json()
        return related_events
