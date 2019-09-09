import time
from functools import wraps
import json
import uuid
import zmq
from aser.database.db_API import preprocess_event
from aser.utils.config import ASERCmd

class ASERClient(object):
    def __init__(self, port=8000, port_out=8001, timeout=-1):
        """ A client object of ASER

        :param port <int>: port for push request from a client to the server
        :param port_out <int>: port for Subscribe return data from the server
        to a server
        :param timeout: client receiver timeout (milliseconds), -1 means no
        timeout
        """
        self.client_id = str(uuid.uuid4()).encode("ascii")
        context = zmq.Context()
        self.sender = context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.sender.connect("tcp://localhost:%d" % port)
        self.receiver = context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.client_id)
        self.receiver.connect("tcp://localhost:%d" % port_out)
        self.request_num = 0
        self.timeout = timeout
        # this is a hack, waiting for connection between SUB/PUB
        time.sleep(1)

    def close(self):
        self.sender.close()
        self.receiver.close()

    def _timeout(func):
        """
            Raise timeout error while there's no response for a while
            this code is from https://github.com/hanxiao/bert-as-service/
            blob/master/client/bert_serving/client/__init__.py
        """
        @wraps(func)
        def arg_wrapper(self, *args, **kwargs):
            if 'blocking' in kwargs and not kwargs['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, *args, **kwargs)
            except zmq.error.Again as _e:
                t_e = TimeoutError(
                    'no response from the server (with "timeout"=%d ms), please check the following:'
                    'is the server still online? is the network broken? are "port" and "port_out" correct? '
                    'are you encoding a huge amount of data whereas the timeout is too small for that?' % self.timeout)
                raise t_e
            finally:
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)

        return arg_wrapper

    def _send(self, cmd, data):
        request_id = b"%d" % self.request_num
        self.sender.send_multipart([
            self.client_id, request_id, cmd, data])
        self.request_num += 1
        return request_id

    @_timeout
    def _recv(self, request_id):
        try:
            while True:
                response = self.receiver.recv_multipart()
                if response[1] == request_id:
                    msg = json.loads(response[-1].decode("ascii"))
                    return msg
        except Exception as e:
            raise e

    def extract_eventualities(self, sentence, only_events=False,
                              ret_type="tokens"):

        """ Extract all eventualities from input sentence

        :param sentence <str>: input sentence. only support one sentence now.
        :param only_events <bool>: output eventualities only
        :param ret_type <str>: "tokens" or "parsed_relations"

        :return: a dictionary, here is a example while ret_type is "tokens"
            Input: 'I am in the kitchen .'

            Output:
            {
                "sentence": 'I am in the kitchen .'
                "eventualities": [
                    {
                        'eid': '2489e3d0a017aca73d30ccc334140869950aad90',
                        'frequency': 800.0,
                        'pattern': 's-be-a',
                        'skeleton_words': 'i be kitchen',
                        'verbs': 'be',
                        'words': 'i be in the kitchen'
                    }
                ]
            }
        """
        request_id = self._send(
            ASERCmd.extract_events, sentence.encode("ascii"))
        msg = self._recv(request_id)
        if not msg:
            return None
        msg = msg[0]
        ret_dict = dict()
        if not only_events:
            if ret_type == "parsed_relations":
                ret_dict["sentence"] = msg["sentence_parsed_relations"]
            elif ret_type == "tokens":
                ret_dict["sentence"] = " ".join(msg["sentence_tokens"])
            else:
                raise RuntimeError("`ret_type` should be 'tokens' or 'parsed_relations'")

        events = list()
        for pattern, activity in msg['activity_list']:
            e = preprocess_event(activity, pattern)
            e["eid"] = e["_id"]
            tmp = self._exact_match_event(e)
            e = tmp if tmp else e
            event_dict = dict()
            event_dict["eid"] = e["_id"]
            event_dict["pattern"] = pattern
            event_dict["verbs"] = e["verbs"]
            event_dict["frequency"] = e["frequency"] if tmp else 0.0
            if ret_type == "parsed_relations":
                event_dict["skeleton_words"] = activity["skeleton_words"]
                event_dict["words"] = activity["words"]
            elif ret_type == "tokens":
                event_dict["skeleton_words"] = e["skeleton_words"]
                event_dict["words"] = e["words"]
            else:
                raise RuntimeError("`ret_type` should be 'tokens' or 'parsed_relations'")
            events.append(event_dict)

        if only_events:
            return events
        else:
            ret_dict["eventualities"] = events
            return ret_dict

    def predict_relation(self, event1, event2, only_exact=False):
        """ Predict relations between two events

        :param event1 <dict>: eventuality dict, should include "eid"
        :param event2 <dict>: eventuality dict, should include "eid"
        :param only_exact <bool>: only return exactly matched relations
        :return: a dictionary of dictionaries
        """
        ret_dict = dict()
        exact_match_rels = self._exact_match_relation(event1, event2)
        if only_exact:
            return exact_match_rels
        else:
            ret_dict["exact_match"] = exact_match_rels
            #TODO probabilistic match results
            return ret_dict

    def fetch_related_events(self, event):
        """ Fetch related events given one event

        :param event <dict>: eventuality dict, should include "eid"
        :return: a dictionary of each relation-related events
        """
        eid = event['eid'].encode("ascii")
        request_id = self._send(ASERCmd.fetch_related_events, eid)
        msg = self._recv(request_id)
        for rel_type, elist in msg.items():
            for e in elist:
                e["eid"] = e.pop("_id")
                del e["skeleton_words_clean"]
        return msg

    def _exact_match_event(self, event):
        eid = event['eid'].encode("ascii")
        request_id = self._send(ASERCmd.exact_match_event, eid)
        msg = self._recv(request_id)
        return msg

    def _exact_match_relation(self, event1, event2):
        data = (event1['eid'] + "$" + event2['eid']).encode("ascii")
        request_id = self._send(ASERCmd.exact_match_relation, data)
        msg = self._recv(request_id)
        if not msg:
            return {}
        del msg['_id']
        del msg['event1_id']
        del msg['event2_id']
        return {key: val for key, val in msg.items() if val > 0.0}