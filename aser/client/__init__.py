import time
from functools import wraps
import json
import uuid
import zmq
from aser.utils.config import ASERCmd

class ASERClient(object):
    def __init__(self, ip="localhost", port=8000, port_out=8001, timeout=-1):
        """ A client object of ASER

        :type ip: str
        :type port: int
        :type port_out: int
        :type timeout: float
        :param ip: ip address of the server
        :param port: port for push request from a client to the server
        :param port_out: port for Subscribe return data from the server to a server
        :param timeout: client receiver timeout (milliseconds), -1 means no timeout
        """
        self.client_id = str(uuid.uuid4()).encode("ascii")
        context = zmq.Context()
        self.sender = context.socket(zmq.PUSH)
        self.sender.setsockopt(zmq.LINGER, 0)
        self.sender.connect("tcp://{}:{}".format(ip, port))
        self.receiver = context.socket(zmq.SUB)
        self.receiver.setsockopt(zmq.LINGER, 0)
        self.receiver.setsockopt(zmq.SUBSCRIBE, self.client_id)
        self.receiver.connect("tcp://{}:{}".format(ip, port_out))
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

    def extract_eventualities(self, sentence, only_events=False):
        """ Extract and linking all eventualities from input sentence

        :type sentence: str
        :type only_events: bool
        :param sentence: input sentence. only support one sentence now.
        :param only_events: output eventualities only
        :return: a dictionary, here is a example while ret_type is "tokens"
        :rtype: dict

        .. highlight:: python
        .. code-block:: python

            Input: 'The dog barks loudly'

            Output:

            [{'eventuality_list': [{'dependencies': [[[2, 'dog', 'NN'],
                                                      'det',
                                                      [1, 'the', 'DT']],
                                                     [[3, 'bark', 'VBZ'],
                                                      'nsubj',
                                                      [2, 'dog', 'NN']],
                                                     [[3, 'bark', 'VBZ'],
                                                      'advmod',
                                                      [4, 'loudly', 'RB']]],
                                    'eid': 'b47ba21a77206552509f2cb0c751b959aaa3a625',
                                    'frequency': 0.0,
                                    'pattern': 's-v',
                                    'skeleton_dependencies': [[[3, 'bark', 'VBZ'],
                                                               'nsubj',
                                                               [2, 'dog', 'NN']]],
                                    'skeleton_words': [['dog', 'NN'],
                                                       ['bark', 'VBZ']],
                                    'verbs': 'bark',
                                    'words': [['the', 'DT'],
                                              ['dog', 'NN'],
                                              ['bark', 'VBZ'],
                                              ['loudly', 'RB']]}],
            'sentence_dependencies': [[[2, 'dog', 'NN'], 'det', [1, 'the', 'DT']],
                                      [[3, 'bark', 'VBZ'], 'nsubj', [2, 'dog', 'NN']],
                                      [[3, 'bark', 'VBZ'], 'advmod', [4, 'loudly', 'RB']],
                                      [[3, 'bark', 'VBZ'], 'punct', [5, '.', '.']]],
            'sentence_tokens': [['the', 'DT'],
                                ['dog', 'NN'],
                                ['bark', 'VBZ'],
                                ['loudly', 'RB'],
                                ['.', '.']]}]
        """
        request_id = self._send(
            ASERCmd.extract_events, sentence.encode("ascii"))
        msg = self._recv(request_id)
        if not msg:
            return None

        rst_list = []
        for rst in msg:
            for eventuality in rst["eventuality_list"]:
                tmp = self._exact_match_event(eventuality)
                eventuality["frequency"] = tmp["frequency"] if tmp else 0.0
            if only_events:
                rst_list.extend(rst["eventuality_list"])
            else:
                rst_list.append(rst)
        return rst_list


    def predict_relation(self, event1, event2, only_exact=False):
        """ Predict relations between two events

        :type event1: dict
        :type event2: dict
        :type only_exact: bool
        :param event1: eventuality dict, should include "eid"
        :param event2: eventuality dict, should include "eid"
        :param only_exact: only return exactly matched relations
        :return: a dictionary of dictionaries
        :rtype: dict
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

        :type event: dict
        :param event <dict>: eventuality dict, should include "eid"
        :return: a dictionary of each relation-related events
        :rtype: dict
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