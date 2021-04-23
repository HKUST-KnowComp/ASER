import time
import uuid
import zmq
import ujson as json
from functools import wraps
from ..concept import ASERConcept
from ..eventuality import Eventuality
from ..relation import Relation
from ..utils.config import ASERCmd


class ASERClient(object):
    def __init__(self, ip="localhost", port=8000, port_out=8001, timeout=-1):
        """ A client object of ASER

        :param ip: ip address of the server
        :type ip: str
        :param port: port for push request from a client to the server
        :type port: int
        :param port_out: port for Subscribe return data from the server to a server
        :type port_out: int
        :param timeout: client receiver timeout (milliseconds), -1 means no timeout
        :type timeout: float
        """

        self.client_id = str(uuid.uuid4()).encode("utf-8")
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
            this code is from
            https://github.com/hanxiao/bert-as-service/blob/master/client/bert_serving/client/__init__.py
        """
        @wraps(func)
        def arg_wrapper(self, **kw):
            if 'blocking' in kwargs and not kwargs['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, **kw)
            except zmq.error.Again as _e:
                t_e = TimeoutError(
                    'no response from the server (with "timeout"=%d ms), please check the following:'
                    'is the server still online? is the network broken? are "port" and "port_out" correct? '
                    'are you encoding a huge amount of data whereas the timeout is too small for that?' % self.timeout
                )
                raise t_e
            finally:
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)

        return arg_wrapper

    def _send(self, cmd, data):
        request_id = b"%d" % self.request_num
        self.sender.send_multipart([self.client_id, request_id, cmd, data])
        self.request_num += 1
        return request_id

    @_timeout
    def _recv(self, request_id):
        try:
            while True:
                response = self.receiver.recv_multipart()
                if len(response) > 1:
                    if response[1] == request_id:
                        msg = json.loads(response[-1].decode(encoding="utf-8"))
                        return msg
                else:
                    return []
        except Exception as e:
            raise e

    def extract_eventualities(self, sentence):
        """ Extract and linking all eventualities from input sentence

        :param sentence: input sentence (only support one sentence now)
        :type sentence: str
        :return: a dictionary, here is a example while ret_type is "tokens"
        :rtype: List[Dict[str, List]]

        .. highlight:: python
        .. code-block:: python

            Input:

            "The dog barks loudly"

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

        request_id = self._send(ASERCmd.extract_events, sentence.encode("utf-8"))
        msg = self._recv(request_id)
        if not msg:
            return None

        rst_list = []
        for eventuality_encoded_list in msg:
            rst = list()
            for eventuality_encoded in eventuality_encoded_list:
                eventuality = Eventuality().decode(eventuality_encoded, encoding=None)
                macthed_eventuality_encoded = self._exact_match_eventuality(eventuality.eid)
                eventuality.frequency = macthed_eventuality_encoded["frequency"]\
                    if macthed_eventuality_encoded else 0.0
                rst.append(eventuality)
            rst_list.append(rst)
        return rst_list

    def conceptualize_eventuality(self, eventuality):
        request_id = self._send(ASERCmd.conceptualize_event, eventuality.encode("utf-8"))
        msg = self._recv(request_id)
        if not msg:
            return None
        rst_list = list()
        for words, score in msg:
            concept = ASERConcept(words)
            rst_list.append((concept, score))
        return rst_list

    def predict_relation(self, eventuality1, eventuality2):
        """ Predict relations between two events

        :param eventuality1: one `Eventuality` object
        :type eventuality1: aser.eventuality.Eventuality
        :param eventuality1: the other `Eventuality` object
        :type eventuality2: aser.eventuality.Eventuality
        :return: a `Relation` object between the two given eventualities
        :rtype: aser.relation.Relation
        """

        return self._exact_match_relation(eventuality1, eventuality2)

    def fetch_related_eventualities(self, event):
        """ Fetch related events given one event

        :param event: an `Eventuality` object
        :type event: aser.eventuality.Eventuality
        :return: a list of related eventualities associated with corresponding relations
        :rtype: List[Tuple[aser.eventuality.Eventuality, aser.relation.Relation]]
        """

        eid = eventuality.eid.encode("utf-8")
        request_id = self._send(ASERCmd.fetch_related_events, eid)
        msg = self._recv(request_id)
        return [
            (Eventuality().decode(e_encoded, encoding=None), Relation().decode(r_encoded, encoding=None))
            for e_encoded, r_encoded in msg
        ]

    def _exact_match_eventuality(self, eid):
        request_id = self._send(ASERCmd.exact_match_event, eid.encode("utf-8"))
        msg = self._recv(request_id)
        return msg if msg != ASERCmd.none else None

    def _exact_match_relation(self, eventuality1, eventuality2):
        """ Predict relations between two events by exactly matching

        :param eventuality1: one `Eventuality` object
        :type eventuality1: aser.eventuality.Eventuality
        :param eventuality1: the other `Eventuality` object
        :type eventuality2: aser.eventuality.Eventuality
        :return: a `Relation` object between the two given eventualities
        :rtype: aser.relation.Relation
        """

        data = (eventuality1.eid + "$" + eventuality2.eid).encode("utf-8")
        request_id = self._send(ASERCmd.exact_match_relation, data)
        msg = self._recv(request_id)
        if msg == ASERCmd.none:
            return None
        else:
            return Relation().decode(msg, encoding=None)
