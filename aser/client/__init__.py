import time
import uuid
import zmq
import ujson as json
from functools import wraps
from ..concept import ASERConcept
from ..eventuality import Eventuality
from ..relation import Relation
from ..utils.config import ASERCmd, ASERError


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
        def arg_wrapper(self, *args, **kw):
            if 'blocking' in kw and not kw['blocking']:
                # override client timeout setting if `func` is called in non-blocking way
                self.receiver.setsockopt(zmq.RCVTIMEO, -1)
            else:
                self.receiver.setsockopt(zmq.RCVTIMEO, self.timeout)
            try:
                return func(self, *args, **kw)
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

                        if isinstance(msg, str) and msg.startswith(ASERError):
                            msg = msg[len(ASERError):-1]
                            start_idx = msg.index("(")
                            if msg[:start_idx] == "ValueError":
                                raise ValueError(msg[start_idx+1:])
                            elif msg[:start_idx] == "TimeoutError":
                                raise TimeoutError(msg[start_idx+1:])
                            elif msg[:start_idx] == "AttributeError":
                                raise AttributeError(msg[start_idx + 1:])
                        return msg
                else:
                    return []
        except BaseException as e:
            raise e

    def extract_eventualities(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        else:
            data = json.dumps(data).encode("utf-8")
        request_id = self._send(ASERCmd.extract_eventualities, data)
        msg = self._recv(request_id)
        if not msg:
            return None

        ret_data = []
        for sent_eventualities in msg:
            rst = list()
            for e_encoded in sent_eventualities:
                eventuality = Eventuality().decode(e_encoded, encoding=None)
                rst.append(eventuality)
            ret_data.append(rst)
        return ret_data

    def extract_relations(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        else:
            if len(data) == 2:
                data = [
                    data[0],
                    [[e.encode(encoding=None) for e in sent_eventualities] for sent_eventualities in data[1]]
                ]
                data = json.dumps(data).encode("utf-8")
            else:
                raise ValueError("Error: your message should be text or (parsed_results, para_eventualities).")
        request_id = self._send(ASERCmd.extract_relations, data)
        msg = self._recv(request_id)
        if not msg:
            return None

        ret_data = []
        for sent_relations in msg:
            rst = list()
            for r_encoded in sent_relations:
                relation = Relation().decode(r_encoded, encoding=None)
                rst.append(relation)
            ret_data.append(rst)
        return ret_data

    def extract_eventualities_and_relations(self, data):
        if isinstance(data, str):
            data = data.encode("utf-8")
        else:
            data = json.dumps(data).encode("utf-8")
        request_id = self._send(ASERCmd.extract_eventualities, data)
        msg = self._recv(request_id)
        if not msg:
            return None

        ret_eventualities, ret_relations = [], []
        for sent_eventualities, sent_relations in msg:
            rst_eventualities, rst_relations = list(), list()
            for e_encoded in sent_eventualities:
                eventuality = Eventuality().decode(e_encoded, encoding=None)
                rst_eventualities.append(eventuality)
            ret_eventualities.append(rst_eventualities)
            for r_encoded in sent_relations:
                relation = Relation().decode(r_encoded, encoding=None)
                rst_relations.append(relation)
            ret_relations.append(rst_relations)
        return ret_eventualities, ret_relations

    def conceptualize_eventuality(self, data):
        request_id = self._send(ASERCmd.conceptualize_eventuality, data.encode("utf-8"))
        msg = self._recv(request_id)
        if not msg:
            return None

        ret_data = list()
        for c_encoded, score in msg:
            concept = ASERConcept().decode(c_encoded, encoding=None)
            ret_data.append((concept, score))
        return ret_data

    def predict_eventuality_relation(self, eventuality1, eventuality2):
        if isinstance(eventuality1, str):
            hid = eventuality1
        else:
            hid = eventuality1.eid
        if isinstance(eventuality2, str):
            tid = eventuality2
        else:
            tid = eventuality2.eid
        rid = Relation.generate_rid(hid, tid).encode("utf-8")
        request_id = self._send(ASERCmd.exact_match_eventuality_relation, rid)
        msg = self._recv(request_id)
        if msg == ASERCmd.none:
            return None
        else:
            return Relation().decode(msg, encoding=None)

    def fetch_related_eventualities(self, data):
        if isinstance(data, str): # eid
            data = data.encode("utf-8")
        else:
            data = data.eid.encode("utf-8")
        request_id = self._send(ASERCmd.fetch_related_eventualities, data)
        msg = self._recv(request_id)
        return [
            (Eventuality().decode(e_encoded, encoding=None), Relation().decode(r_encoded, encoding=None)) for e_encoded, r_encoded in msg
        ]

    def predict_concept_relation(self, concept1, concept2):
        if isinstance(concept1, str):
            hid = concept1
        else:
            hid = concept1.cid
        if isinstance(concept2, str):
            tid = concept2
        else:
            tid = concept2.cid
        rid = Relation.generate_rid(hid, tid).encode("utf-8")
        request_id = self._send(ASERCmd.exact_match_concept_relation, rid)
        msg = self._recv(request_id)
        if msg == ASERCmd.none:
            return None
        else:
            return Relation().decode(msg, encoding=None)

    def fetch_related_concepts(self, data):
        if isinstance(data, str): # eid
            data = data.encode("utf-8")
        else:
            data = data.cid.encode("utf-8")
        request_id = self._send(ASERCmd.fetch_related_concepts, data)
        msg = self._recv(request_id)
        return [
            (ASERConcept().decode(c_encoded, encoding=None), Relation().decode(r_encoded, encoding=None)) for c_encoded, r_encoded in msg
        ]

