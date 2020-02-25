import json
import multiprocessing
from multiprocessing import Process
import os
import random
import time
import traceback
import zmq
import zmq.decorators as zmqd
from aser.database.kg_connection import ASERKGConnection
from aser.server.utils import *
from aser.extract.aser_extractor import SeedRuleASERExtractor
from aser.utils.config import ASERCmd


class ASERServer(object):
    def __init__(self, opt):
        self.opt = opt
        self.port = opt.port
        self.n_concurrent_back_socks = opt.n_concurrent_back_socks
        self.n_workers = opt.n_workers
        self.aser_sink = None
        self.aser_db = None
        self.aser_workers = []

        self.run()

    def run(self):
        self._run()

    def close(self):
        for corenlp in self.corenlp_servers:
            corenlp.close()
        self.aser_sink.close()
        self.aser_db.close()
        for worker in self.aser_workers:
            worker.close()

    @zmqd.context()
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PUSH)
    def _run(self, ctx, client_msg_receiver, sink_addr_receiver, db_sender):
        total_st = time.time()

        client_msg_receiver.bind("tcp://*:%d" % self.port)

        sink_addr_receiver_addr = sockets_ipc_bind(sink_addr_receiver)
        self.aser_sink = ASERSink(self.opt, sink_addr_receiver_addr)
        self.aser_sink.start()
        sink_receiver_addr = sink_addr_receiver.recv().decode("utf-8")

        db_senders = []
        db_addr_list = []
        for _ in range(self.n_concurrent_back_socks):
            _socket = ctx.socket(zmq.PUSH)
            addr = sockets_ipc_bind(_socket)
            db_senders.append(_socket)
            db_addr_list.append(addr)

        self.aser_db = ASERDataBase(self.opt, db_addr_list, sink_receiver_addr)
        self.aser_db.start()

        worker_senders = []
        worker_addr_list = []
        for _ in range(self.n_concurrent_back_socks):
            _socket = ctx.socket(zmq.PUSH)
            addr = sockets_ipc_bind(_socket)
            worker_senders.append(_socket)
            worker_addr_list.append(addr)

        for i in range(self.n_workers):
            self.aser_workers.append(
                ASERWorker(self.opt, i, worker_addr_list, sink_receiver_addr)
            )
            self.aser_workers[i].start()

        print("Loading Server Finished in {:.4f} s".format(time.time() - total_st))
        worker_sender_id = -1
        db_sender_id = -1
        cnt = 0
        st = time.time()
        while True:
            try:
                client_msg = client_msg_receiver.recv_multipart()
                client_id, req_id, cmd, data = client_msg
                if cmd == ASERCmd.extract_events:
                    worker_sender_id, worker_sender = random.choice(
                        [(i, sender) for i, sender in enumerate(worker_senders)
                         if i != worker_sender_id])
                    worker_sender.send_multipart(client_msg)
                else:
                    db_sender_id, db_sender = random.choice(
                        [(i, sender) for i, sender in enumerate(db_senders)
                         if i != db_sender_id])
                    db_sender.send_multipart(client_msg)
                cnt += 1
                # print("sender speed: {:.4f} / call".format((time.time() - st) / cnt))
                print("Sender cnt {}".format(cnt))
            except Exception:
                print(traceback.format_exc())


class ASERDataBase(Process):
    def __init__(self, opt, db_sender_addr_list, sink_addr):
        super().__init__()
        self.db_sender_addr_list = db_sender_addr_list
        self.sink_addr = sink_addr
        print("Connect to the KG...")
        st = time.time()
        kg_dir = opt.kg_dir
        self.ASER_KG = ASERKGConnection(db_path=os.path.join(kg_dir, "KG.db"), mode="cache")
        print("Connect to the KG finished in {:.4f} s".format(time.time() - st))

    def run(self):
        self._run()

    def close(self):
        self.ASER_KG.close()
        self.terminate()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _run(self, ctx, sink):
        receiver_sockets = []
        poller = zmq.Poller()
        for db_sender_addr in self.db_sender_addr_list:
            _socket = ctx.socket(zmq.PULL)
            _socket.connect(db_sender_addr)
            receiver_sockets.append(_socket)
            poller.register(_socket)
        sink.connect(self.sink_addr)
        print("ASER DB started")

        cnt = 0
        st = time.time()
        while True:
            try:
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(receiver_sockets):
                    if sock in events:
                        client_id, req_id, cmd, data = sock.recv_multipart()
                        # print("DB received msg ({}, {}, {}, {})".format(
                        #     client_id.decode("utf-8"), req_id.decode("utf-8"),
                        #     cmd.decode("utf-8"), data.decode("utf-8")
                        # ))
                        if cmd == ASERCmd.exact_match_event:
                            ret_data = self.handle_exact_match_event(data)
                        elif cmd == ASERCmd.exact_match_relation:
                            ret_data = self.handle_exact_match_relation(data)
                        elif cmd == ASERCmd.fetch_related_events:
                            ret_data = self.handle_fetch_related_events(data)
                        else:
                            raise RuntimeError
                        sink.send_multipart([client_id, req_id, cmd, ret_data])
                        cnt += 1
                        print("DB cnt {}".format(cnt))
                # print("DB speed: {:.4f} / call".format((time.time() - st) / cnt))
            except Exception:
                print(traceback.format_exc())

    def handle_exact_match_event(self, data):
        eid = data.decode("utf-8")
        matched_event = self.ASER_KG.get_exact_match_eventuality(eid)
        if matched_event:
            ret_data = json.dumps(matched_event.encode(encoding=None)).encode("utf-8")
        else:
            ret_data = json.dumps(ASERCmd.none).encode(encoding="utf-8")
        return ret_data

    def handle_exact_match_relation(self, data):
        eid1, eid2 = data.decode("utf-8").split("$")
        matched_relation = self.ASER_KG.get_exact_match_relation([eid1, eid2])[0]
        print(matched_relation)
        if matched_relation:
            ret_data = json.dumps(matched_relation.encode(encoding=None)).encode("utf-8")
        else:
            ret_data = json.dumps(ASERCmd.none).encode(encoding="utf-8")
        return ret_data

    def handle_fetch_related_events(self, data):
        h_eid = data.decode("utf-8")
        related_events = self.ASER_KG.get_related_eventualities(h_eid)
        rst = [(event.encode(encoding=None), relation.encode(encoding=None))
               for event, relation in related_events]
        ret_data = json.dumps(rst).encode("utf-8")
        return ret_data


class ASERWorker(Process):
    def __init__(self, opt, id, worker_addr_list, sink_addr):
        super().__init__()
        self.worker_id = id
        self.worker_addr_list = worker_addr_list
        self.sink_addr = sink_addr
        self.eventuality_extractor = SeedRuleASERExtractor(
            corenlp_path = opt.corenlp_path,
            corenlp_port=opt.base_corenlp_port + id)
        self.is_ready = multiprocessing.Event()

    def run(self):
        self._run()

    def close(self):
        self.is_ready.clear()
        self.eventuality_extractor.close()
        self.terminate()
        self.join()

    @zmqd.context()
    @zmqd.socket(zmq.PUSH)
    def _run(self, ctx, sink):
        print("ASER Worker %d started" % self.worker_id)
        receiver_sockets = []
        poller = zmq.Poller()
        for worker_addr in self.worker_addr_list:
            _socket = ctx.socket(zmq.PULL)
            _socket.connect(worker_addr)
            receiver_sockets.append(_socket)
            poller.register(_socket)
        sink.connect(self.sink_addr)

        while True:
            try:
                events = dict(poller.poll())
                for sock_idx, sock in enumerate(receiver_sockets):
                    if sock in events:
                        client_id, req_id, cmd, data = sock.recv_multipart()
                        print("Worker {} received msg ({}, {}, {}, {})".format(
                            self.worker_id,
                            client_id.decode("utf-8"), req_id.decode("utf-8"),
                            cmd.decode("utf-8"), data.decode("utf-8")
                        ))
                        if cmd == ASERCmd.extract_events:
                            ret_data = self.handle_extract_events(data)
                            sink.send_multipart([client_id, req_id, cmd, ret_data])
                        else:
                            raise RuntimeError
            except Exception:
                print(traceback.format_exc())

    def handle_extract_events(self, data):
        sentence = data.decode("utf-8")
        eventualities_list = self.eventuality_extractor.extract_eventualities_from_text(sentence)
        print(eventualities_list)

        rst = [[e.encode(encoding=None) for e in eventualities] for eventualities in eventualities_list]
        ret_data = json.dumps(rst).encode("utf-8")
        return ret_data


class ASERSink(Process):
    def __init__(self, args, sink_addr_receiver_addr):
        super().__init__()
        self.port_out = args.port_out
        self.sink_addr_receiver_addr = sink_addr_receiver_addr

    def run(self):
        self._run()

    @zmqd.context()
    @zmqd.socket(zmq.PAIR)
    @zmqd.socket(zmq.PULL)
    @zmqd.socket(zmq.PUB)
    def _run(self, _, addr_sender, receiver, sender):
        addr_sender.connect(self.sink_addr_receiver_addr)
        receiver_addr = sockets_ipc_bind(receiver).encode("utf-8")
        addr_sender.send(receiver_addr)
        sender.bind("tcp://*:%d" % self.port_out)
        print("ASER Sink started")
        cnt = 0
        while True:
            try:
                msg = receiver.recv_multipart()
                sender.send_multipart(msg)
                cnt += 1
                print("Sink cnt {}".format(cnt))
            except Exception:
                print(traceback.format_exc())
