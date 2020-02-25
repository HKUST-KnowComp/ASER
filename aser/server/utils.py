import os
import socket
import uuid
import zmq


def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False


def sockets_ipc_bind(socket):
    tmp_dir = os.path.join("/tmp", str(uuid.uuid1())[:8])
    socket.bind('ipc://{}'.format(tmp_dir))
    return socket.getsockopt(zmq.LAST_ENDPOINT).decode('ascii')