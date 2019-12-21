import socket
import os
from stanfordnlp.server import CoreNLPClient



def is_port_occupied(ip='127.0.0.1', port=80):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect((ip, int(port)))
        s.shutdown(2)
        return True
    except:
        return False

def get_corenlp_client(corenlp_path, corenlp_port):
    os.environ["CORENLP_HOME"] = corenlp_path

    assert not is_port_occupied(corenlp_port), "Port {} is occupied by other process".format(corenlp_port)
    corenlp_client = CoreNLPClient(
        annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'], timeout=60000,
        memory='5G', endpoint="http://localhost:%d" % corenlp_port,
        start_server=True, be_quiet=False)
    corenlp_client.annotate("hello world",
                            annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'depparse'],
                            output_format="json")
    return corenlp_client


if __name__ == "__main__":
    client = get_corenlp_client(
        corenlp_path="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        corenlp_port=11001)
    client.annotate("hello world")
    client.stop()