import os
import uuid
import shlex
import subprocess
import requests
from .utils import ANNOTATORS

props_file_name = f"corenlp_server-{uuid.uuid4().hex[:16]}.props"


class ShouldRetryException(Exception):
    """ Exception raised if the service should retry the request. """
    pass


def write_corenlp_props(annotators):
    props_dict = {
        'annotators': annotators,
        'timeout': 60000,
        'outputFormat': 'serialized',
        'serializer': 'edu.stanford.nlp.pipeline.ProtobufAnnotationSerializer'
    }
    """ Write a Stanford CoreNLP properties dict to a file """
    file_name = props_file_name
    with open(file_name, 'w') as props_file:
        for k, v in props_dict.items():
            if isinstance(v, list):
                writeable_v = ",".join(v)
            else:
                writeable_v = v
            props_file.write(f'{k} = {writeable_v}\n\n')
    return file_name


def start_service(start_cmd):
    stderr = open(os.devnull, 'w')
    print(f"Starting server with command: {' '.join(start_cmd)}")
    server = subprocess.Popen(start_cmd, stderr=stderr, stdout=stderr)
    return server


def is_alive(port):
    try:
        return requests.get(f"http://localhost:{port}/ping").ok
    except requests.exceptions.ConnectionError as e:
        raise ShouldRetryException(e)


def start_server(server_num, annotators, server_port=9101, corenlp_path=None):
    servers = []
    for port in range(server_num):
        port += server_port
        anno = ','.join(annotators)
        props_path = write_corenlp_props(annotators=anno)

        start_cmd = f"java -Djava.io.tmpdir={os.path.dirname(corenlp_path)}/.tmp -Xmx11G -cp {corenlp_path}/* " \
                    f"edu.stanford.nlp.pipeline.StanfordCoreNLPServer -port {port} " \
                    f"-timeout 60000 -threads 5 -maxCharLength 100000 " \
                    f"-quiet True -serverProperties {props_path} -preload {anno}"

        start_cmd = start_cmd and shlex.split(start_cmd)
        servers.append(start_service(start_cmd))
        while True:
            try:
                if is_alive(port):
                    break
            except ShouldRetryException:
                pass
    print('all server prepared..')
    while True:
        if input('kill all servers(q):') == 'q':
            break
    for s in servers:
        s.kill()


if __name__ == '__main__':
    corenlp_path = "/home/xliucr/stanford-corenlp-3.9.2"
    start_server(server_num=10, annotators=list(ANNOTATORS), corenlp_path=corenlp_path)
    # start_server(server_num=10, annotators=['tokenize', 'ssplit', 'parse'])
