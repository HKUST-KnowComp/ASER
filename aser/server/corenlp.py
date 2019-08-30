#!/usr/bin/env python3
# Copyright 2019-present, HKUST.
# All rights reserved.
#
# Author: Haojie Pan
# Email:  myscarletpan@gmail.com

import time
from pycorenlp import StanfordCoreNLP
from aser.server.utils import is_port_occupied
import subprocess
import warnings


class StanfordCoreNLPServer(object):
    def __init__(self, **kwargs):
        self.corenlp_path = kwargs.get("corenlp_path", "./")
        self.port = kwargs.get("port", "9000")
        self.connect(self.port)
        pass

    def connect(self, port):
        try:
            nlp = StanfordCoreNLP("http://localhost:{}".format(port))
            nlp.annotate("hello world")
        except:
            assert not is_port_occupied(port=port), "Port {} is already occupied!".format(port)
            print("[CoreNLP] Start a new corenlp at port {}".format(port))
            corenlp_args = [
                "java",
                "-mx4g",
                "-cp",
                "{}/*".format(self.corenlp_path),
                "edu.stanford.nlp.pipeline.StanfordCoreNLPServer",
                "-port",
                str(port),
                "-timeout",
                "15000"
            ]
            subprocess.Popen(args=corenlp_args, stdin=None, stdout=None, stderr=None, close_fds=True)
            print("[CoreNLP] connect port {} succeed".format(port))
            time.sleep(2)
            # nlp = StanfordCoreNLP("http://localhost:{}".format(port))
            # nlp.annotate("hello world")


    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        try:
            StanfordCoreNLP("http://localhost:{}".format(self.port))
            cmd = 'wget "localhost:{}/shutdown?key=`cat /tmp/corenlp.shutdown`" -O -'.format(self.port)
            subprocess.run(cmd, shell=True)
        except:
            warnings.warn("[CoreNLP] corenlp at port is already closed.", RuntimeWarning)