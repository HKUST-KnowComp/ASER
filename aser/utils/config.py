import argparse

class ASERCmd:
    extract_events = b"__EXTRACT_EVENTS__"
    exact_match_event = b"__EXACT_MATCH_EVENT__"
    exact_match_relation = b"__EXACT_MATCH_RELATION__"
    fetch_related_events = b"__FETCH_RELATED_EVENTS__"


def get_server_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n_workers", type=int, default=5,
                        help="Number of ASER workers, "
                             "same as num of corenlp workers")
    parser.add_argument("-n_concurrent_back_socks", type=int, default=10,
                        help="Number of concurrent workers sockets")
    parser.add_argument("-port", type=int, default=8000,
                        help="server port for receiving msg from client")
    parser.add_argument("-port_out", type=int, default=8001,
                        help="client port for receving return data from server")

    # Stanford Corenlp
    parser.add_argument("-corenlp_path", type=str, default="./",
                        help="StanfordCoreNLP path")
    parser.add_argument("-base_corenlp_port", type=int, default=9000,
                        help="Base port of corenlp"
                             "[base_corenlp_port, base_corenlp_port + n_workers - 1]"
                             "should be reserved")

    # KG
    parser.add_argument("-kg_dir", type=str, default="./",
                        help="ASER KG directory")

    # I/O
    parser.add_argument("-log_path", type=str, default="./.tmp.log",
                        help="Logging path of server output")

    return parser