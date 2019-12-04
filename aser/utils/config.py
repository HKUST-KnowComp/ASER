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


def get_pipe_args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("-n_extractors", type=int, default=5,
                        help="Number of ASER extractors")
    parser.add_argument("-n_workers", type=int, default=5,
                        help="Number of ASER workers, "
                             "same as num of corenlp workers")
    # Stanford Corenlp
    parser.add_argument("-corenlp_path", type=str, default="./",
                        help="StanfordCoreNLP path")
    parser.add_argument("-base_corenlp_port", type=int, default=9000,
                        help="Base port of corenlp"
                             "[base_corenlp_port, base_corenlp_port + n_workers - 1]"
                             "should be reserved")
    # Raw Data
    parser.add_argument("-raw_dir", type=str, default="./",
                        help="ASER raw data directory")
    # Processed Data
    parser.add_argument("-processed_dir", type=str, default="./",
                        help="ASER processed_dir data directory")              
    # KG
    parser.add_argument("-kg_dir", type=str, default="./",
                        help="ASER KG directory")
    parser.add_argument("-eventuality_frequency_lower_cnt_threshold", type=float, default=0.0,
                        help="eventualities whose frequencies are lower than this will be filtered")
    parser.add_argument("-eventuality_frequency_upper_percent_threshold", type=float, default=1.0,
                        help="eventualities whose frequency percents are higher than this will be filtered")
    parser.add_argument("-relation_frequency_lower_cnt_threshold", type=float, default=0.0,
                        help="relations whose frequencies are lower than this will be filtered")
    parser.add_argument("-relation_frequency_upper_percent_threshold", type=float, default=1.0,
                        help="relations whose frequency percents are higher than this will be filtered")
    # Log
    parser.add_argument("-log_path", type=str, default="./.tmp.log",
                        help="Logging path of pipe output")

    return parser