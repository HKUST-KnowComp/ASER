from aser.server import ASERServer
from aser.utils.config import get_server_args_parser

if __name__ == "__main__":
    parser = get_server_args_parser()
    args = parser.parse_args([
        "-n_workers", "1",
        "-n_concurrent_back_socks", "10",
        "-port", "20097",
        "-port_out", "20098",
        "-corenlp_path", "/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        "-base_corenlp_port", "20099",
        "-kg_dir", "/data/hjpan/ASER/nyt_test_filtered"
    ])
    server = ASERServer(args)

