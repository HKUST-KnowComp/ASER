from aser.server import ASERServer
from aser.utils.config import get_server_args_parser

if __name__ == "__main__":
    parser = get_server_args_parser()
    args = parser.parse_args([
        "-n_workers", "2",
        "-n_concurrent_back_socks", "10",
        "-port", "12000",
        "-port_out", "12001",
        "-corenlp_path", "/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        "-base_corenlp_port", "11000",
        "-kg_dir", "/data/hjpan/ASER/tiny"
    ])
    server = ASERServer(args)

