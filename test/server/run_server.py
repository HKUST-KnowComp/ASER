from aser.server import ASERServer
from aser.utils.config import get_server_args_parser

if __name__ == "__main__":
    parser = get_server_args_parser()
    args = parser.parse_args([
        "-n_workers", "2",
        "-n_concurrent_back_socks", "10",
        "-port", "8000",
        "-port_out", "8001",
        "-corenlp_path", "/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        "-base_corenlp_port", "9000",
        "-kg_dir", "/data/hjpan/ASER/core"
    ])
    server = ASERServer(args)

