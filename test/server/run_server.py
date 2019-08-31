from aser.server import ASERServer

if __name__ == "__main__":
    server = ASERServer(
        port=8000,
        db_dir="/home/hjpan/cache/core",
        corenlp_path="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        base_corenlp_port=9000,
        corenlp_num=1)