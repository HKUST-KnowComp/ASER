from aser.server import ASERServer

if __name__ == "__main__":
    server = ASERServer(
        port=8000,
        db_path="/home/hjpan/cache/KG.tiny.db",
        corenlp_path="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        base_corenlp_port=9000,
        corenlp_num=2)