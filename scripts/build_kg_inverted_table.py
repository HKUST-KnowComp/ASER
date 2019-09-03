import json
import sys
import time
from aser.database.db_API import KG_Connection
from aser.database._kg_connection import relation_senses
from tqdm import tqdm

if __name__ == "__main__":
    st = time.time()
    print("Loading db...")
    kg_db = KG_Connection(db_path=sys.argv[1])
    print("Loading db finished in %.2f s" % (time.time() - st))

    tmp = kg_db._conn._conn.execute("SELECT * FROM Relations")
    event_inverted_table = {}
    for row in tqdm(tmp):
        e1_id = row[1]
        e2_id = row[2]
        if e1_id not in event_inverted_table:
            event_inverted_table[e1_id] = {}
        for rel, cnt in zip(relation_senses, row[3:]):
            if cnt > 0:
                if rel not in event_inverted_table[e1_id]:
                    event_inverted_table[e1_id][rel] = []
                event_inverted_table[e1_id][rel].append(e2_id)
    with open(sys.argv[2], "w") as f:
        json.dump(event_inverted_table, f)