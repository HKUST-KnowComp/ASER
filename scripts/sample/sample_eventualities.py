import numpy as np
import pickle
import json
import os
from collections import defaultdict
from aser.database.kg_connection import ASERKGConnection
from aser.extract.parsed_reader import ParsedReader
from aser.extract.aser_extractor import DiscourseASERExtractor2


if __name__ == "__main__":
    processed_path = r"D:\Data\ASER\data"
    db_path = r"D:\Data\ASER\database\core_0.3\KG.db"
    eid2sids_path = r"D:\Data\ASER\database\core_0.3\eid2sids.pkl"
    sampled_eventualities_path = r"D:\Data\ASER\database\core_0.3\eventualities.json"
    N = 10
    seed = 0

    np.random.seed(seed)

    kg_conn = ASERKGConnection(db_path, mode="memory")
    parsed_reader = ParsedReader()
    aser_extractor = DiscourseASERExtractor2()
    with open(eid2sids_path, "rb") as f:
        eid2sids = pickle.load(f)

    pattern2eids = defaultdict(list)
    for eid, eventuality in kg_conn.eid2eventuality_cache.items():
        pattern2eids[eventuality.pattern].append(eid)
    sampled_eventualities = dict()
    for pattern, eids in pattern2eids.items():
        eventualities = list()
        np.random.shuffle(eids)
        for i in range(min(len(eids), N)):
            eid = eids[i]
            sids = list(eid2sids[eid])
            np.random.shuffle(sids)
            sid = sids[0]
            sentence = parsed_reader.get_parsed_sentence_and_context(os.path.join(processed_path, sid))["sentence"]
            extracted_eventualities = aser_extractor.extract_eventualities_from_parsed_result(sentence)
            eventualities.append({
                "sid": sid,
                "sentence": sentence,
                "eid": eid,
                "eventualities": [e.encode(encoding="utf-8").decode("utf-8") for e in extracted_eventualities]
                })
        sampled_eventualities[pattern] = eventualities
        print(pattern, len(eventualities))
    
    with open(sampled_eventualities_path, "w") as f:
        json.dump(sampled_eventualities, f)
    kg_conn.close()
    parsed_reader.close()
    aser_extractor.close()