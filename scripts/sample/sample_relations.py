import numpy as np
import pickle
import json
import os
from collections import defaultdict
from aser.database.kg_connection import ASERKGConnection
from aser.extract.parsed_reader import ParsedReader
from aser.extract.aser_extractor import DiscourseASERExtractor


if __name__ == "__main__":
    processed_path = r"D:\Data\ASER\data"
    db_path = r"D:\Data\ASER\database\core_0.3\KG.db"
    rid2sids_path = r"D:\Data\ASER\database\core_0.3\rid2sids.pkl"
    sampled_relations_path = r"D:\Data\ASER\database\core_0.3\relations.json"
    N = 10
    seed = 0

    np.random.seed(seed)

    kg_conn = ASERKGConnection(db_path, mode="memory")
    parsed_reader = ParsedReader()
    aser_extractor = DiscourseASERExtractor()
    with open(rid2sids_path, "rb") as f:
        rid2sids = pickle.load(f)

    relation2rids = defaultdict(list)
    for rid, relation in kg_conn.rid2relation_cache.items():
        for sense in relation.relations:
            relation2rids[sense].append(rid)
    sampled_relations = dict()
    for sense, rids in relation2rids.items():
        relations = list()
        np.random.shuffle(rids)
        for rid in rids:
            relation = kg_conn.get_exact_match_relation(rid)
            hid, tid = relation.hid, relation.tid
            sids = list(rid2sids[rid])
            np.random.shuffle(sids)
            for (sid1, sid2) in sids:
                sentence1 = parsed_reader.get_parsed_sentence_and_context(os.path.join(processed_path, sid1))["sentence"]
                if sid2 != sid1:
                    sentence2 = parsed_reader.get_parsed_sentence_and_context(os.path.join(processed_path, sid2))["sentence"]
                    extracted_eventualities, extracted_relations = aser_extractor.extract_from_parsed_result([sentence1, sentence2], in_order=False)
                else:
                    sentence2 = sentence1
                    extracted_eventualities, extracted_relations = aser_extractor.extract_from_parsed_result([sentence1], in_order=False)
                if len(extracted_relations) > 0 and not sense in set.union(*[set(r.relations.keys()) for r in extracted_relations]):
                    continue
                else:
                    relations.append({
                        "sids": (sid1, sid2),
                        "sentences": [sentence1] if sid1 == sid2 else [sentence1, sentence2],
                        "hid": hid,
                        "tid": tid,
                        "rid": rid,
                        "eventualities": [e.encode(encoding="utf-8").decode("utf-8") for e in extracted_eventualities],
                        "relations": [r.encode(encoding="utf-8").decode("utf-8") for r in extracted_relations]
                        })
                    break
            if len(relations) >= N:
                break
            
        sampled_relations[sense] = relations
        print(sense, len(relations))
    
    with open(sampled_relations_path, "w") as f:
        json.dump(sampled_relations, f)
    kg_conn.close()
    parsed_reader.close()
    aser_extractor.close()
