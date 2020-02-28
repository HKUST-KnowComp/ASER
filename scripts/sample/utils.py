import json
from itertools import chain
from aser.eventuality import Eventuality
from aser.relation import Relation


def load_eventualities_from_json_file(json_file):
    with open(json_file, "r") as f:
        eventualities = json.load(f)
    for pattern, es in eventualities.items():
        for e in es:
            e["eventualities"] = [Eventuality().decode(x.encode("utf-8"), "utf-8") for x in e["eventualities"]]
    return eventualities

def load_relations_from_json_file(json_file):
    with open(json_file, "r") as f:
        relations = json.load(f)
    for sense, rs in relations.items():
        for r in rs:
            r["eventualities"] = [Eventuality().decode(x.encode("utf-8"), "utf-8") for x in r["eventualities"]]
            r["relations"] = [Relation().decode(x.encode("utf-8"), "utf-8") for x in r["relations"]]
    return relations
        
def print_extracted_results(extracted_results):
    extracted_eventualities, extracted_relations = extracted_results
    
    print("Eventualities:")
    if len(extracted_eventualities) > 0:
        if isinstance(extracted_eventualities[0], Eventuality):
            tmp_e = [(e.eid, e.words) for e in extracted_eventualities]
            for x in tmp_e:
                print(x)
        else:
            tmp_e = [(e.eid, e.words) for e in chain.from_iterable(extracted_eventualities)]
            for x in tmp_e:
                print(x)
        eid2event = dict(tmp_e)
    else:
        print()
        eid2event = dict()
    
    print("Relations:")
    if len(extracted_relations) > 0:
        if isinstance(extracted_relations[0], Relation):
            for x in [(r.rid, eid2event[r.hid], eid2event[r.tid], r.relations) for r in extracted_relations]:
                print(x)
        else:
            for x in [(r.rid, eid2event[r.hid], eid2event[r.tid], r.relations) for r in chain.from_iterable(extracted_relations)]:
                print(x)
    else:
        print()