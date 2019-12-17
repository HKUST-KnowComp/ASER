import json
import traceback
from pprint import pprint
from tqdm import tqdm
from aser.extract.eventuality_extractor import SeedRuleEventualityExtractor
from aser.extract.relation_extractor import SeedRuleRelationExtractor
from aser.extract.utils import sort_dependencies_position, parse_sentense_with_stanford


def equal_dependencies(dep1, dep2, dep_all=None):
    if dep_all:
        d_all = set()
        for t in dep_all:
            d_all.add((tuple(t[0]), t[1], tuple(t[2])))
    else:
        d_all = None
    d1 = set()
    for t in dep1:
        triple = (tuple(t[0]), t[1], tuple(t[2]))
        if d_all is None or triple in d_all:
            d1.add(triple)
    d1 = list(d1)
    d2 = set()
    for t in dep2:
        d2.add((tuple(t[0]), t[1], tuple(t[2])))
    d2 = list(d2)
    if len(dep2) != len(d2):
        return False
    d1, _, _ = sort_dependencies_position(d1, reset_position=True)
    d2, _, _ = sort_dependencies_position(d2, reset_position=True)
    dep1_str = json.dumps(d1)
    dep2_str = json.dumps(d2)
    return dep1_str == dep2_str


def equal_relations(rels1, rels2):
    rs1 = list(sorted(rels1))
    rs2 = list(sorted(rels2))
    if len(rs1) != len(rs2):
        return False
    for r1, r2 in zip(rs1, rs2):
        if r1 != r2:
            return False
    return True


if __name__ == "__main__":
    e_extractor = SeedRuleEventualityExtractor(
        corenlp_path="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        corenlp_port=13000)
    r_extractor = SeedRuleRelationExtractor()
    with open("/data/hjpan/nyt_2007_06_sActivityNet.json") as f:
        test_data = json.load(f)
    print(len(test_data))
    is_passed = True
    for i, document in tqdm(enumerate(test_data)):
        try:
            text = document["doc"]
            sent_parsed_results = parse_sentense_with_stanford(text, e_extractor.corenlp_client)
            pred_eventualities_list = [e_extractor.extract_from_parsed_result(t)
                                       for t in sent_parsed_results]
            grt_eventualities = document["eventualities"]
            grt_eventualities.sort(key=lambda x: " ".join([t[1] for t in x["tokens"]]))
            pred_eventualities = [e for elist in pred_eventualities_list
                                  for e in elist.eventualities]
            pred_eventualities.sort(key=lambda x: x.__repr__())
            if len(grt_eventualities) != len(pred_eventualities):
                pred_sentences = e_extractor.extract(text)
                print("DOCUMENT %d" % i)
                pprint(grt_eventualities)
                print(pred_eventualities)
                raise RuntimeError("Length is not equal")
            for grt_e, pred_e in zip(grt_eventualities, pred_eventualities):
                if not equal_dependencies(grt_e["parsed_relations"], pred_e.dependencies):
                    print("DOCUMENT %d" % i)
                    print(pred_e)
                    pprint(grt_e)
                    raise RuntimeError("Dependencies is not equal")

                # if pred_e.eid == '0f6e829070890d0ec2c6f3be7738f63ec9b234ae':
                #     print()
                if not equal_dependencies(
                        grt_e["skeleton_parsed_relations"], pred_e.skeleton_dependencies,
                        grt_e["parsed_relations"]):
                    print("DOCUMENT %d" % i)
                    print(pred_e)
                    pprint(grt_e)
                    raise RuntimeError("Skeleton Dependencies is not equal")
            grt_relations = list()
            for eid1, eid2, rel in document["seed_single_relations"]:
                grt_relations.append(rel)
            for eid1, eid2, rel in document["seed_double_relations"]:
                grt_relations.append(rel)

            pred_relations = r_extractor.extract(list(zip(sent_parsed_results, pred_eventualities_list)))
            pred_relations = [t[1] for t in pred_relations if t[1] != 'Co_Occurrence']
            if not equal_relations(grt_relations, pred_relations):
                print("DOCUMENT %d" % i)
                print(sorted(grt_relations))
                print(sorted(pred_relations))
                pred_relations = r_extractor.extract(list(zip(sent_parsed_results, pred_eventualities_list)))
                raise RuntimeError("Relations not equal")
        except:
            traceback.print_exc()
            is_passed = False
            break
    if is_passed:
        print("passed")
    e_extractor.close()

