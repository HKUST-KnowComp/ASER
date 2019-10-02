import json
import traceback
from pprint import pprint as print
from tqdm import tqdm
from aser.extract.event_extractor import EventualityExtractor

def equal_dependencies(dep1, dep2):
    d1 = set()
    for t in dep1:
        d1.add((tuple(t[0]), t[1], tuple(t[2])))
    d2 = set()
    for t in dep2:
        d2.add((tuple(t[0]), t[1], tuple(t[2])))
    dep1 = sorted(d1, key=lambda x: (x[0][0], x[2][0]))
    dep2 = sorted(d2, key=lambda x: (x[0][0], x[2][0]))
    dep1_str = json.dumps(dep1)
    dep2_str = json.dumps(dep2)
    return dep1_str == dep2_str


if __name__ == "__main__":
    e_extractor = EventualityExtractor(
        corenlp_path="/home/software/stanford-corenlp/stanford-corenlp-full-2018-02-27/",
        corenlp_port=11001)
    with open("test/vc/nyt_2007_06_sActivityNet.json") as f:
        test_data = json.load(f)

    is_passed = True
    for i, document in tqdm(enumerate(test_data)):
        if i < 385:
            continue
        try:
            text = document["doc"]

            pred_sentences = e_extractor.extract_eventualities(
                text, only_events=False, output_format="json")
            grt_eventualities = document["eventualities"]
            pred_eventualities = [e for sent in pred_sentences for e in sent["eventuality_list"]]
            if len(grt_eventualities) != len(pred_eventualities):
                print("DOCUMENT %d" % i)
                print(grt_eventualities)
                print(pred_eventualities)
                pred_sentences = e_extractor.extract_eventualities(
                    text, only_events=False, output_format="json")
                raise RuntimeError("Length is not equal")
            for grt_e, pred_e in zip(grt_eventualities, pred_eventualities):
                if not equal_dependencies(grt_e["parsed_relations"], pred_e["dependencies"]):
                    print("DOCUMENT %d" % i)
                    print(pred_e)
                    print(grt_e)
                    raise RuntimeError("Dependencies is ot equal")
                if not equal_dependencies(grt_e["skeleton_parsed_relations"], pred_e["skeleton_dependencies"]):
                    print("DOCUMENT %d" % i)
                    print(pred_e)
                    print(grt_e)
                    raise RuntimeError("Dependencies is ot equal")
        except:
            traceback.print_exc()
            is_passed = False
            break
    if is_passed:
        print("passed")
    e_extractor.close()

