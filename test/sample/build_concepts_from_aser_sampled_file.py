import sys
import json
from tqdm import tqdm
from aser.eventuality import Eventuality
from aser.conceptualize.aser_conceptualizer import ASERSeedConceptualizer, ASERProbaseConceptualizer


if __name__ == "__main__":
    in_eventuality_file = sys.argv[1]
    out_concept_file = sys.argv[2]
    probase_path = sys.argv[3]

    with open(in_eventuality_file) as f:
        all_records = json.load(f)

    conceptualizer = ASERProbaseConceptualizer(probase_path=probase_path, probase_topk=5)

    for pattern, records in all_records.items():
        print("{}:\n".format(pattern))
        for record in tqdm(records):
            concept_list = list()
            for e_str in record["eventualities"]:
                e = Eventuality().decode(json.loads(e_str), encoding=None)
                concepts = conceptualizer.conceptualize(e)
                c_strs = [(str(c), score) for c, score in concepts]
                concept_list.append(c_strs)
            record["concepts"] = concept_list

    with open(out_concept_file, "w") as f:
        json.dump(all_records, f)