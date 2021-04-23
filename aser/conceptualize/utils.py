from collections import defaultdict
from copy import copy, deepcopy
from tqdm import tqdm
from ..eventuality import Eventuality
from ..relation import Relation

def conceptualize_eventualities(aser_concept_extractor, eventualities):
    cid2concept = dict()
    concept_instance_pairs = []
    cid_to_filter_score = dict()
    for eventuality in tqdm(eventualities):
        results = aser_concept_extractor.conceptualize(eventuality)
        for concept, score in results:
            if concept.cid not in cid2concept:
                cid2concept[concept.cid] = deepcopy(concept)
            concept = cid2concept[concept.cid]
            if (eventuality.eid, eventuality.pattern, score) not in concept.instances:
                concept.instances.append(((eventuality.eid, eventuality.pattern, score)))
                if concept.cid not in cid_to_filter_score:
                    cid_to_filter_score[concept.cid] = 0.0
                cid_to_filter_score[concept.cid] += score * eventuality.frequency
            concept_instance_pairs.append((concept, eventuality, score))
    return cid2concept, concept_instance_pairs, cid_to_filter_score


def build_concept_relations(aser_concept_conn, relations):
    rid2relation = dict()
    hid2related_events = defaultdict(list)
    for relation in tqdm(relations):
        hid2related_events[relation.hid].append((relation.tid, relation))

    for h_cid in tqdm(aser_concept_conn.cids):
        instances = aser_concept_conn.get_eventualities_given_concept(h_cid)
        for h_eid, pattern, instance_score in instances:
            # eid -> event -> related eids -> related events, relations -> related concepts, relations
            related_events = hid2related_events[h_eid]
            for t_eid, relation in related_events:
                concept_score_pairs = aser_concept_conn.get_concepts_given_eventuality(t_eid)
                for t_concept, score in concept_score_pairs:
                    t_cid = t_concept.cid
                    if h_cid == t_cid:
                        continue
                    rid = Relation.generate_rid(h_cid, t_cid)
                    if rid not in rid2relation:
                        rid2relation[rid] = Relation(h_cid, t_cid)
                    rid2relation[rid].update({k: v * instance_score * score for k, v in relation.relations.items()})
    return rid2relation