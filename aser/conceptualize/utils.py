from collections import defaultdict
from copy import copy, deepcopy
from tqdm import tqdm
from ..eventuality import Eventuality
from ..relation import Relation


def conceptualize_eventualities(aser_conceptualizer, eventualities):
    """ Conceptualize eventualities by an ASER conceptualizer

    :param aser_conceptualizer: an ASER conceptualizer
    :type aser_conceptualizer: aser.conceptualize.aser_conceptualizer.BaseASERConceptualizer
    :param eventualities: a list of eventualities
    :type eventualities: List[aser.event.Eventuality]
    :return: a dictionary from cid to concept, a list of concept-instance pairs, a dictionary from cid to weights
    :rtype: Dict[str, aser.concept.ASERConcept], List[aser.concept.ASERConcept, aser.eventuality.Eventuality, float], Dict[str, float]
    """

    cid2concept = dict()
    concept_instance_pairs = []
    cid2score = dict()
    for eventuality in tqdm(eventualities):
        results = aser_conceptualizer.conceptualize(eventuality)
        for concept, score in results:
            if concept.cid not in cid2concept:
                cid2concept[concept.cid] = deepcopy(concept)
            concept = cid2concept[concept.cid]
            if (eventuality.eid, eventuality.pattern, score) not in concept.instances:
                concept.instances.append(((eventuality.eid, eventuality.pattern, score)))
                if concept.cid not in cid2score:
                    cid2score[concept.cid] = 0.0
                cid2score[concept.cid] += score * eventuality.frequency
            concept_instance_pairs.append((concept, eventuality, score))
    return cid2concept, concept_instance_pairs, cid2score


def build_concept_relations(concept_conn, relations):
    """ Build relations between conceptualized eventualities from the given relations between eventualities

    :param concept_conn: ASER concept KG connection
    :type concept_conn: aser.database.kg_connection.ASERConceptConnection
    :param relations: relations between eventualities
    :type relations: List[aser.relation.Relations]
    :return: a dictionary from rid to relations between conceptualized eventualities
    :rtype: Dict[str, aser.relation.Relation]
    """

    rid2relation = dict()
    hid2related_events = defaultdict(list)
    for relation in tqdm(relations):
        hid2related_events[relation.hid].append((relation.tid, relation))

    for h_cid in tqdm(concept_conn.cids):
        instances = concept_conn.get_eventualities_given_concept(h_cid)
        for h_eid, pattern, instance_score in instances:
            # eid -> event -> related eids -> related events, relations -> related concepts, relations
            related_events = hid2related_events[h_eid]
            for t_eid, relation in related_events:
                concept_score_pairs = concept_conn.get_concepts_given_eventuality(t_eid)
                for t_concept, score in concept_score_pairs:
                    t_cid = t_concept.cid
                    if h_cid == t_cid:
                        continue
                    rid = Relation.generate_rid(h_cid, t_cid)
                    if rid not in rid2relation:
                        rid2relation[rid] = Relation(h_cid, t_cid)
                    rid2relation[rid].update({k: v * instance_score * score for k, v in relation.relations.items()})
    return rid2relation
