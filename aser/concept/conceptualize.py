from .concept_extractor import ASERConceptExtractor
from .concept_connection import ASERConceptConnection


class ASERConceptAPI(object):
    def __init__(self, concept_db, aser_concept_extractor , aser_kg_conn):
        self.aser_concept_extractor = aser_concept_extractor
        self.aser_concept_conn = concept_db
        self.aser_kg_conn = aser_kg_conn

    def conceptualize(self, eventuality):
        concepts = self.aser_concept_conn.get_concepts_given_eventuality(eventuality)
        if not concepts:
            tmp_concepts = self.aser_concept_extractor.conceptualize(eventuality)
            concepts = list()
            for i, (concept, prob) in enumerate(tmp_concepts):
                tmp = self.aser_concept_conn.get_exact_match_concept(concept.cid)
                if tmp:
                    concepts.append((tmp, prob))
                else:
                    concepts.append((concept, prob))
        return concepts

    def instantiate(self, concept):
        tmp =  self.aser_concept_conn.get_exact_match_concept(concept.cid)
        concept = tmp if tmp else concept
        return concept.instantiate(self.aser_kg_conn)

    def get_related_concepts(self, concept):
        return self.aser_concept_conn.get_related_concepts(concept)