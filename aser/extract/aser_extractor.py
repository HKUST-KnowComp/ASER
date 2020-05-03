import bisect
try:
    import ujson as json
except:
    import json
from copy import copy, deepcopy
from itertools import chain
from aser.eventuality import Eventuality
from aser.relation import Relation, relation_senses
from aser.extract.eventuality_extractor import SeedRuleEventualityExtractor, DiscourseEventualityExtractor
from aser.extract.relation_extractor import SeedRuleRelationExtractor, DiscourseRelationExtractor
from aser.extract.discourse_parser import ConnectiveExtractor, ArgumentPositionClassifier, \
    SSArgumentExtractor, PSArgumentExtractor, ExplicitSenseClassifier, SyntaxTree
from aser.extract.utils import parse_sentense_with_stanford, get_corenlp_client, powerset, get_clauses
from aser.extract.utils import ANNOTATORS, EMPTY_SENT_PARSED_RESULT

class BaseASERExtractor(object):
    def __init__(self, **kw):
        self.corenlp_path = kw.get("corenlp_path", "")
        self.corenlp_port = kw.get("corenlp_port", 0)
        self.annotators = kw.get("annotators", list(ANNOTATORS))

        _, self.is_externel_corenlp = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
        
        self.eventuality_extractor = None
        self.relation_extractor = None

    def close(self):
        if not self.is_externel_corenlp:
            corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port)
            corenlp_client.stop()
        if self.eventuality_extractor:
            self.eventuality_extractor.close()
        if self.relation_extractor:
            self.relation_extractor.close()

    def __del__(self):
        self.close()
    
    def parse_text(self, text, annotators=None):
        if annotators is None:
            annotators = self.annotators

        corenlp_client, _ = get_corenlp_client(corenlp_path=self.corenlp_path, corenlp_port=self.corenlp_port, annotators=annotators)
        parsed_result = parse_sentense_with_stanford(text, corenlp_client, self.annotators)
        return parsed_result

    def extract_eventualities_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        """ This method extracts eventualities from parsed_result of one sentence.
        """
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities_from_parsed_result only supports Eventuality or json.")

        return self.eventuality_extractor.extract_from_parsed_result(parsed_result, 
            output_format=output_format, in_order=in_order, **kw)

    def extract_eventualities_from_text(self, text, output_format="Eventuality", in_order=True, annotators=None, **kw):
        """ This method extracts all eventualities for each sentence.
        """
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities_from_text only supports Eventuality or json.")

        parsed_result = self.parse_text(text, annotators=annotators)
        return self.extract_eventualities_from_parsed_result(parsed_result, 
            output_format=output_format, in_order=in_order, **kw)

    def extract_relations_from_parsed_result(self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw):
        """ This method extracts relations among extracted eventualities.
        """
        if output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_relations_from_parsed_result only supports Relation or triple.")
            
        return self.relation_extractor.extract_from_parsed_result(parsed_result, para_eventualities, 
            output_format=output_format, in_order=in_order, **kw)

    def extract_relations_from_text(self, text, output_format="Relation", in_order=True, annotators=None, **kw):
        """ This method extracts relations from parsed_result of one paragraph.
        """
        if output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_relations_from_text only supports Relation or triple.")

        parsed_result = self.parse_text(text, annotators=annotators)
        para_eventualities = self.extract_eventualities_from_parsed_result(parsed_result)
        return self.extract_relations_from_parsed_result(parsed_result, para_eventualities, 
            output_format=output_format, in_order=in_order, **kw)

    def extract_from_parsed_result(self, parsed_result, eventuality_output_format="Eventuality", relation_output_format="Relation", in_order=True, **kw):
        """ This method extracts eventualities and relations from parsed_result of one paragraph.
        """
        if eventuality_output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
        if relation_output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_relations only supports Relation or triple.")
        
        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False

        para_eventualities = self.extract_eventualities_from_parsed_result(parsed_result, 
            output_format="Eventuality", in_order=True, **kw)
        para_relations = self.extract_relations_from_parsed_result(parsed_result, para_eventualities, 
            output_format="Relation", in_order=True, **kw)

        if in_order:
            if eventuality_output_format == "json":
                para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
                    for sent_eventualities in para_eventualities]
            if relation_output_format == "triple":
                relations = [list(chain.from_iterable([relation.to_triple() for relation in sent_relations])) \
                    for sent_relations in para_relations]
            if is_single_sent:
                return para_eventualities[0], para_relations[0]
            else:
                return para_eventualities, para_relations
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if eventuality_output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif eventuality_output_format == "json":
                eventualities = sorted([eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()], key=lambda e: e["eid"])
            
            rid2relation = dict()
            for relation in chain.from_iterable(para_relations):
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update(relation)
            if relation_output_format == "Relation":
                relations = sorted(rid2relation.values(), key=lambda r: r.rid)
            elif relation_output_format == "triple":
                relations = sorted(chain.from_iterable([relation.to_triples() for relation in rid2relation.values()]))
            return eventualities, relations

    def extract_from_text(self, text, eventuality_output_format="Eventuality", relation_output_format="Relation", in_order=True, annotators=None, **kw):
        """ This method extracts eventualities and relations for each sentence.
        """
        if eventuality_output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
        if relation_output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_relations only supports Relation or triple.")

        parsed_result = self.parse_text(text, annotators=annotators)
        return self.extract_from_parsed_result(parsed_result, 
            eventuality_output_format=eventuality_output_format, relation_output_format=relation_output_format, in_order=in_order, **kw)


class SeedRuleASERExtractor(BaseASERExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)
        from aser.extract.rule import CLAUSE_WORDS
        self.eventuality_extractor = SeedRuleEventualityExtractor(skip_words=CLAUSE_WORDS)
        self.relation_extractor = SeedRuleRelationExtractor(**kw)

class DiscourseASERExtractor1(BaseASERExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.eventuality_extractor = SeedRuleEventualityExtractor(**kw)
        self.relation_extractor = DiscourseRelationExtractor(**kw)

class DiscourseASERExtractor2(BaseASERExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.eventuality_extractor = DiscourseEventualityExtractor(**kw)
        self.relation_extractor = DiscourseRelationExtractor(**kw)
    
    def extract_from_parsed_result(self, parsed_result, eventuality_output_format="Eventuality", relation_output_format="Relation", in_order=True, **kw):
        if "syntax_tree_cache" not in kw:
            kw["syntax_tree_cache"] = dict()
        return super().extract_from_parsed_result(parsed_result, 
            eventuality_output_format=eventuality_output_format, relation_output_format=relation_output_format, in_order=in_order, **kw)

class DiscourseASERExtractor3(BaseASERExtractor):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.eventuality_extractor = SeedRuleEventualityExtractor(**kw)
        self.conn_extractor = ConnectiveExtractor(**kw)
        self.argpos_classifier = ArgumentPositionClassifier(**kw)
        self.ss_extractor = SSArgumentExtractor(**kw)
        self.ps_extractor = PSArgumentExtractor(**kw)
        self.explicit_classifier = ExplicitSenseClassifier(**kw)

    def _extract_eventualities_from_clause(self, sent_parsed_result, clause):
        len_clause = len(clause)
        idx_mapping = {j: i for i, j in enumerate(clause)}
        indices_set = set(clause)
        clause_parsed_result = {
            "text": "",
            "dependencies": [(idx_mapping[dep[0]], dep[1], idx_mapping[dep[2]]) for dep in sent_parsed_result["dependencies"] \
                if dep[0] in indices_set and dep[2] in indices_set],
            "tokens": [sent_parsed_result["tokens"][idx] for idx in clause],
            "pos_tags": [sent_parsed_result["pos_tags"][idx] for idx in clause],
            "lemmas": [sent_parsed_result["lemmas"][idx] for idx in clause]}
        if "ners" in sent_parsed_result:
            clause_parsed_result["ners"] = [sent_parsed_result["ners"][idx] for idx in clause]
        if "mentions" in sent_parsed_result:
            clause_parsed_result["mentions"] = list()
            for mention in sent_parsed_result["mentions"]:
                start_idx = bisect.bisect_left(clause, mention["start"])
                if not (start_idx < len_clause and clause[start_idx] == mention["start"]):
                    continue
                end_idx = bisect.bisect_left(clause, mention["end"]-1)
                if not (end_idx < len_clause and clause[end_idx] == mention["end"]-1):
                    continue
                mention = copy(mention)
                mention["start"] = start_idx
                mention["end"] = end_idx+1
                clause_parsed_result["mentions"].append(mention)
        eventualities = self.eventuality_extractor.extract_from_parsed_result(
            clause_parsed_result, output_format="Eventuality", in_order=True)
        for eventuality in eventualities:
            for k, v in eventuality.raw_sent_mapping.items():
                eventuality.raw_sent_mapping[k] = clause[v]
            eventuality.eid = Eventuality.generate_eid(eventuality)
        return eventualities

    def _append_new_eventuaities_to_list(self, existed_eventualities, new_eventualities):
        len_existed_eventualities = len(existed_eventualities)
        for new_e in new_eventualities:
            is_existed = False
            for old_idx in range(len_existed_eventualities):
                old_e = existed_eventualities[old_idx]
                if old_e.eid == new_e.eid and old_e.raw_sent_mapping == new_e.raw_sent_mapping:
                    is_existed = True
                    break
            if not is_existed:
                existed_eventualities.append(new_e)

    def extract_eventualities_from_parsed_result(self, parsed_result, output_format="Eventuality", in_order=True, **kw):
        if output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Eventuality or json.")
        
        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False

        syntax_tree_cache = kw.get("syntax_tree_cache", dict())
        
        para_eventualities = [list() for _ in range(len(parsed_result))]
        para_clauses = self._extract_clauses(parsed_result, syntax_tree_cache)
        for sent_parsed_result, sent_clauses, sent_eventualities in zip(parsed_result, para_clauses, para_eventualities):
            for clause in sent_clauses:
                sent_eventualities.extend(self._extract_eventualities_from_clause(sent_parsed_result, clause))
        
        if in_order:
            if output_format == "json":
                para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
                    for sent_eventualities in para_eventualities]
            if is_single_sent:
                return para_eventualities[0]
            else:
                return para_eventualities
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif output_format == "json":
                eventualities = sorted([eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()], key=lambda e: e["eid"])
            return eventualities
    
    def extract_relations_from_parsed_result(self, parsed_result, para_eventualities, output_format="Relation", in_order=True, **kw):
        if output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_relations_from_parsed_result only supports Relation or triple.")

        len_sentences = len(parsed_result)
        if len_sentences == 0:
            if in_order:
                return [list()]
            else:
                return list()

        similarity = kw.get("similarity", "simpson").lower()
        threshold = kw.get("threshold", 0.8)
        if threshold < 0.0 or threshold > 1.0:
            raise ValueError("Error: threshold should be between 0.0 and 1.0.")
        if similarity == "simpson":
            similarity_func = self._match_argument_eventuality_by_Simpson
        elif similarity == "jaccard":
            similarity_func = self._match_argument_eventuality_by_Jaccard
        elif similarity == "discourse":
            similarity_func = self._match_argument_eventuality_by_dependencies
        else:
            raise NotImplementedError("Error: extract_from_parsed_result only supports Simpson or Jaccard.")

        syntax_tree_cache = kw.get("syntax_tree_cache", dict())

        para_relations = [list() for _ in range(2*len_sentences-1)]

        # replace sentences that contains no eventuality with empty sentences
        filtered_parsed_result = list()
        for sent_idx, (sent_parsed_result, sent_eventualities) in enumerate(zip(parsed_result, para_eventualities)):
            if len(sent_eventualities) > 0:
                relations_in_sent = para_relations[sent_idx]
                for e1_idx in range(len(sent_eventualities)-1):
                    heid = sent_eventualities[e1_idx].eid
                    for e2_idx in range(e1_idx+1, len(sent_eventualities)):
                        teid = sent_eventualities[e2_idx].eid
                        relations_in_sent.append(Relation(heid, teid, ["Co_Occurrence"]))
                filtered_parsed_result.append(sent_parsed_result)
            else:
                filtered_parsed_result.append(EMPTY_SENT_PARSED_RESULT) # empty sentence
                # filtered_parsed_result.append(sent_parsed_result)

        connectives = self.conn_extractor.extract(filtered_parsed_result, syntax_tree_cache)
        SS_connectives, PS_connectives = self.argpos_classifier.classify(filtered_parsed_result, connectives, syntax_tree_cache)
        SS_connectives = self.ss_extractor.extract(filtered_parsed_result, SS_connectives, syntax_tree_cache)
        PS_connectives = self.ps_extractor.extract(filtered_parsed_result, PS_connectives, syntax_tree_cache)
        connectives = self.explicit_classifier.classify(filtered_parsed_result, SS_connectives+PS_connectives, syntax_tree_cache)
        connectives.sort(key=lambda x: (x["sent_idx"], x["indices"][0] if len(x["indices"]) > 0 else -1))

        for connective in connectives:
            conn_indices = connective.get("indices", None)
            arg1 = connective.get("arg1", None)
            arg2 = connective.get("arg2", None)
            sense = connective.get("sense", None)
            if conn_indices and arg1 and arg2 and (sense and sense != "None"):
                arg1_sent_idx = arg1["sent_idx"]
                arg2_sent_idx = arg2["sent_idx"]
                relation_list_idx = arg1_sent_idx if arg1_sent_idx == arg2_sent_idx else arg1_sent_idx + len_sentences
                relations = para_relations[relation_list_idx]
                sent_parsed_result1, sent_eventualities1 = parsed_result[arg1_sent_idx], para_eventualities[arg1_sent_idx]
                sent_parsed_result2, sent_eventualities2 = parsed_result[arg2_sent_idx], para_eventualities[arg2_sent_idx]
                arg1_eventualities = [e for e in sent_eventualities1 if \
                    similarity_func(sent_parsed_result1, arg1, e, threshold=threshold, conn_indices=conn_indices)]
                arg2_eventualities = [e for e in sent_eventualities2 if \
                    similarity_func(sent_parsed_result2, arg2, e, threshold=threshold, conn_indices=conn_indices)]
                cnt = 0.0
                if len(arg1_eventualities) > 0 and len(arg2_eventualities) > 0:
                    cnt = 1.0 / (len(arg1_eventualities) * len(arg2_eventualities))
                for e1 in arg1_eventualities:
                    heid = e1.eid
                    for e2 in arg2_eventualities:
                        teid = e2.eid
                        existed_relation = False
                        for relation in relations:
                            if relation.hid == heid and relation.tid == teid:
                                relation.update({sense: cnt})
                                existed_relation = True
                                break
                        if not existed_relation:
                            relations.append(Relation(heid, teid, {sense: cnt}))

        if in_order:
            if output_format == "Relation":
                return para_relations
            elif output_format == "triple":
                return [sorted(chain.from_iterable([r.to_triples() for r in relations])) \
                    for relations in para_relations]
        else:
            if output_format == "Relation":
                rid2relation = dict()
                for relation in chain(*para_relations):
                    if relation.rid not in rid2relation:
                        rid2relation[relation.rid] = deeocopy(relation)
                    else:
                        rid2relation[relation.rid].update(relation)
                return sorted(rid2relation.values(), key=lambda r: r.rid)
            if output_format == "triple":
                return sorted([r.to_triples() for relations in para_relations for r in relations])

    def extract_from_parsed_result(self, parsed_result, eventuality_output_format="Eventuality", relation_output_format="Relation", in_order=True, **kw):
        if eventuality_output_format not in ["Eventuality", "json"]:
            raise NotImplementedError("Error: extract_eventualities only supports Eventuality or json.")
        if relation_output_format not in ["Relation", "triple"]:
            raise NotImplementedError("Error: extract_relations only supports Relation or triple.")
        
        if not isinstance(parsed_result, (list, tuple, dict)):
            raise NotImplementedError
        if isinstance(parsed_result, dict):
            is_single_sent = True
            parsed_result = [parsed_result]
        else:
            is_single_sent = False
        
        syntax_tree_cache = kw.get("syntax_tree_cache", dict())

        len_sentences = len(parsed_result)
        para_eventualities = [list() for _ in range(len_sentences)]
        para_relations = [list() for _ in range(2*len_sentences-1)]

        connectives = self.conn_extractor.extract(parsed_result, syntax_tree_cache)
        SS_connectives, PS_connectives = self.argpos_classifier.classify(parsed_result, connectives, syntax_tree_cache)
        SS_connectives = self.ss_extractor.extract(parsed_result, SS_connectives, syntax_tree_cache)
        PS_connectives = self.ps_extractor.extract(parsed_result, PS_connectives, syntax_tree_cache)
        connectives = self.explicit_classifier.classify(parsed_result, SS_connectives+PS_connectives, syntax_tree_cache)
        connectives.sort(key=lambda x: (x["sent_idx"], x["indices"][0] if len(x["indices"]) > 0 else -1))

        for connective in connectives:
            conn_indices = connective.get("indices", None)
            arg1 = connective.get("arg1", None)
            arg2 = connective.get("arg2", None)
            sense = connective.get("sense", None)
            if conn_indices and arg1 and arg2:
                arg1_sent_idx = arg1["sent_idx"]
                arg2_sent_idx = arg2["sent_idx"]
                senses = []
                if arg1_sent_idx == arg2_sent_idx:
                    senses.append("Co_Occurrence")
                if sense and sense != "None":
                    senses.append(sense)
                if len(senses) == 0:
                    continue
                relation_list_idx = arg1_sent_idx if arg1_sent_idx == arg2_sent_idx else arg1_sent_idx + len_sentences
                relations = para_relations[relation_list_idx]
                sent_parsed_result1, sent_eventualities1 = parsed_result[arg1_sent_idx], para_eventualities[arg1_sent_idx]
                sent_parsed_result2, sent_eventualities2 = parsed_result[arg2_sent_idx], para_eventualities[arg2_sent_idx]
                arg1_eventualities = self._extract_eventualities_from_clause(sent_parsed_result1, arg1["indices"])
                arg2_eventualities = self._extract_eventualities_from_clause(sent_parsed_result2, arg2["indices"])
                self._append_new_eventuaities_to_list(sent_eventualities1, arg1_eventualities)
                self._append_new_eventuaities_to_list(sent_eventualities2, arg2_eventualities)

                cnt = 0.0
                if len(arg1_eventualities) > 0 and len(arg2_eventualities) > 0:
                    cnt = 1.0 / (len(arg1_eventualities) * len(arg2_eventualities))
                for e1 in arg1_eventualities:
                    heid = e1.eid
                    for e2 in arg2_eventualities:
                        teid = e2.eid
                        is_existed = False
                        for relation in relations:
                            if relation.hid == heid and relation.tid == teid:
                                relation.update({sense: cnt for sense in senses})
                                is_existed = True
                                break
                        if not is_existed:
                            relations.append(Relation(heid, teid, {sense: cnt for sense in senses}))

        if in_order:
            if eventuality_output_format == "json":
                para_eventualities = [[eventuality.encode(encoding=None) for eventuality in sent_eventualities] \
                    for sent_eventualities in para_eventualities]
            if relation_output_format == "triple":
                relations = [list(chain.from_iterable([relation.to_triple() for relation in sent_relations])) \
                    for sent_relations in para_relations]
            if is_single_sent:
                return para_eventualities[0], para_relations[0]
            else:
                return para_eventualities, para_relations
        else:
            eid2eventuality = dict()
            for eventuality in chain.from_iterable(para_eventualities):
                eid = eventuality.eid
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = deepcopy(eventuality)
                else:
                    eid2eventuality[eid].update(eventuality)
            if eventuality_output_format == "Eventuality":
                eventualities = sorted(eid2eventuality.values(), key=lambda e: e.eid)
            elif eventuality_output_format == "json":
                eventualities = sorted([eventuality.encode(encoding=None) for eventuality in eid2eventuality.values()], key=lambda e: e["eid"])
            
            rid2relation = dict()
            for relation in chain.from_iterable(para_relations):
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update(relation)
            if relation_output_format == "Relation":
                relations = sorted(rid2relation.values(), key=lambda r: r.rid)
            elif relation_output_format == "triple":
                relations = sorted(chain.from_iterable([relation.to_triples() for relation in rid2relation.values()]))
            return eventualities, relations

    def _extract_clauses(self, parsed_result, syntax_tree_cache):
        para_arguments = [set() for _ in range(len(parsed_result))]
        connectives = self.conn_extractor.extract(parsed_result, syntax_tree_cache)
        para_connectives = [set() for _ in range(len(parsed_result))]
        for connective in connectives:
            sent_idx, indices = connective["sent_idx"], tuple(connective["indices"])
            para_connectives[sent_idx].add(indices)
        for sent_idx, sent_parsed_result in enumerate(parsed_result):
            sent_connectives = para_connectives[sent_idx]
            sent_arguments = para_arguments[sent_idx]

            if sent_idx in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx]
            else:
                syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])
            
            # more but slower
            # for indices in powerset(sent_connectives):
            #     indices = set(chain.from_iterable(indices))
            #     sent_arguments.update(get_clauses(sent_parsed_result, syntax_tree, index_seps=indices))
            sent_arguments.update(get_clauses(sent_parsed_result, syntax_tree, index_seps=set(chain.from_iterable(sent_connectives))))
        return para_arguments