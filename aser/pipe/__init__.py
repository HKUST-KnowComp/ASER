import json
import os
import random
import time
import pickle
import multiprocessing
from tqdm import tqdm
from copy import copy, deepcopy
from functools import partial
from collections import Counter, defaultdict
from aser.extract.utils import get_corenlp_client
from aser.extract.sentence_parser import SentenceParser
from aser.extract.eventuality_extractor import EventualityExtractor
from aser.extract.relation_extractor import SeedRuleRelationExtractor, DiscourseRelationExtractor
from aser.extract.parsed_reader import ParsedReader
from aser.extract.utils import iter_files, parse_sentense_with_stanford
from aser.utils.logging import init_logger, close_logger
from aser.relation import Relation
from aser.database.kg_connection import ASERKGConnection

def run_file(raw_path=None, processed_path=None, prefix_to_be_removed="",
    sentence_parser=None, parsed_reader=None, eventuality_extractor=None, relation_extractor=None):

    empty_sent_parsed_result = {
        'text': '.',
        'dependencies': [],
        'parse': '(ROOT\r\n  (NP (. .)))',
        'tokens': ['.'],
        'lemmas': ['.'],
        'pos_tags': ['.']}

    # process raw data or load processed data
    if os.path.exists(processed_path):
        processed_data = load_processed_data(processed_path, parsed_reader)
    elif os.path.exists(raw_path):
        processed_data = process_raw_file(raw_path, processed_path, sentence_parser)
    else:
        raise ValueError("Error: at least one of raw_path and processed_path should not be None.")
    
    # remove prefix of sids
    document = list()
    for paragraph in processed_data:
        for sentence in paragraph:
            sentence["doc"] = os.path.splitext(os.path.basename(processed_path))[0]
            sentence["sid"] = sentence["sid"].replace(prefix_to_be_removed, "", 1)
            document.append(sentence)
        # document.append(empty_sent_parsed_result)

    # extract eventualities from processed data
    eventuality_lists = extract_eventualities(document, eventuality_extractor)
    eid2sids = defaultdict(list)
    eid2eventuality = dict()
    for sentence, eventuality_list in zip(document, eventuality_lists):
        for eventuality in eventuality_list:
            eid2sids[eventuality.eid].append(sentence["sid"])
            if eventuality.eid not in eid2eventuality:
                eid2eventuality[eventuality.eid] = copy(eventuality)
            else:
                eid2eventuality[eventuality.eid].update_frequency(eventuality)

    # extract relations from eventualities
    relation_lists = extract_relations(document, eventuality_lists, relation_extractor)
    rid2sids = defaultdict(list)
    rid2relation = dict()
    len_doc = len(document)

    # SS
    for idx in range(len_doc):
        relation_list = relation_lists[idx]
        for relation in relation_list:
            if sum(relation.relations.values()) > 0:
                rid2sids[relation.rid].append((document[idx]["sid"], document[idx]["sid"]))
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update_relations(relation.relations)
    # PS
    for idx in range(len_doc-1):
        relation_list = relation_lists[len_doc+idx]
        for relation in relation_list:
            if sum(relation.relations.values()) > 0:
                rid2sids[relation.rid].append((document[idx]["sid"], document[idx+1]["sid"]))
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update_relations(relation.relations)
    
    return eid2sids, rid2sids, eid2eventuality, rid2relation

def process_raw_file(raw_path, processed_path, sentence_parser):
    return sentence_parser.parse_raw_file(raw_path, processed_path, max_len=99999)

def load_processed_data(processed_path, parsed_reader):
    return parsed_reader.get_parsed_paragraphs_from_file(processed_path)

def extract_eventualities(processed_data, eventuality_extractor):
    return list(map(eventuality_extractor.extract_from_parsed_result, processed_data))

def extract_relations(processed_data, eventuality_lists, relation_extractor):
    return relation_extractor.extract(list(zip(processed_data, eventuality_lists)), output_format="relation", in_order=True)

class ASERPipe(object):
    def __init__(self, opt):
        self.opt = opt
        self.n_workers = opt.n_workers
        self.n_extractors = opt.n_extractors
        if opt.corenlp_path:
            self.sentence_parsers = [SentenceParser(opt.corenlp_path, opt.base_corenlp_port+_id) for _id in range(self.n_extractors)]
        else:
            self.sentence_parsers = [SentenceParser() for _id in range(self.n_extractors)]
        self.parsed_readers = [ParsedReader() for _id in range(self.n_extractors)]
        if opt.corenlp_path:
            self.eventuality_extractors = [EventualityExtractor(opt.corenlp_path, opt.base_corenlp_port+_id) for _id in range(self.n_extractors)]
        else:
            self.eventuality_extractors = [EventualityExtractor() for _id in range(self.n_extractors)]
        # self.relation_extractors = [SeedRuleRelationExtractor() for _id in range(self.n_extractors)]
        self.relation_extractors = [DiscourseRelationExtractor() for _id in range(self.n_extractors)]
        self.logger = init_logger(log_file=opt.log_path)

    def __del__(self):
        self.close()

    def close(self):
        for eventuality_extractor in self.eventuality_extractors:
            eventuality_extractor.close()
        self.logger.info("%d EventualityExtractors are closed." % (len(self.eventuality_extractors)))
        for relation_extractor in self.relation_extractors:
            relation_extractor.close()
        self.logger.info("%d RelationExtractors are closed." % (len(self.relation_extractors)))
        close_logger(self.logger)

    def run(self):
        with multiprocessing.Pool(self.n_workers) as pool:
            self.logger.info("Start the pipeline.")
            results = list()
            if os.path.exists(self.opt.processed_dir):
                self.logger.info("Loading processed data from %s." % (self.opt.processed_dir))
                processed_file_names = [file_name for file_name in iter_files(self.opt.processed_dir) if file_name.endswith(".jsonl")]
                for idx, processed_path in enumerate(processed_file_names):
                    results.append(pool.apply_async(run_file, args=(
                        None, processed_path, self.opt.processed_dir+os.sep, 
                        None, self.parsed_readers[idx%self.n_extractors], self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors])))
                    # results.append(run_file(
                    #     None, processed_path, self.opt.processed_dir+os.sep,
                    #     None, self.parsed_readers[idx%self.n_extractors], self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors]))
            elif os.path.exists(self.opt.raw_dir):
                self.logger.info("Processing raw data from %s." % (self.opt.raw_dir))
                raw_file_names = [file_name for file_name in iter_files(self.opt.raw_dir)]
                for idx, raw_path in enumerate(raw_file_names):
                    processed_path = os.path.splitext(raw_path)[0].replace(self.opt.raw_dir, self.opt.processed_dir, 1) + ".jsonl"
                    if not os.path.exists(os.path.dirname(processed_path)):
                        os.makedirs(os.path.dirname(processed_path))
                    results.append(pool.apply_async(run_file, args=(
                        raw_path, processed_path, self.opt.processed_dir+os.sep,
                        self.sentence_parsers[idx%self.n_extractors], None, self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors])))
                    # results.append(run_file(
                    #     raw_path, processed_path, self.opt.processed_dir+os.sep,
                    #     self.sentence_parsers[idx%self.n_extractors], None, self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors]))
            else:
                raise ValueError("Error: at least one of raw_dir and processed_dir should not be None.")
            pool.close()
            
            # merge all results
            eid2sids, rid2sids, eid2eventuality, rid2relation = defaultdict(list), defaultdict(list), dict(), dict()
            eventuality_counter, relation_counter = Counter(), Counter()
            for x in tqdm(results):
                x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = x.get()
                # x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = x
                eid2sids.update(x_eid2sids)
                rid2sids.update(x_rid2sids)
                for eid, eventuality in x_eid2eventuality.items():
                    eventuality_counter[eid] += eventuality.frequency
                    if eid not in eid2eventuality:
                        eid2eventuality[eid] = copy(eventuality)
                    else:
                        eid2eventuality[eid].update_frequency(eventuality)
                for rid, relation in x_rid2relation.items():
                    relation_counter[rid] += sum(relation.relations.values())
                    if rid not in rid2relation:
                        rid2relation[rid] = deepcopy(relation)
                    else:
                        rid2relation[rid].update_relations(relation)
            # del x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation
            total_eventuality, total_relation = sum(eventuality_counter.values()), sum(relation_counter.values())
            self.logger.info("%d eventualities (%d unique) have been extracted." % (total_eventuality, len(eid2eventuality)))
            self.logger.info("%d relations (%d unique) have been extracted." % (total_relation, len(rid2relation)))
            
            # filter high-frequency and low-frequency eventualities
            self.logger.info("Filtering high-frequency and low-frequency eventualities.")
            eventuality_frequency_lower_cnt_threshold = self.opt.eventuality_frequency_lower_cnt_threshold
            eventuality_frequency_upper_cnt_threshold = self.opt.eventuality_frequency_upper_percent_threshold * total_eventuality
            filtered_eids = set([eid for eid, freq in eventuality_counter.items() \
                if freq < eventuality_frequency_lower_cnt_threshold or freq > eventuality_frequency_upper_cnt_threshold])
            for filtered_eid in filtered_eids:
                eid2sids.pop(filtered_eid)
                eid2eventuality.pop(filtered_eid)
                total_eventuality -= eventuality_counter.pop(filtered_eid)
            del eventuality_counter
            self.logger.info("%d eventualities (%d unique) will be inserted into KG." % (total_eventuality, len(eid2eventuality)))

            # filter high-frequency and low-frequency relations
            self.logger.info("Filtering high-frequency and low-frequency relations.")
            relation_frequency_lower_cnt_threshold = self.opt.relation_frequency_lower_cnt_threshold
            relation_frequency_upper_cnt_threshold = self.opt.relation_frequency_upper_percent_threshold * total_relation
            filtered_rids = set([rid for rid, freq in relation_counter.items() \
                if freq < relation_frequency_lower_cnt_threshold or freq > relation_frequency_upper_cnt_threshold])
            filtered_rids.update(set([rid for rid, relation in rid2relation.items() \
                if relation.hid in filtered_eids or relation.tid in filtered_eids]))
            for filtered_rid in filtered_rids:
                rid2sids.pop(filtered_rid)
                rid2relation.pop(filtered_rid)
                total_relation -= relation_counter.pop(filtered_rid)
            del relation_counter
            self.logger.info("%d relations (%d unique) will be inserted into KG." % (total_relation, len(rid2relation)))

            # build eventuality KG
            self.logger.info("Storing inverted tables and building the KG.")
            if not os.path.exists(self.opt.kg_dir):
                os.mkdir(self.opt.kg_dir)
            with open(os.path.join(self.opt.kg_dir, "eid2sids.pkl"), "wb") as f:
                pickle.dump(eid2sids, f)
            with open(os.path.join(self.opt.kg_dir, "rid2sids.pkl"), "wb") as f:
                pickle.dump(rid2sids, f)
            del eid2sids, rid2sids

            kg_conn = ASERKGConnection(os.path.join(self.opt.kg_dir, "KG.db"), mode='insert')
            kg_conn.insert_eventualities(list(eid2eventuality.values()))
            kg_conn.insert_relations(list(rid2relation.values()))
            kg_conn.close()
            del eid2eventuality, rid2relation
            self.logger.info("Done.")