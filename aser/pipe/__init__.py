import json
import os
import random
import time
import pickle
import multiprocessing
from tqdm import tqdm
from copy import deepcopy
from functools import partial
from collections import Counter, defaultdict
from aser.extract.eventuality_extractor import EventualityExtractor
from aser.extract.relation_extractor import SeedRuleRelationExtractor
from aser.extract.parsed_reader import ParsedReader
from aser.extract.utils import iter_files, parse_sentense_with_stanford
from aser.utils.logging import init_logger, close_logger
from aser.relation import Relation
from aser.database.kg_connection import ASERKGConnection

def run_file(raw_path=None, processed_path=None, sentence_parser=None, parsed_reader=None, eventuality_extractor=None, relation_extractor=None):
    # process raw data or load processed data
    if processed_path:
        processed_data = load_processed_data(processed_path, parsed_reader)
    elif raw_path:
        processed_data = process_raw_file(raw_path, processed_path, sentence_parser)
    else:
        raise ValueError("Error: at least one of raw_path and processed_path should not be None.")

    # remove prefix of sids
    prefix = os.path.split(processed_path)[0] + os.sep
    sids = list()
    for processed_para in processed_data:
        sids_para = [sent["sid"].replace(prefix, "", 1) for sent in processed_para]
        sids.append(sids_para)

    # extract eventualities from processed data
    eventuality_lists = extract_eventualities(processed_data, eventuality_extractor)
    eid2sids = defaultdict(list)
    eid2eventuality = dict()
    for sids_para, es_para in zip(sids, eventuality_lists):
        if len(sids_para) != len(es_para):
            raise ValueError("Error: len(sids_para) != len(es_para)", len(sids_para), len(es_para))
        for sid, es_sent in zip(sids_para, es_para):
            for idx, e in enumerate(es_sent):
                eid2sids[e.eid].append(sid)
                if e.eid not in eid2eventuality:
                    eid2eventuality[e.eid] = e
                else:
                    eid2eventuality[e.eid].frequency += e.frequency
                    es_sent[idx] = eid2eventuality[e.eid]
    
    # extract relations from eventuality_lists
    relation_lists = extract_relations(processed_data, eventuality_lists, relation_extractor)
    rid2sids = defaultdict(list)
    rid2relation = dict()
    for sids_para, rs_para in zip(sids, relation_lists):
        len_para = len(sids_para)
        if len_para > 0 and len(rs_para) != 2*len_para-1:
            raise ValueError("Error: len(rs_para) != 2*len_para-1:", len_para, len(rs_para))
        for idx in range(len_para):
            rs_in_sent = rs_para[idx]
            for r in rs_in_sent:
                rid2sids[r.rid].append((sids_para[idx],))
                if r.rid not in rid2relation:
                    rid2relation[r.rid] = deepcopy(r)
                else:
                    rid2relation[r.rid].update_relations(r.relations)
        for idx in range(len_para - 1):
            rs_between_sents = rs_para[len_para+idx]
            for r in rs_between_sents:
                rid2sids[r.rid].append((sids_para[idx], sids_para[idx+1]))
                if r.rid not in rid2relation:
                    rid2relation[r.rid] = deepcopy(r)
                else:
                    rid2relation[r.rid].update_relations(r.relations)
    
    return sids, eid2sids, rid2sids, eid2eventuality, rid2relation

def process_raw_file(raw_path, processed_path, sentence_parser):
    # TODO: read data from raw_path
    raw_data = list()
    # TODO: process data
    processed_data = list()
    # TODO: save processed data to processed_path

    return processed_data

def load_processed_data(processed_path, parsed_reader):
    processed_data = parsed_reader.get_parsed_paragraphs_from_file(processed_path)
    return processed_data

def extract_eventualities(processed_data, eventuality_extractor):
    eventualities = list()
    for para in processed_data:
        eventualities.append(list(map(eventuality_extractor.extract_from_parsed_result, para)))
    return eventualities

def extract_relations(processed_data, eventuality_lists, relation_extractor):
    relations = list()
    for para, es_para in zip(processed_data, eventuality_lists):
        relations.append(relation_extractor.extract(list(zip(para, es_para)), output_format="relation", in_order=True))
    return relations

class ASERPipe(object):
    def __init__(self, opt):
        self.opt = opt
        self.n_workers = opt.n_workers
        self.n_extractors = opt.n_extractors
        self.sentence_parser = None
        self.parsed_reader = ParsedReader()
        self.eventuality_extractors = [EventualityExtractor() for _id in range(self.n_extractors)]
        self.relation_extractors = [SeedRuleRelationExtractor() for _id in range(self.n_extractors)]
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
            if os.path.exists(self.opt.processed_dir):
                self.logger.info("Loading processed data from %s." % (self.opt.processed_dir))
                processed_file_names = [file_name for file_name in iter_files(self.opt.processed_dir) if file_name.endswith(".jsonl")]
                results = list()
                for idx, processed_path in enumerate(processed_file_names):
                    results.append(pool.apply_async(run_file, args=(
                        None, processed_path, None, self.parsed_reader, self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors])))
                    # results.append(run_file(
                    #     None, processed_path, None, self.parsed_reader, self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors]))
            elif os.path.exists(self.opt.raw_dir):
                self.logger.info("Processing raw data from %s." % (self.opt.raw_dir))
                raw_file_names = [file_name for file_name in iter_files(self.opt.raw_dir) if file_name.endswith(".txt")]
                for idx, raw_path in enumerate(raw_file_names):
                    processed_path = os.path.splitext(raw_path)[0].replace(self.opt.raw_dir, self.opt.processed_dir, 1) + ".jsonl"
                    results.append(pool.apply_async(run_file, args=(
                        raw_path, processed_path, self.sentence_parser, None, self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors])))
                    # results.append(run_file(
                    #     raw_path, processed_path, self.sentence_parser, None, self.eventuality_extractors[idx%self.n_extractors], self.relation_extractors[idx%self.n_extractors]))
            else:
                raise ValueError("Error: at least one of raw_dir and processed_dir should not be None.")
            pool.close()
            
            # merge all results
            sids, eid2sids, rid2sids, eid2eventuality, rid2relation = list(), defaultdict(list), defaultdict(list), dict(), dict()
            eventuality_counter, relation_counter = Counter(), Counter()
            for x in tqdm(results):
                x_sids, x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = x.get()
                # x_sids, x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = x
                sids.extend(x_sids)
                eid2sids.update(x_eid2sids)
                rid2sids.update(x_rid2sids)
                for eid, eventuality in x_eid2eventuality.items():
                    eventuality_counter[eid] += eventuality.frequency
                    if eid not in eid2eventuality:
                        eid2eventuality[eid] = eventuality
                    else:
                        eid2eventuality[eid].update_frequency(eventuality)
                for rid, relation in x_rid2relation.items():
                    relation_counter[rid] += sum(relation.relations.values())
                    if rid not in rid2relation:
                        rid2relation[rid] = relation
                    else:
                        rid2relation[rid].update_relations(relation)
            del x_sids, x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation
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
            kg_conn.insert_eventualities(eid2eventuality.values())
            kg_conn.insert_relations(rid2relation.values())
            kg_conn.close()
            del eid2eventuality, rid2relation
            self.logger.info("Done.")