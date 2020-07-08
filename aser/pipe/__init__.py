import json
import os
import shutil
import random
import time
import pickle
import multiprocessing
import math
from tqdm import tqdm
from copy import copy, deepcopy
from functools import partial
from collections import Counter, defaultdict
from aser.extract.sentence_parser import SentenceParser
from aser.extract.parsed_reader import ParsedReader
from aser.extract.aser_extractor import SeedRuleASERExtractor, DiscourseASERExtractor1, DiscourseASERExtractor2, DiscourseASERExtractor3
from aser.extract.utils import EMPTY_SENT_PARSED_RESULT
from aser.extract.utils import iter_files, parse_sentense_with_stanford
from aser.utils.logging import init_logger, close_logger
from aser.eventuality import Eventuality
from aser.relation import Relation
from aser.database.kg_connection import ASERKGConnection



def run_files(raw_paths=None, processed_paths=None, prefix_to_be_removed="",
    sentence_parser=None, parsed_reader=None, aser_extractor=None):
    eid2sids = defaultdict(list)
    rid2sids = defaultdict(list)
    eid2eventuality = dict()
    rid2relation = dict()
    if raw_paths:
        for raw_path, processed_path in zip(raw_paths, processed_paths):
            x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = \
                run_file(raw_path, processed_path, prefix_to_be_removed,
                    sentence_parser, parsed_reader, aser_extractor)
            for eid, sids in x_eid2sids.items():
                eid2sids[eid].extend(sids)
            for rid, sids in x_rid2sids.items():
                rid2sids[rid].extend(sids)
            for eid, eventuality in x_eid2eventuality.items():
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = eventuality
                else:
                    eid2eventuality[eid].update(eventuality)
            for rid, relation in x_rid2relation.items():
                if rid not in rid2relation:
                    rid2relation[rid] = relation
                else:
                    rid2relation[rid].update(relation)
    else:
        for processed_path in processed_paths:
            x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = \
                run_file(None, processed_path, prefix_to_be_removed,
                    sentence_parser, parsed_reader, aser_extractor)
            for eid, sids in x_eid2sids.items():
                eid2sids[eid].extend(sids)
            for rid, sids in x_rid2sids.items():
                rid2sids[rid].extend(sids)
            for eid, eventuality in x_eid2eventuality.items():
                if eid not in eid2eventuality:
                    eid2eventuality[eid] = eventuality
                else:
                    eid2eventuality[eid].update(eventuality)
            for rid, relation in x_rid2relation.items():
                if rid not in rid2relation:
                    rid2relation[rid] = relation
                else:
                    rid2relation[rid].update(relation)
    del x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation
    return eid2sids, rid2sids, eid2eventuality, rid2relation

def run_file(raw_path=None, processed_path=None, prefix_to_be_removed="",
    sentence_parser=None, parsed_reader=None, aser_extractor=None):

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
        # document.append(EMPTY_SENT_PARSED_RESULT)

    eventuality_lists, relation_lists = aser_extractor.extract_from_parsed_result(document)

    # merge eventualities
    eid2sids = defaultdict(list)
    eid2eventuality = dict()
    for sentence, eventuality_list in zip(document, eventuality_lists):
        for eventuality in eventuality_list:
            eid2sids[eventuality.eid].append(sentence["sid"])
            if eventuality.eid not in eid2eventuality:
                eid2eventuality[eventuality.eid] = deepcopy(eventuality)
            else:
                eid2eventuality[eventuality.eid].update(eventuality)

    # merge relations
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
                    rid2relation[relation.rid].update(relation)
    # PS
    for idx in range(len_doc-1):
        relation_list = relation_lists[len_doc+idx]
        for relation in relation_list:
            if sum(relation.relations.values()) > 0:
                rid2sids[relation.rid].append((document[idx]["sid"], document[idx+1]["sid"]))
                if relation.rid not in rid2relation:
                    rid2relation[relation.rid] = deepcopy(relation)
                else:
                    rid2relation[relation.rid].update(relation)
    
    return eid2sids, rid2sids, eid2eventuality, rid2relation

def process_raw_file(raw_path, processed_path, sentence_parser):
    return sentence_parser.parse_raw_file(raw_path, processed_path, max_len=1000)

def load_processed_data(processed_path, parsed_reader):
    return parsed_reader.get_parsed_paragraphs_from_file(processed_path)

class ASERPipe(object):
    def __init__(self, opt):
        self.opt = opt
        self.n_workers = opt.n_workers
        self.n_extractors = opt.n_extractors
        self.sentence_parsers = [SentenceParser(
            corenlp_path=opt.corenlp_path, corenlp_port=opt.base_corenlp_port+_id) for _id in range(self.n_extractors)]
        self.parsed_readers = [ParsedReader() for _id in range(self.n_extractors)]
        # self.aser_extractors = [SeedRuleASERExtractor() for _id in range(self.n_extractors)]
        # self.aser_extractors = [DiscourseASERExtractor1() for _id in range(self.n_extractors)]
        self.aser_extractors = [DiscourseASERExtractor2() for _id in range(self.n_extractors)]
        # self.aser_extractors = [DiscourseASERExtractor3() for _id in range(self.n_extractors)]
        self.logger = init_logger(log_file=opt.log_path)

    def __del__(self):
        self.close()

    def close(self):
        for _id in range(self.n_extractors):
            self.sentence_parsers[_id].close()
            self.parsed_readers[_id].close()
            self.aser_extractors[_id].close()
        self.logger.info("%d ASER Extractors are closed." % (len(self.aser_extractors)))
        close_logger(self.logger)

    def run(self):
        self.logger.info("Start the pipeline.")
        if os.path.exists(self.opt.raw_dir):
            if not os.path.exists(self.opt.processed_dir):
                os.mkdir(self.opt.processed_dir)
            self.logger.info("Processing raw data from %s." % (self.opt.raw_dir))
            raw_paths, processed_paths = list(), list()
            for file_name in iter_files(self.opt.raw_dir):
                raw_paths.append(file_name)
                processed_paths.append(
                    os.path.splitext(file_name)[0].replace(self.opt.raw_dir, self.opt.processed_dir, 1) + ".jsonl")
        elif os.path.exists(self.opt.processed_dir):
            self.logger.info("Loading processed data from %s." % (self.opt.processed_dir))
            raw_paths = list()
            processed_paths = [file_name for file_name in iter_files(self.opt.processed_dir) if file_name.endswith(".jsonl")]
        else:
            raise ValueError("Error: at least one of raw_dir and processed_dir should not be None.")
        self.logger.info("Number of files: %d." % (len(processed_paths)))
        prefix_to_be_removed = self.opt.processed_dir+os.sep

        
        if self.n_workers > 1:
            with multiprocessing.Pool(self.n_workers) as pool:
                results = list()
                if len(processed_paths) < 10000:
                    for worker_idx, (raw_path, processed_path) in enumerate(zip(raw_paths, processed_paths)):
                        extractor_idx = worker_idx%self.n_extractors
                        results.append(pool.apply_async(run_file, args=(
                            raw_path, processed_path, prefix_to_be_removed, 
                            self.sentence_parsers[extractor_idx], self.parsed_readers[extractor_idx], 
                            self.aser_extractors[extractor_idx])))
                else:
                    chunk_size = 10
                    while math.ceil(len(processed_paths)/chunk_size) > 10000:
                        chunk_size *= 10
                    for worker_idx in range(math.ceil(len(processed_paths)/chunk_size)):
                        extractor_idx = worker_idx%self.n_extractors
                        i = worker_idx * chunk_size
                        j = min(i +  chunk_size, len(processed_paths))
                        results.append(pool.apply_async(run_files, args=(
                            raw_paths[i:j], processed_paths[i:j], prefix_to_be_removed, 
                            self.sentence_parsers[extractor_idx], self.parsed_readers[extractor_idx], 
                            self.aser_extractors[extractor_idx])))
                pool.close()
                
                # merge all results
                eid2sids, rid2sids, eid2eventuality, rid2relation = defaultdict(list), defaultdict(list), dict(), dict()
                eventuality_counter, relation_counter = Counter(), Counter()
                for x in tqdm(results):
                    x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = x.get()
                    if len(eid2eventuality) == 0 and len(rid2relation) == 0:
                        eid2sids, rid2sids, eid2eventuality, rid2relation = x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation
                        for eid, eventuality in x_eid2eventuality.items():
                            eventuality_counter[eid] += eventuality.frequency
                        for rid, relation in x_rid2relation.items():
                            relation_counter[rid] += sum(relation.relations.values())
                    else:
                        for eid, sids in x_eid2sids.items():
                            eid2sids[eid].extend(sids)
                        for rid, sids in x_rid2sids.items():
                            rid2sids[rid].extend(sids)
                        for eid, eventuality in x_eid2eventuality.items():
                            eventuality_counter[eid] += eventuality.frequency
                            if eid not in eid2eventuality:
                                eid2eventuality[eid] = eventuality
                            else:
                                eid2eventuality[eid].update(eventuality)
                        for rid, relation in x_rid2relation.items():
                            relation_counter[rid] += sum(relation.relations.values())
                            if rid not in rid2relation:
                                rid2relation[rid] = relation
                            else:
                                rid2relation[rid].update(relation)
                if len(eid2sids) > 0 or len(rid2sids) > 0:
                    del x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation
        else:
            eid2sids, rid2sids, eid2eventuality, rid2relation = run_files(
                raw_paths, processed_paths, prefix_to_be_removed, 
                self.sentence_parsers[0], self.parsed_readers[0], 
                self.aser_extractors[0])
            eventuality_counter, relation_counter = Counter(), Counter()
            for eid, eventuality in eid2eventuality.items():
                eventuality_counter[eid] += eventuality.frequency
            for rid, relation in rid2relation.items():
                relation_counter[rid] += sum(relation.relations.values())

        total_eventuality, total_relation = sum(eventuality_counter.values()), sum(relation_counter.values())
        self.logger.info("%d eventualities (%d unique) have been extracted." % (total_eventuality, len(eid2eventuality)))
        self.logger.info("%d relations (%d unique) have been extracted." % (total_relation, len(rid2relation)))

        if self.opt.full_kg_dir:
            # build eventuality KG
            self.logger.info("Storing inverted tables.")
            if not os.path.exists(self.opt.full_kg_dir):
                os.mkdir(self.opt.full_kg_dir)
            with open(os.path.join(self.opt.full_kg_dir, "eid2sids.pkl"), "wb") as f:
                pickle.dump(eid2sids, f)
            with open(os.path.join(self.opt.full_kg_dir, "rid2sids.pkl"), "wb") as f:
                pickle.dump(rid2sids, f)

            self.logger.info("Building the full KG.")
            kg_conn = ASERKGConnection(os.path.join(self.opt.full_kg_dir, "KG.db"), mode='insert')
            kg_conn.insert_eventualities(eid2eventuality.values())
            kg_conn.insert_relations(rid2relation.values())
            kg_conn.close()
            self.logger.info("Done.")

        if self.opt.core_kg_dir:
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
            # del eventuality_counter
            self.logger.info("%d eventualities (%d unique) will be inserted into the core KG." % (total_eventuality, len(eid2eventuality)))

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
            # del relation_counter
            self.logger.info("%d relations (%d unique) will be inserted into the core KG." % (total_relation, len(rid2relation)))

            if len(filtered_eids) == 0 and len(filtered_rids) == 0 and self.opt.full_kg_dir:
                # copy KG
                self.logger.info("Copying the full KG as the core KG.")
                if not os.path.exists(self.opt.core_kg_dir):
                    os.mkdir(self.opt.core_kg_dir)
                shutil.copyfile(os.path.join(self.opt.full_kg_dir, "eid2sids.pkl"), os.path.join(self.opt.core_kg_dir, "eid2sids.pkl"))
                shutil.copyfile(os.path.join(self.opt.full_kg_dir, "rid2sids.pkl"), os.path.join(self.opt.core_kg_dir, "rid2sids.pkl"))
                shutil.copyfile(os.path.join(self.opt.full_kg_dir, "KG.db"), os.path.join(self.opt.core_kg_dir, "KG.db"))
            else:
                # build eventuality KG
                self.logger.info("Storing inverted tables.")
                if not os.path.exists(self.opt.core_kg_dir):
                    os.mkdir(self.opt.core_kg_dir)
                with open(os.path.join(self.opt.core_kg_dir, "eid2sids.pkl"), "wb") as f:
                    pickle.dump(eid2sids, f)
                with open(os.path.join(self.opt.core_kg_dir, "rid2sids.pkl"), "wb") as f:
                    pickle.dump(rid2sids, f)
                # del eid2sids, rid2sids

                self.logger.info("Building the core KG.")
                kg_conn = ASERKGConnection(os.path.join(self.opt.core_kg_dir, "KG.db"), mode='insert')
                kg_conn.insert_eventualities(eid2eventuality.values())
                kg_conn.insert_relations(rid2relation.values())
                kg_conn.close()
                # del eid2eventuality, rid2relation
                self.logger.info("Done.")