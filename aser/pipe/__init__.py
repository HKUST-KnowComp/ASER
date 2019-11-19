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
from aser.extract.event_extractor import EventualityExtractor
from aser.extract.relation_extractor import SeedRuleRelationExtractor
from aser.extract.parsed_reader import ParsedReader
from aser.extract.utils import iter_files, parse_sentense_with_stanford
from aser.utils.logging import init_logger, close_logger
from aser.relation import Relation
from aser.database.kg_connection import ASERKGConnection
from .utils import extract_eventualities_from_parsed_paragraph, extract_relations_from_parsed_paragraph

class ASERPipe(object):
    def __init__(self, opt):
        self.opt = opt
        self.n_workers = opt.n_workers
        # self.event_extractors = [EventualityExtractor(corenlp_path=opt.corenlp_path, corenlp_port=opt.base_corenlp_port+_id) for _id in range(opt.n_extractors)]
        self.event_extractors = [EventualityExtractor() for _id in range(opt.n_extractors)]
        self.relation_extractors = [SeedRuleRelationExtractor() for _id in range(opt.n_extractors)]
        self.logger = init_logger(log_file=opt.log_path)

    def __del__(self):
        self.close()

    def close(self):
        for event_extractor in self.event_extractors:
            event_extractor.close()
        self.logger.info("%d EventualityExtractors are closed." % (len(self.event_extractors)))
        for relation_extractor in self.relation_extractors:
            relation_extractor.close()
        self.logger.info("%d RelationExtractors are closed." % (len(self.relation_extractors)))
        close_logger(self.logger)

    def run(self):
        self.logger.info("Start the pipeline.")
        # load processed_data
        if not os.path.exists(self.opt.processed_dir):
            self.logger.info("Processing raw data from %s." % (self.opt.raw_dir))
            processed_data = self.process_raw_data(self.opt.raw_dir, self.opt.processed_dir)
            self.logger.info("%d processed data have been generated from %s." % (len(processed_data), self.opt.raw_dir))
        else:
            self.logger.info("Loading processed data from %s." % (self.opt.processed_dir))
            processed_data = self.load_processed_data(self.opt.processed_dir)
            self.logger.info("%d processed data have been loaded from %s." % (len(processed_data), self.opt.processed_dir))

        # remove prefixes of sids
        prefix = self.opt.processed_dir
        if not prefix.endswith(os.sep):
            prefix = prefix + os.sep
        sids = list()
        for processed_para in processed_data:
            sids_para = [sent["sid"].replace(prefix, "", 1) for sent in processed_para]
            sids.append(sids_para)

        # extract eventualities from processed data
        self.logger.info("Extracting eventualities from processed data.")
        eventuality_lists = self.extract_eventualities(processed_data)
        eid2sids = defaultdict(list)
        eid2eventuality = dict()
        for sids_para, es_para in zip(sids, eventuality_lists):
            if len(sids_para) != len(es_para):
                print("Error: len(sids_para) != len(es_para)", len(sids_para), len(es_para))
                print(sids_para)
                print(es_para)
                raise ValueError
            for sid, es_sent in zip(sids_para, es_para):
                for idx, e in enumerate(es_sent):
                    eid2sids[e.eid].append(sid)
                    if e.eid not in eid2eventuality:
                        eid2eventuality[e.eid] = e
                    else:
                        eid2eventuality[e.eid].frequency += e.frequency
                        es_sent[idx] = eid2eventuality[e.eid]

        # filter high-frequency and low-frequency eventualities
        event_counter = Counter()
        for eid, e in eid2eventuality.items():
            event_counter[eid] = e.frequency
        event_total = sum(event_counter.values())
        self.logger.info("%d eventualities (%d unique) have been extracted." % (event_total, len(event_counter)))
        event_frequency_lower_cnt_threshold = self.opt.eventuality_frequency_lower_cnt_threshold
        event_frequency_upper_cnt_threshold = self.opt.eventuality_frequency_upper_percent_threshold * event_total
        filtered_eids = set([eid for eid, freq in event_counter.items() \
            if freq < event_frequency_lower_cnt_threshold or freq > event_frequency_upper_cnt_threshold])
        for filtered_eid in filtered_eids:
            eid2sids.pop(filtered_eid)
            eid2eventuality.pop(filtered_eid)
            event_counter.pop(filtered_eid)
        for es_para in eventuality_lists:
            for es_sent in es_para:
                es_sent.filter_by_frequency(event_frequency_lower_cnt_threshold, event_frequency_upper_cnt_threshold)
        event_total = sum(event_counter.values())
        self.logger.info("%d eventualities (%d unique) will be inserted into KG." % (event_total, len(event_counter)))

        # extract relations from eventuality_lists
        self.logger.info("Extracting relations from processed data.")
        relation_lists = self.extract_relations(processed_data, eventuality_lists)
        rid2sids = defaultdict(list)
        rid2relation = dict()
        for sids_para, rs_para in zip(sids, relation_lists):
            len_para = len(sids_para)
            if len(rs_para) != 2*len_para-1:
                print("Error: len(rs_para) != 2*len_para-1:", len_para, len(rs_para))
                print(sids_para)
                print(rs_para)
                raise ValueError
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
           
        # filter high-frequency and low-frequency relations
        relation_counter = Counter()
        for rid, r in rid2relation.items():
            relation_counter[rid] = sum(r.relations.values())
        relation_total = sum(relation_counter.values())
        self.logger.info("%d relations (%d unique) have been extracted." % (relation_total, len(relation_counter)))
        relation_frequency_lower_cnt_threshold = self.opt.relation_frequency_lower_cnt_threshold
        relation_frequency_upper_cnt_threshold = self.opt.relation_frequency_upper_percent_threshold * relation_total
        filtered_rids = set([rid for rid, freq in relation_counter.items() \
            if freq < relation_frequency_lower_cnt_threshold or freq > relation_frequency_upper_cnt_threshold])
        for filtered_rid in filtered_rids:
            rid2sids.pop(filtered_rid)
            rid2relation.pop(filtered_rid)
            relation_counter.pop(filtered_rid)
        relation_total = sum(relation_counter.values())
        self.logger.info("%d relations (%d unique) will be inserted into KG." % (relation_total, len(relation_counter)))

        # build eventuality KG
        self.logger.info("Start storing inverted tables and building the KG.")
        if not os.path.exists(self.opt.kg_dir):
            os.mkdir(self.opt.kg_dir)
        with open(os.path.join(self.opt.kg_dir, "eid2sids.pkl"), "wb") as f:
            pickle.dump(eid2sids, f)
        with open(os.path.join(self.opt.kg_dir, "rid2sids.pkl"), "wb") as f:
            pickle.dump(rid2sids, f)

        kg_conn = ASERKGConnection(os.path.join(self.opt.kg_dir, "KG.db"), mode='insert')
        kg_conn.insert_eventualities(eid2eventuality.values())
        kg_conn.insert_relations(rid2relation.values())
        kg_conn.close()
        self.logger.info("Done.")

    def process_raw_data(self, raw_dir, processed_dir):
        # corenlp_path=opt.corenlp_path, corenlp_port=opt.base_corenlp_port+_id)
        with multiprocessing.Pool(self.n_workers) as pool:
            file_names = [file_name for file_name in iter_files(raw_dir) if file_name.endswith(".txt")]
            processed_data = list()
            # TODO
            raise NotImplementedError

        return processed_data

    def load_processed_data(self, processed_dir):
        with multiprocessing.Pool(self.n_workers) as pool:
            file_names = [file_name for file_name in iter_files(processed_dir) if file_name.endswith(".jsonl")]
            parsed_reader = ParsedReader()
            processed_data = list()

            r = pool.map_async(parsed_reader.get_parsed_paragraphs_from_file, tqdm(file_names, ascii=True))
            pool.close()
            pool.join()
            for x in r.get():
                processed_data.extend(x)
        return processed_data

    def extract_eventualities(self, processed_data):
        with multiprocessing.Pool(self.n_workers) as pool:
            eventualities = list()
            pool_results = list()

            pbar = tqdm(total=len(processed_data), ascii=True)
            def update(*a):
                pbar.update()

            for idx, processed_para in enumerate(processed_data):
                pool_results.append(pool.apply_async(
                    extract_eventualities_from_parsed_paragraph, 
                    args=(self.event_extractors[idx%len(self.event_extractors)], processed_para),
                    callback=update))
                # eventualities.append(extract_eventualities_from_parsed_paragraph(
                #     self.event_extractors[idx%len(self.event_extractors)], processed_para))
            pool.close()
            pool.join()
            for x in pool_results:
                eventualities.append(x.get())
        return eventualities

    def extract_relations(self, processed_data, eventuality_lists):
        with multiprocessing.Pool(self.n_workers) as pool:
            relations = list()
            pool_results = list()

            pbar = tqdm(total=len(processed_data), ascii=True)
            def update(*a):
                pbar.update()

            for idx, (processed_para, eventualities_para) in enumerate(zip(processed_data, eventuality_lists)):
                pool_results.append(pool.apply_async(extract_relations_from_parsed_paragraph,
                    args=(self.relation_extractors[idx%len(self.relation_extractors)], processed_para, eventualities_para),
                    callback=update))
                # relations.append(extract_relations_from_parsed_paragraph(
                #     self.relation_extractors[idx%len(self.relation_extractors)], 
                #     processed_para, eventualities_para))
            pool.close()
            pool.join()
            for x in pool_results:
                relations.append(x.get())
        return relations