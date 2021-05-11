import os
import shutil
import pickle
import multiprocessing
import math
from collections import Counter, defaultdict
from tqdm import tqdm
from ..conceptualize.aser_conceptualizer import SeedRuleASERConceptualizer, ProbaseASERConceptualizer
from ..conceptualize.utils import conceptualize_eventualities, build_concept_relations
from ..database.kg_connection import ASERKGConnection, ASERConceptConnection
from ..extract.aser_extractor import BaseASERExtractor, SeedRuleASERExtractor, DiscourseASERExtractor
from ..extract.parsed_reader import ParsedReader
from ..extract.sentence_parser import SentenceParser
from ..extract.utils import EMPTY_SENT_PARSED_RESULT
from ..extract.utils import extract_file
from ..extract.utils import iter_files
from ..utils.logging import init_logger, close_logger

MAX_THREADS = 1024


def extract_files(
    raw_paths=None,
    processed_paths=None,
    prefix_to_be_removed="",
    sentence_parser=None,
    parsed_reader=None,
    aser_extractor=None
):
    """ Extract eventualities and relations from files (which contain raw texts or parsed results)

    :param raw_paths: the file paths each of which contains raw texts
    :type raw_paths: Tuple[List[str], None]
    :param processed_paths: the file paths  each of which stores the parsed result
    :type processed_paths: List[str]
    :param prefix_to_be_removed: the prefix in sids to remove
    :type prefix_to_be_removed: str
    :param sentence_parser: the sentence parser to parse raw text
    :type sentence_parser: SentenceParser
    :param parsed_reader: the parsed reader to load parsed results
    :type parsed_reader: ParsedReader
    :param aser_extractor: the ASER extractor to extract both eventualities and relations
    :type aser_extractor: BaseASERExtractor
    :return: a dictionary from eid to sids, a dictionary from rid to sids, a dictionary from eid to eventuality, and a dictionary from rid to relation
    :rtype: Tuple[Dict[str, List[str]], Dict[str, List[str]], Dict[str, aser.eventuality.Eventuality], Dict[str, aser.relation.Relation]]
    """

    eid2sids = defaultdict(list)
    rid2sids = defaultdict(list)
    eid2eventuality = dict()
    rid2relation = dict()

    if raw_paths is None or len(raw_paths) == 0:
        raw_paths = [""] * len(processed_paths)

    for raw_path, processed_path in zip(raw_paths, processed_paths):
        x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = extract_file(
            raw_path, processed_path, prefix_to_be_removed, sentence_parser, parsed_reader, aser_extractor
        )
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



class ASERPipe(object):
    def __init__(self, opt):
        self.opt = opt
        self.n_workers = opt.n_workers
        self.n_extractors = opt.n_extractors
        self.sentence_parsers = []
        self.aser_extractors = []

        for _id in range(self.n_extractors):
            corenlp_path = opt.corenlp_path
            corenlp_port = opt.base_corenlp_port + _id
            self.sentence_parsers.append(SentenceParser(corenlp_path=corenlp_path, corenlp_port=corenlp_port))
            self.aser_extractors.append(DiscourseASERExtractor(corenlp_path=corenlp_path, corenlp_port=corenlp_port))

        parsed_reader = ParsedReader()
        self.parsed_readers = [parsed_reader] * self.n_extractors  # shallow copy

        if opt.concept_kg_dir == "":
            self.conceptualizer = None
        else:
            if opt.concept_method == "probase":
                self.conceptualizer = ProbaseASERConceptualizer(
                    probase_path=opt.probase_path, probase_topk=opt.concept_topk
                )
            elif opt.concept_method == "seed":
                self.conceptualizer = SeedRuleASERConceptualizer()
            else:
                raise ValueError("Error: %s = is not supported." % (opt.concept_method))

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
        self.logger.info("Starting the pipeline.")

        if os.path.exists(self.opt.raw_dir):
            if not os.path.exists(self.opt.processed_dir):
                os.mkdir(self.opt.processed_dir)
            self.logger.info("Processing raw data from %s." % (self.opt.raw_dir))
            raw_paths, processed_paths = list(), list()
            for file_name in iter_files(self.opt.raw_dir):
                raw_paths.append(file_name)
                processed_paths.append(
                    os.path.splitext(file_name)[0].replace(self.opt.raw_dir, self.opt.processed_dir, 1) + ".jsonl"
                )
        elif os.path.exists(self.opt.processed_dir):
            self.logger.info("Loading processed data from %s." % (self.opt.processed_dir))
            raw_paths = list()
            processed_paths = [
                file_name for file_name in iter_files(self.opt.processed_dir) if file_name.endswith(".jsonl")
            ]
        else:
            raise ValueError("Error: at least one of raw_dir and processed_dir should not be None.")

        self.logger.info("Number of files: %d." % (len(processed_paths)))

        prefix_to_be_removed = self.opt.processed_dir + os.sep
        raw_paths.sort()
        processed_paths.sort()

        # constructing extractors is time-consuming so that we apply `extract_files` instead of `extract_file`
        if self.n_workers > 1:
            with multiprocessing.Pool(self.n_workers) as pool:
                results = list()

                chunk_size = 1
                while math.ceil(len(processed_paths) / chunk_size) > MAX_THREADS:
                    chunk_size *= 2
                for worker_idx in range(math.ceil(len(processed_paths) / chunk_size)):
                    extractor_idx = worker_idx % self.n_extractors
                    i = worker_idx * chunk_size
                    j = min(i + chunk_size, len(processed_paths))
                    results.append(
                        pool.apply_async(
                            extract_files,
                            args=(
                                raw_paths[i:j], processed_paths[i:j], prefix_to_be_removed,
                                self.sentence_parsers[extractor_idx], self.parsed_readers[extractor_idx],
                                self.aser_extractors[extractor_idx]
                            )
                        )
                    )
                pool.close()

                # merge all results
                eid2sids, rid2sids, eid2eventuality, rid2relation = defaultdict(list), defaultdict(list), dict(), dict()
                eventuality_counter, relation_counter = Counter(), Counter()
                for x in tqdm(results):
                    x_eid2sids, x_rid2sids, x_eid2eventuality, x_rid2relation = x.get()
                    if len(eid2eventuality) == 0 and len(rid2relation) == 0:
                        eid2sids = x_eid2sids
                        rid2sids = x_rid2sids
                        eid2eventuality = x_eid2eventuality
                        rid2relation = x_rid2relation
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
            eid2sids, rid2sids, eid2eventuality, rid2relation = extract_files(
                raw_paths, processed_paths, prefix_to_be_removed, self.sentence_parsers[0], self.parsed_readers[0],
                self.aser_extractors[0]
            )
            eventuality_counter, relation_counter = Counter(), Counter()
            for eid, eventuality in eid2eventuality.items():
                eventuality_counter[eid] += eventuality.frequency
            for rid, relation in rid2relation.items():
                relation_counter[rid] += sum(relation.relations.values())

        total_eventuality, total_relation = sum(eventuality_counter.values()), sum(relation_counter.values())
        self.logger.info(
            "%d eventualities (%d unique) have been extracted." % (total_eventuality, len(eid2eventuality))
        )
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
            # filter low-frequency eventualities
            self.logger.info("Filtering low-frequency eventualities.")
            eventuality_frequency_threshold = self.opt.eventuality_frequency_threshold
            filtered_eids = set(
                [eid for eid, freq in eventuality_counter.items() if freq < eventuality_frequency_threshold]
            )
            for filtered_eid in filtered_eids:
                eid2sids.pop(filtered_eid)
                eid2eventuality.pop(filtered_eid)
                total_eventuality -= eventuality_counter.pop(filtered_eid)
            self.logger.info(
                "%d eventualities (%d unique) will be inserted into the core KG." %
                (total_eventuality, len(eid2eventuality))
            )

            # filter low-frequency relations
            self.logger.info("Filtering low-frequency relations.")
            relation_weight_threshold = self.opt.relation_weight_threshold
            filtered_rids = set([rid for rid, freq in relation_counter.items() if freq < relation_weight_threshold])
            filtered_rids.update(set([rid for rid, relation in rid2relation.items() \
                                      if relation.hid in filtered_eids or relation.tid in filtered_eids]))
            for filtered_rid in filtered_rids:
                rid2sids.pop(filtered_rid)
                rid2relation.pop(filtered_rid)
                total_relation -= relation_counter.pop(filtered_rid)
            self.logger.info(
                "%d relations (%d unique) will be inserted into the core KG." % (total_relation, len(rid2relation))
            )

            if len(filtered_eids) == 0 and len(filtered_rids) == 0 and self.opt.full_kg_dir:
                # copy KG
                self.logger.info("Copying the full KG as the core KG.")
                if not os.path.exists(self.opt.core_kg_dir):
                    os.mkdir(self.opt.core_kg_dir)
                shutil.copyfile(
                    os.path.join(self.opt.full_kg_dir, "eid2sids.pkl"),
                    os.path.join(self.opt.core_kg_dir, "eid2sids.pkl")
                )
                shutil.copyfile(
                    os.path.join(self.opt.full_kg_dir, "rid2sids.pkl"),
                    os.path.join(self.opt.core_kg_dir, "rid2sids.pkl")
                )
                shutil.copyfile(
                    os.path.join(self.opt.full_kg_dir, "KG.db"), os.path.join(self.opt.core_kg_dir, "KG.db")
                )
            else:
                # build eventuality KG
                self.logger.info("Storing inverted tables.")
                if not os.path.exists(self.opt.core_kg_dir):
                    os.mkdir(self.opt.core_kg_dir)
                with open(os.path.join(self.opt.core_kg_dir, "eid2sids.pkl"), "wb") as f:
                    pickle.dump(eid2sids, f)
                with open(os.path.join(self.opt.core_kg_dir, "rid2sids.pkl"), "wb") as f:
                    pickle.dump(rid2sids, f)

                self.logger.info("Building the core KG.")
                kg_conn = ASERKGConnection(os.path.join(self.opt.core_kg_dir, "KG.db"), mode='insert')
                kg_conn.insert_eventualities(eid2eventuality.values())
                kg_conn.insert_relations(rid2relation.values())
                kg_conn.close()
                self.logger.info("Done.")

        if self.opt.concept_kg_dir:
            self.logger.info("Conceptualizing eventualities.")

            eventuality_threshold_to_conceptualize = self.opt.eventuality_threshold_to_conceptualize
            filtered_eids = set(
                [eid for eid, freq in eventuality_counter.items() if freq < eventuality_threshold_to_conceptualize]
            )
            for filtered_eid in filtered_eids:
                eid2sids.pop(filtered_eid)
                eid2eventuality.pop(filtered_eid)
                total_eventuality -= eventuality_counter.pop(filtered_eid)
            self.logger.info(
                "%d eventualities (%d unique) will be conceptualized by %s." %
                (total_eventuality, len(eid2eventuality), self.opt.concept_method)
            )

            # constructing conceptualizers is memory-consuming so that we do not use multiprocessing
            cid2concept, concept_instance_pairs, cid_to_filter_score = \
                conceptualize_eventualities(self.conceptualizer, eid2eventuality.values())
            self.logger.info("%d unique concepts  will be inserted into the concept KG." % (len(cid2concept)))
            self.logger.info(
                "%d unique concept-instance pairs  will be inserted into the concept KG." %
                (len(concept_instance_pairs))
            )

            self.logger.info("Building the concept KG.")
            if not os.path.exists(self.opt.concept_kg_dir):
                os.mkdir(self.opt.concept_kg_dir)
            concept_conn = ASERConceptConnection(os.path.join(self.opt.concept_kg_dir, "concept.db"), mode="memory")

            self.logger.info("Inserting concepts.")
            concept_conn.insert_concepts(cid2concept.values())

            self.logger.info("Inserting concept-instance pairs.")
            concept_conn.insert_concept_instance_pairs(concept_instance_pairs)

            rid2relation = build_concept_relations(concept_conn, rid2relation.values())
            self.logger.info(
                "%d unique concept-concept relations will be inserted into the concept KG." % (len(rid2relation))
            )

            concept_conn.insert_relations(rid2relation.values())
            concept_conn.close()
            self.logger.info("Done.")

        self.logger.info("Ending.")
