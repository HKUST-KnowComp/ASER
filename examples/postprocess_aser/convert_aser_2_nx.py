import time

import argparse
import networkx as nx
from tqdm import tqdm
from itertools import chain
from kg_connection import ASERKGConnection


parser = argparse.ArgumentParser()

parser.add_argument("--kg_path", type=str, default="/home/data/corpora/aser/database/filter_2.0/2/KG.db")
parser.add_argument("--networkx_path", type=str, default="/home/data/zwanggy/aser_graph/test.pickle")

args = parser.parse_args()
print(args)
st = time.time()

kg_conn = ASERKGConnection(args.kg_path,
                           mode='memory', grain="words", load_types=["merged_eventuality", "words", "eventuality"])

print('time:', time.time() - st)

# merge nodes with the same name
G_aser = nx.DiGraph()
for node in tqdm(kg_conn.merged_eventuality_cache):
    G_aser.add_node(node,
                    freq=kg_conn.get_event_frequency(node))
                    # info=kg_conn.get_event_info(node))

# traverse all the nodes, and get it's all neighbors
gather_relations = lambda key, successor_dict: \
    list(set(chain(*[list(item[1].keys()) for item in successor_dict[key]])))

gather_weights = lambda key, successor_dict, rels: \
    dict([(r, sum([item[1].get(r, 0) for item in successor_dict[key]])) for r in rels])

for node in tqdm(kg_conn.merged_eventuality_cache):
    successor_dict = kg_conn.merged_eventuality_relation_cache["head_words"].get(node, {})
    selected_tails = [(key, gather_weights(key, successor_dict, gather_relations(key, successor_dict))) \
                      for key, relations in successor_dict.items()]  # tail, relation
    for key, rels in selected_tails:
        G_aser.add_edge(node, key,
                        relations=rels, )

nx.write_gpickle(G_aser, args.networkx_path)
