import re
import argparse
import networkx as nx
from tqdm import tqdm
from itertools import chain
from nltk.corpus import stopwords
from aser_to_glucose import generate_aser_to_glucose_dict

top1000_kept_nodes = ['PersonX be sorry', 'PersonX die', 'PersonY be glad', 'PersonY love PersonX',
                      'PersonX love PersonY', 'PersonX do not care', 'PersonY do not care', 'PersonY be pretty sure',
                      'PersonX be pretty sure', 'PersonX wake up', 'PersonY wake up', 'PersonX smile', 'PersonY smile',
                      'PersonX be interested', 'PersonY be interested', 'PersonY be so glad', 'PersonX be so glad',
                      'PersonX sleep', 'PersonY sleep', 'PersonY like PersonX', 'PersonX like PersonY',
                      'PersonX go out',
                      'PersonY go out', 'PersonY like PersonX', 'PersonX like PersonY', 'PersonX go back',
                      'PersonY go back',
                      'PersonX be hungry', 'PersonY be hungry', 'PersonY kill PersonX', 'PersonX kill PersonY',
                      'PersonY meet PersonX', 'PersonX meet PersonY', 'PersonX be ready', 'PersonY be ready',
                      'PersonY apologize', 'PersonX apologize',
                      'PersonX feel better', 'PersonY feel better', 'the food be delicious', 'PersonY be pregnant',
                      'PersonX be pregnant',
                      'PersonY sit down', 'PersonX sit down', 'PersonX need to know', 'PersonY need to know',
                      'PersonY leave PersonX', 'PersonX leave PersonY',
                      'PersonX lie', 'PersonY lie', 'PersonY can understand', 'PersonX can understand',
                      'PersonX be sick',
                      'PersonY be sick',
                      'PersonX have time', 'PersonY have time', 'PersonX be curious', 'PersonY be curious',
                      'PersonY go away', 'PersonX go away', 'PersonX get up', 'PersonY get up',
                      'PersonX come home', 'PersonY come home', 'PersonY call PersonX', 'PersonX call PersonY',
                      'PersonX be a child', 'PersonY be a child', 'PersonY feel bad', 'PersonX feel bad',
                      'PersonY be crazy', 'PersonX be crazy', 'PersonX come out', 'PersonY come out',
                      'PersonY be worry', 'PersonX be worry', 'PersonX be marry', 'PersonY be marry',
                      'PersonX need PersonY', 'PersonY need PersonX', 'PersonX be drunk', 'PersonY be drunk',
                      'PersonX be okay', 'PersonY be okay', 'PersonX get out', 'PersonY get out',
                      'PersonX can not see', 'PersonY can not see', 'PersonX go home', 'PersonY go home',
                      'PersonY be surprise', 'PersonX be surprise', 'PersonX agree with PersonY',
                      'PersonY agree with PersonX']


def reverse_px_py(original: str):
    return original.replace("PersonX", "[PX]").replace("PersonY", "[PY]").replace("[PX]", "PersonY").replace(
        "[PY]", "PersonX")


def merge_rel_dict(d1: dict, d2: dict):
    d_merge = {}
    for key in set(d1.keys()) | set(d2.keys()):
        d_merge[key] = d1.get(key, 0) + d2.get(key, 0)
    return d_merge


def add_update_nodes(G, new_G, node_name, new_node_name):
    node_attr = G.nodes[node_name]
    if new_G.has_node(new_node_name):
        new_G.nodes[new_node_name]["freq"] = new_G.nodes[new_node_name]["freq"] + node_attr["freq"]
        new_G.nodes[new_node_name]["info"] = new_G.nodes[new_node_name]["info"] | set(node_attr["info"])
    else:
        new_G.add_node(new_node_name, freq=node_attr["freq"], info=set(node_attr["info"]))


def add_update_edges(new_G, edge_attr, con_head, con_tail):
    relations = edge_attr["relations"]
    if new_G.has_edge(con_head, con_tail):
        new_G.add_edge(con_head, con_tail,
                       relation=merge_rel_dict(new_G[con_head][con_tail]["relations"],
                                               relations)
                       )
    else:
        new_G.add_edge(con_head, con_tail, relation=relations)


def get_normalized_graph(G: nx.DiGraph):
    G_conceptualized = nx.DiGraph()

    for head, tail, edge_attr in tqdm(G.edges.data()):
        _, con_head, con_tail, _ = generate_aser_to_glucose_dict(head, tail, True)
        con_head_reverse, con_tail_reverse = reverse_px_py(con_head), reverse_px_py(con_tail)

        assert len(head) > 0 and len(tail) > 0
        assert len(con_head) > 0 and len(con_tail) > 0
        assert len(con_head_reverse) > 0 and len(con_tail_reverse) > 0

        add_update_nodes(G, G_conceptualized, head, con_head)
        add_update_nodes(G, G_conceptualized, tail, con_tail)
        add_update_nodes(G, G_conceptualized, head, con_head_reverse)
        add_update_nodes(G, G_conceptualized, tail, con_tail_reverse)

        add_update_edges(G_conceptualized, edge_attr, con_head, con_tail)
        add_update_edges(G_conceptualized, edge_attr, con_head_reverse, con_tail_reverse)

    return G_conceptualized


def find_URL(string):
    URL_REGEX = re.compile(
        r'(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))',
        re.IGNORECASE
    )
    url = re.findall(URL_REGEX,string)
    return bool([x[0] for x in url])


def find_number(string):
    return any(char.isdigit() for char in string)


def filter_main(args):
    G_aser = nx.read_gpickle(args.input_nx_path)
    
    logs = []
    def print_log(printstr):
        print(printstr)
        logs.append(printstr)

    if args.pronoun_norm:
        G_aser_conceptualized = get_normalized_graph(G_aser)
        print_log("Before pronoun normalization: Number of Edges: {}\tNumber of Nodes: {}".format(
            len(G_aser.edges),
            len(G_aser.nodes)))
        print_log("After pronoun normalization: Number of Edges: {}\tNumber of Nodes: {}".format(
            len(G_aser_conceptualized.edges),
            len(G_aser_conceptualized.nodes)))
        print()
        G_aser = G_aser_conceptualized

    if args.filter_top1000_degree:
        deg_nodes_sorted, degs = zip(*sorted(G_aser.degree(), key=lambda x: x[1], reverse=True))
        G_aser_top1000_filter = G_aser.copy()
        all_filtered_nodes = list(set(deg_nodes_sorted[:1000]) - set(top1000_kept_nodes))
        G_aser_top1000_filter.remove_nodes_from(all_filtered_nodes)
        G_aser = G_aser_top1000_filter
        print_log("{} nodes filtered in top1000-degree step".format(len(all_filtered_nodes)))
        print_log("After top1000-degree filtering: Number of Edges: {}\tNumber of Nodes: {}".format(
            len(G_aser.edges),
            len(G_aser.nodes)
        ))
        print()

    if args.filter_by_key_words:
        include_filter = ["there", "here", "those", "these", "that", "this"]
        starts_with_filter = ["be", "how", "t", "what", "when", "where", "which", "who", "why",
                              "do", "-lsb-", "whatever", 'whose', 'whom', "if", ]
        stopword_list = stopwords.words("english") + ["'s", "'d", "'ll", "'re", "'ve", "be"]
        all_filtered_nodes = []
        for node in tqdm(G_aser.nodes, "filtering nodes by key words rules"):
            tokens = node.split()
            # rule out nodes starting with tokens in `starts_with_filter`
            if tokens[0] in starts_with_filter:
                all_filtered_nodes.append(node)
                continue
            # rule out nodes that contain one of tokens in `include_filter`
            if any(t in include_filter for t in tokens):
                all_filtered_nodes.append(node)
                continue
            # rule out nodes whose all tokens are stopwords
            if all(t in stopword_list for t in tokens):
                all_filtered_nodes.append(node)
            # rule out simple nodes, which is usually very frequent, like "I do", "You know"
#             if len(tokens) <= 2 and any(kw in tokens for kw in ["say", "do", "know", "tell", "think", ]):
            if len(tokens) <= 2 and any(kw in tokens for kw in ["say", "do", "tell"]):
                all_filtered_nodes.append(node)
        G_aser_rules = G_aser.copy()
        G_aser_rules.remove_nodes_from(all_filtered_nodes)
        G_aser = G_aser_rules
        print_log("{} nodes filtered in key words step".format(len(all_filtered_nodes)))
        print_log("After key word filtering: Number of Edges: {}\tNumber of Nodes: {}".format(
            len(G_aser.edges),
            len(G_aser.nodes)
        ))
        print()

    if args.filter_by_degree:
        low_threshold, high_threshold = args.low_degree_threshold, args.high_degree_threshold
        low_in_node_list, high_in_node_list = [], []
        low_out_node_list, high_out_node_list = [], []

        for node in tqdm(G_aser.nodes, "filtering by in-degree/out-degree threshold"):
            in_degree, out_degree = G_aser.in_degree[node], G_aser.out_degree[node]
            if in_degree <= low_threshold:
                low_in_node_list.append(node)
            if in_degree >= high_threshold:
                high_in_node_list.append(node)
            if out_degree <= low_threshold:
                low_out_node_list.append(node)
            if out_degree >= high_threshold:
                high_out_node_list.append(node)
        print_log("{} nodes' in-degree lower than {}".format(len(low_in_node_list), low_threshold))
        print_log("{} nodes' in-degree higher than {}".format(len(high_in_node_list), high_threshold))
        print_log("{} nodes' out-degree lower than {}".format(len(low_out_node_list), low_threshold))
        print_log("{} nodes' out-degree higher than {}".format(len(high_out_node_list), high_threshold))
        G_aser_by_degree = G_aser.copy()
        all_filtered_nodes = set(chain(low_in_node_list, high_in_node_list, low_out_node_list, high_out_node_list))
        G_aser_by_degree.remove_nodes_from(all_filtered_nodes)
        G_aser = G_aser_by_degree
        print_log("{} nodes filtered in low/high in/out-degree step".format(len(all_filtered_nodes)))
        print_log("After low/high in/out-degree filtering: Number of Edges: {}\tNumber of Nodes: {}".format(
            len(G_aser.edges),
            len(G_aser.nodes)
        ))
        print()

    if args.filter_by_URL or args.filter_by_number:
        nodes_with_URL, nodes_with_number = [], []
        for node in tqdm(G_aser.nodes, "filtering by URL or(/and) number"):
            if args.filter_by_URL and find_URL(node):
                nodes_with_URL.append(node)
            if args.filter_by_number and find_number(node):
                nodes_with_number.append(node)
        print_log("{} nodes contain URL".format(len(nodes_with_URL)))
        print_log("{} nodes contain number".format(len(nodes_with_number)))
        G_aser_by_URL_number = G_aser.copy()
        all_filtered_nodes = set(chain(nodes_with_URL, nodes_with_number))
        G_aser_by_URL_number.remove_nodes_from(all_filtered_nodes)
        G_aser = G_aser_by_URL_number
        print_log("{} nodes filtered in URL/number step".format(len(all_filtered_nodes)))
        print_log("After URL/number filtering: Number of Edges: {}\tNumber of Nodes: {}".format(
            len(G_aser.edges),
            len(G_aser.nodes)
        ))
        print()
    
    if len(logs)>0:
        print('Saving logs to file: logs.txt')
        with open('logs.txt', 'w') as f:
            f.write('\n'.join(logs))
    nx.write_gpickle(G_aser, args.output_nx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_nx_path", type=str, default="/home/data/zwanggy/aser_graph/ASER_2_no_filter.pickle")
    parser.add_argument("--output_nx_path", type=str,
                        default='/home/data/zwanggy/aser_graph/ASER_2_person_norm.pickle')
    parser.add_argument("--pronoun_norm", action="store_true",
                        help="replace pronouns, like I, you, my, he, she, and so on, with PersonX/Y/Z")
    parser.add_argument("--filter_top1000_degree", action="store_true",
                        help="filter 1000 nodes with highest frequencies")
    parser.add_argument("--filter_by_key_words", action="store_true",
                        help="filter nodes that satisfies four key words rules")
    parser.add_argument("--filter_by_degree", action="store_true",
                        help="filter nodes with too low/high in-degree/out-degree")
    parser.add_argument("--low_degree_threshold", type=int, default=5,
                        help="filter nodes with in-degree/out-degree lower than (or equal to) this threshold." +
                             "It is useful only if `filter_by_degree` is True")
    parser.add_argument("--high_degree_threshold", type=int, default=1000,
                        help="filter nodes with in-degree/out-degree higher than (or equal to) this threshold." +
                             "It is useful only if `filter_by_degree` is True")
    parser.add_argument("--filter_by_URL", action="store_true", help="filter nodes that contain URL")
    parser.add_argument("--filter_by_number", action="store_true", help="filter nodes that contain number")
    args = parser.parse_args()
    print(args)
    filter_main(args)
