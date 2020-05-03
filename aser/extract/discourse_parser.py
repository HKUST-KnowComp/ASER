import re
import os
try:
    import ujson as json
except:
    import json
import bisect
import time
import subprocess
import numpy as np
import scipy
import pickle
from collections import deque
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from scipy import sparse
from copy import deepcopy, copy
from ete3 import Tree
from itertools import chain
from aser.extract.utils import PUNCTUATION_SET, CLAUSE_SEPARATOR_SET
from aser.extract.utils import index_from, get_clauses, strip_punctuation, get_prev_token_index, get_next_token_index


######################################################
##################    Feature    #####################
######################################################
class Feature:
    #featDict : 3:1, 10:0.5, 7:1
    def __init__(self, dimension, feat_dict, name=""):
        self.dimension = dimension # feature dimension
        # self.feat_string = self.featdict2str(feat_dict) # feature string: "3:1 7:1 10:0.5"
        self.feat_dict = feat_dict
        self.name = name # feature name


    def to_str(self, zero_based=False):
        if zero_based:
            feats = [str(key)+":"+str(self.feat_dict[key]) for key in sorted(self.feat_dict.keys())]
        else:
            feats = [str(key+1)+":"+str(self.feat_dict[key]) for key in sorted(self.feat_dict.keys())]
        return " ".join(feats)

    def to_csr(self):
        data = list(self.feat_dict.values())
        indptr = [0, len(data)]
        indices = list(self.feat_dict.keys())
        csr =  sparse.csr_matrix((data, indices, indptr), shape=(1, self.dimension), dtype=np.float)
        return csr

    def to_dict(self):
        x = {"name": self.name, "dimension": self.dimension, "feat_dict": self.feat_dict}
        return x

    @staticmethod
    def from_dict(x):
        f = Feature(x["dimension"], x["feat_dict"], x["name"])
        return f

    @staticmethod
    def get_feature_by_list(feat_list, name=""):
        feat_dict = {}
        for idx, item in enumerate(feat_list):
            feat_dict[idx] = item
        return Feature(len(feat_list), feat_dict, name)

    @staticmethod
    def get_feature_by_feat(feat2idx, feat, name=""):
        feat_dict = {}
        idx = feat2idx.get(feat, -1)
        if idx != -1:
            feat_dict[idx] = 1
        return Feature(len(feat2idx), feat_dict, name)

    @staticmethod
    def get_feature_by_feat_list(feat2idx, feat_list, name=""):
        feat_dict = {}
        for feat in feat_list:
            idx = feat2idx.get(feat, -1)
            if idx != -1:
                feat_dict[idx] = 1
        return Feature(len(feat2idx), feat_dict, name)

    @staticmethod
    def merge_features(feature_list, name=""):
        dimension = 0
        feat_dict = {}
        for feature in feature_list:
            # if len(feature.feat_dict) > 0 and max(feature.feat_dict.keys()) >= feature.dimension:
            #     raise ValueError(max(feature.feat_dict.keys()), feature.dimension)
            for key, value in feature.feat_dict.items():
                feat_dict[key+dimension] = value
            dimension += feature.dimension
        return Feature(dimension, feat_dict, name)


######################################################
##################     Syntax    #####################
######################################################
class SyntaxTree:
    def __init__(self, parse_tree="()"):
        newick_text = self.to_newick_format(parse_tree)
        
        if newick_text == None:
            self.tree = None
            self.leaves = list()
        else:
            try:
                self.tree = Tree(newick_text, format=1)
                if self.tree:
                    self.leaves = self.tree.get_leaves()
                    for idx, leaf in enumerate(self.leaves):
                        leaf.add_feature("index", idx)
                else:
                    self.leaves = list()
            except BaseException as e:
                print(e)
                self.tree = None
                self.leaves = list()

    def print_tree(self):
        print(self.tree.get_ascii(show_internal=True))

    def get_node_path_to_root(self, node):
        path = ""
        while (not node.is_root()):
            path += node.name + "-->"
            node = node.up
        path += node.name
        return path

    def get_leaf_node_by_token_index(self, token_index):
        if token_index < len(self.leaves):
            return self.leaves[token_index]
        else:
            return None

    def get_self_category_node_by_token_indices(self, token_indices):
        # retrieve common_ancestor according token indices
        if len(token_indices) == 1:
            return self.get_leaf_node_by_token_index(token_indices[0]).up
        else:
            return self.get_common_ancestor_by_token_indices(token_indices)

    def get_common_ancestor_by_token_indices(self, token_indices):
        len_leaves = len(self.leaves)
        leaves = [self.leaves[token_index] for token_index in token_indices if token_index < len_leaves]
        return self.tree.get_common_ancestor(leaves)

    def get_left_sibling_category_node_by_token_indices(self, token_indices):
        self_category_node = self.get_self_category_node_by_token_indices(token_indices)

        if self_category_node.up == None:
            return None

        children = self_category_node.up.get_children()
        index = children.index(self_category_node)
        if index == 0:
            return None
        else:
            return children[index-1]

    def get_right_sibling_category_node_by_token_indices(self, token_indices):
        self_category_node = self.get_self_category_node_by_token_indices(token_indices)

        if self_category_node.up == None:
            return None

        children = self_category_node.up.get_children()
        index = children.index(self_category_node)
        if index == len(children) - 1:
            return None
        else:
            return children[index+1]

    def get_parent_category_node_by_token_indices(self, token_indices):
        self_category_node = self.get_self_category_node_by_token_indices(token_indices)
        return self_category_node.up

    def get_subtree_by_token_indices(self, token_indices):
        if self.tree is None:
            return self

        if isinstance(token_indices, (list, tuple)):
            token_indices = set(token_indices)
            
        if len(token_indices) == 0:
            return SyntaxTree()
        elif len(token_indices) == len(self.leaves):
            return self
        else:
            kept_nodes = set()
            kept_leaves = set()
            for idx, leaf in enumerate(self.leaves):
                if idx in token_indices:
                    kept_nodes.add(leaf)
                    kept_leaves.add(leaf)
                    node = leaf.up
                    while node and node not in kept_nodes:
                        kept_nodes.add(node)
                        node = node.up
            
            # prune
            subtree = SyntaxTree()
            subtree.tree = copy(self.tree)

            queue1, queue2 = deque([self.tree]), deque([subtree.tree])
            while len(queue1) > 0 and len(queue2) > 0:
                old_node, new_node = queue1.popleft(), queue2.popleft()
                new_node._children = list()
                for old_child in old_node.get_children():
                    if old_child in kept_nodes:
                        new_child = copy(old_child)
                        new_node.add_child(new_child)
                        new_child.up = new_node
                        if old_child not in kept_leaves:
                            queue1.append(old_child)
                            queue2.append(new_child)
            return subtree

    def to_newick_format(self, parse_tree):
        # replace `<ref>`
        parse_tree = re.sub(r"<ref(.*?)>", "<ref>", parse_tree)

        # replace `url`
        parse_tree = re.sub(r"(?:(?:https?|ftp):\/\/)?[\w/\-?=%&.]+\.[\w/\-?=&%.]+", "<url>", parse_tree)
        parse_tree = re.sub(r"<url>[\(\)\[\]]*<url>", "<url>", parse_tree)

        # replace `,`, `:`, `;`
        parse_tree = parse_tree.replace(",", "*COMMA*")
        parse_tree = parse_tree.replace(":", "*COLON*")
        parse_tree = parse_tree.replace(";", "*SEMICOLON*")

        tree_list = self.load_syntax_tree(parse_tree)
        if len(tree_list) == 0:
            return None
        tree = tree_list[np.argmax([len(t) for t in tree_list])]
        s = self.syntax_tree_to_newick(tree)
        s = s.replace(",)",")")
        if s[-1] == ",":
            s = s[:-1] + ";"
        return s

    def load_syntax_tree(self, text):
        stack = []
        # text = text.replace("ROOT", "", 1)
        text = text.replace("(", " ( ")
        text = text.replace(")", " ) ")
        text = re.sub(r"\s+", " ", text)
        # text = re.sub(r"^\s*\(\s*\(\s*|\s*\)\s*\)\s*$", "", text)
        for c in text.strip().split(" "):
            if c == ")":
                node = list()
                while len(stack) > 0:
                    popped = stack.pop()
                    if popped == "(":
                        break
                    node.append(popped)
                if len(node) == 0:
                    return []
                elif len(node) == 1:
                    stack.append(node[0])
                else:
                    node.reverse()
                    stack.append(node)
            else:
                stack.append(c)
        return stack

    def syntax_tree_to_newick(self,syntax_tree):
        s = "("
        for child in syntax_tree[1:]:
            if not isinstance(child,list):
                s += child
            else:
                s += self.syntax_tree_to_newick(child)
        s += ")" + str(syntax_tree[0]) + ","
        return s

    def get_internal_node_location(self, node):
        # retrieve internal node location without the pos tag
        if len(node.get_children()) > 1:
            child1 = node.get_children()[0]
            child2 = node.get_children()[1]
            # move to leaf
            while not child1.is_leaf():
                child1 = child1.get_children()[0]
            while not child2.is_leaf():
                child2 = child2.get_children()[0]
            index1 = self.leaves.index(child1)
            index2 = self.leaves.index(child2)
            return [index1, index2]
        if len(node.get_children()) == 1:
            child1 = node.get_children()[0]
            # move to leaf
            while not child1.is_leaf():
                child1 = child1.get_children()[0]
            index1 = self.leaves.index(child1)
            return [index1]
            
    def get_node_by_internal_node_location(self, location):
        if len(location) > 1:
            nodes = list()
            for token_index in location:
                node = self.get_leaf_node_by_token_index(token_index)
                nodes.append(node)
            return self.tree.get_common_ancestor(nodes)
        if len(location) == 1:
            return self.get_leaf_node_by_token_index(location[0]).up.up

    def get_right_siblings(self, node):
        if node.is_root():
            return list()
        children = node.up.get_children()
        index = children.index(node)
        return children[index+1:]

    def get_left_siblings(self, node):
        if node.is_root():
            return list()
        children = node.up.get_children()
        index = children.index(node)
        return children[0:index]

    def get_siblings(self, node):
        if node.is_root():
            return list()
        siblings = list()
        children = node.up.get_children()
        children.remove(node)
        return children

    def get_relative_position(self, node1, node2):
        if node1 == node2 or node1.is_root or node2.is_root:
            return "middle"

        common_ancestor = self.tree.get_common_ancestor([node1, node2])
        if common_ancestor == node1 or common_ancestor == node2:
            return "middle"
        find_node1, find_node2 = False, False
        for node in common_ancestor.traverse(strategy="preorder"):
            if node == node1:
                find_node1 = True
                break
            if node == node2:
                find_node2 = True
                break
        if find_node1:
            return "right"
        if find_node2:
            return "left"
        return "middle"

    def get_node_to_node_path(self, node1, node2):
        common_ancestor = self.tree.get_common_ancestor([node1, node2])

        path = ""
        # node1->common_ancestor
        temp = node1
        while temp != common_ancestor:
            path += temp.name +">"
            temp = temp.up
        path += common_ancestor.name
        ## common_ancestor -> node
        p = ""
        temp = node2
        while temp != common_ancestor:
            p = "<" + temp.name + p
            temp = temp.up
        path += p

        return path

    def get_leaves_indices(self, node):
        # retrieve the indices of leaves of the node
        indices = [self.leaves.index(leaf) for leaf in node.get_leaves()]
        return indices

def get_compressed_path(path):
    path = path.split("-->")
    compressed_path = list()
    for idx in range(len(path)-1):
        if path[idx] != path[idx+1]:
            compressed_path.append(path[idx])
    if len(path) > 0:
        if len(compressed_path) > 0:
            if path[-1] != compressed_path[-1]:
                compressed_path.append(path[-1])
        else:
            compressed_path.append(path[-1])
    return "-->".join(compressed_path)


######################################################
#############    Connective Extractor    #############
######################################################
class ConnectiveExtractor:
    def __init__(self, **kw):
        try:
            discourse_path = os.path.join(os.path.dirname(__file__), "discourse")
        except:
            discourse_path = os.path.join("aser", "extract", "discourse")

        with open(kw.get("exp_conn_file", os.path.join(discourse_path, "conn_feats", "exp_conn.txt")), "r") as f:
            self.sorted_conn = list()
            for idx, line in enumerate(f):
                line = line.rstrip()
                if line:
                    self.sorted_conn.append(line)
            self.sorted_conn.sort()

        for feat in [
            "cpos", "prev_conn", "prevpos", "prevpos_cpos", 
            "conn_next", "nextpos", "cpos_nextpos", 
            "cparent_to_root_path", "compressed_cparent_to_root_path", 
            "self_category", "parent_category", "left_category", "right_category", 
            "conn_self_category", "conn_parent_category", "conn_left_category", "conn_right_category", 
            "self_category_parent_category", "self_category_right_category", "self_category_left_category", 
            "parent_category_left_category", "parent_category_right_category", "left_category_right_category", 
            "conn_lower", "conn", 
            "cparent_to_root_path_node_name", "conn_right_ctx", "conn_parent_ctx"]:
            feat_file = feat + "feat_file"
            with open(kw.get(feat_file, os.path.join(discourse_path, "conn_feats", feat + ".txt")), "r") as f:
                x_dict = dict()
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    if line:
                        x_dict[line] = idx
                setattr(self, feat + "_dict", x_dict)

        conn_model_file = kw.get("conn_model_file", os.path.join(discourse_path, "conn_extractor.pkl"))
        with open(conn_model_file, "rb") as f:
            self.conn_model = pickle.load(f)

    # def help_func(self, doc_parsed_result, syntax_tree_cache=None):
    #     doc_connectives = list() # [[sent_idx, connective, indices], ...]
    #     for sent_idx, sent_parsed_result in enumerate(doc_parsed_result):
    #         sent_connectives = self._extract_connectives(sent_parsed_result)
    #         for conn_idx, connective in enumerate(sent_connectives):
    #             connective["sent_idx"] = sent_idx
    #             doc_connectives.append(connective)

    #     doc_conn_feats = self._generate_connective_features(doc_parsed_result, doc_connectives, syntax_tree_cache)

    #     return doc_connectives, doc_conn_feats

    def extract(self, doc_parsed_result, syntax_tree_cache=None):
        if len(doc_parsed_result) == 0:
            return list()

        doc_connectives = list() # [[sent_idx, connective, indices], ...]
        for sent_idx, sent_parsed_result in enumerate(doc_parsed_result):
            sent_connectives = self._extract_connectives(sent_parsed_result)
            for conn_idx, connective in enumerate(sent_connectives):
                connective["sent_idx"] = sent_idx
                doc_connectives.append(connective)
        
        if len(doc_connectives) == 0:
            return list()
        
        doc_conn_feats = self._generate_connective_features(doc_parsed_result, doc_connectives, syntax_tree_cache)
        doc_conn_labels = self._classify_connectives(doc_parsed_result, doc_conn_feats)
        
        return [c for c, l in zip(doc_connectives, doc_conn_labels) if l[1]]

    def _extract_connectives(self, sent_parsed_result):
        return sorted(self._extract_connectives_by_tokens(sent_parsed_result["tokens"]), key=lambda x: x["connective"])
    
    def _extract_connectives_by_tokens(self, tokens):
        all_connectives = list()
        tokens = [t.lower() for t in tokens]
        for t_idx, token in enumerate(tokens):
            c_idx = bisect.bisect_left(self.sorted_conn, token)
            while c_idx < len(self.sorted_conn):
                conn = self.sorted_conn[c_idx]
                c_idx += 1
                if len(conn) < len(token):
                    break
                elif not conn.startswith(token):
                    break

                if ".." in conn:
                    conn_lists = [c.split() for c in conn.split("..")] # c1..c2..
                    if conn_lists[0][0] != token:
                        break
                    if len(conn_lists[0]) + t_idx <= len(tokens):
                        # check conn_lists[0]
                        match = True
                        for w_idx, c in enumerate(conn_lists[0]):
                            if tokens[w_idx + t_idx] != c:
                                match = False
                                break
                        if not match:
                            continue
                        indices = list(range(t_idx, t_idx+len(conn_lists[0])))

                        # check conn_lists[1]
                        for t_idx in index_from(tokens, conn_lists[1][0], start_from=t_idx):
                            match = False
                            if len(conn_lists[1]) + t_idx <= len(tokens):
                                match = True
                                for w_idx, c in enumerate(conn_lists[1]):
                                    if tokens[w_idx + t_idx] != c:
                                        match = False
                                        break
                                if match:
                                    all_connectives.append(
                                        {"connective": conn, "indices": indices + list(range(t_idx, t_idx+len(conn_lists[1])))})
                else:
                    conn_list = conn.split()
                    if conn_list[0] != token:
                        break
                    if len(conn_list) + t_idx <= len(tokens):
                        match = True
                        for w_idx, c in enumerate(conn_list):
                            if tokens[w_idx + t_idx] != c:
                                match = False
                                break
                        if match:
                            all_connectives.append(
                                {"connective": conn, "indices": list(range(t_idx, t_idx+len(conn_list)))})
        # filter shorter and duplicative conn
        all_connectives.sort(key=lambda x: (-len(x["indices"]), -x["indices"][0]))
        filtered_connectives = list()
        used_indices = set()
        for conn_indices in all_connectives:
            indices = conn_indices["indices"]
            # duplicated = len(set(indices) & used_indices) > 0
            duplicated = False
            for idx in indices:
                if idx in used_indices:
                    duplicated = True
                    break
            if not duplicated:
                used_indices.update(indices)
                filtered_connectives.append(conn_indices)
        return filtered_connectives

    def _generate_connective_features(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if syntax_tree_cache is None:
            syntax_tree_cache = dict()
            
        doc_conn_feats = list()
        for conn_idx, connective in enumerate(doc_connectives):
            sent_idx, indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx]
            sent_len = len(sent_parsed_result["tokens"])

            if sent_idx in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx]
            else:
                syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])
            
            # conn
            conn = " ".join([sent_parsed_result["tokens"][idx] for idx in indices])
            cpos = "_".join([sent_parsed_result["pos_tags"][idx] for idx in indices])

            # prev
            prev_sent_idx, prev_idx = get_prev_token_index(doc_parsed_result, sent_idx, indices[0])
            if prev_sent_idx != -1:
                prev, prevpos = doc_parsed_result[prev_sent_idx]["tokens"][prev_idx], doc_parsed_result[prev_sent_idx]["pos_tags"][prev_idx]
            else:
                prev, prevpos = "NONE", "NONE"
                    
            # next
            next_sent_idx, next_idx = get_next_token_index(doc_parsed_result, sent_idx, indices[-1])
            if next_sent_idx != -1:
                next, nextpos = doc_parsed_result[next_sent_idx]["tokens"][next_idx], doc_parsed_result[next_sent_idx]["pos_tags"][next_idx]
            else:
                next, nextpos = "NONE", "NONE"
                    
            # cparent to root
            # compressed cparent to root
            try:
                cparent_to_root_paths = list()
                for idx in indices:
                    node = syntax_tree.get_leaf_node_by_token_index(idx)
                    parent_node = node.up
                    path = syntax_tree.get_node_path_to_root(parent_node)
                    cparent_to_root_paths.append(path)
                cparent_to_root_path_node_names = chain.from_iterable([path.split("-->") for path in cparent_to_root_paths])
                cparent_to_root_path = "&".join(cparent_to_root_paths)
                compressed_cparent_to_root_path = "&".join([get_compressed_path(path) for path in cparent_to_root_paths])
            except:
                cparent_to_root_path_node_names = ["NONE_TREE"]
                cparent_to_root_path = "NONE_TREE"
                compressed_cparent_to_root_path = "NONE_TREE"

            # Pitler
            try:
                category_node = syntax_tree.get_self_category_node_by_token_indices(indices)
                self_category = category_node.name
                parent_category_node = category_node.up
                left_category_node, right_category_node = None, None
                if parent_category_node:
                    parent_category = parent_category_node.name
                    children = parent_category_node.get_children()
                    category_node_id = id(category_node)
                    for child_idx, child in enumerate(children):
                        if category_node_id == id(child):
                            if child_idx > 0:
                                left_category_node = children[child_idx-1]
                            if child_idx < len(children) - 1:
                                right_category_node = children[child_idx+1]
                    left_category = left_category_node.name if left_category_node else "NONE"
                    right_category = right_category_node.name if right_category_node else "NONE"
                else:
                    parent_category = "ROOT"
                    left_category = "NONE"
                    right_category = "NONE"

                # conn_ctx
                conn_ctx = list() # self, parent, left, right
                conn_ctx.append(category_node.name)
                conn_ctx.append(parent_category_node.name if parent_category_node else "NULL")
                conn_ctx.append(left_category_node.name if left_category_node else "NULL")
                conn_ctx.append(right_category_node.name if right_category_node else "NULL")
                conn_ctx = "-".join(conn_ctx)

                # parent_ctx
                if parent_category_node:
                    parent_ctx = list() # self, parent, children
                    parent_ctx.append(parent_category_node.name)
                    parent_ctx.append(parent_category_node.up.name if parent_category_node.up else "NULL")
                    parent_ctx.extend([child.name for child in parent_category_node.get_children()])
                    parent_ctx = "-".join(parent_ctx)
                else:
                    parent_ctx = "None"

                # left_ctx
                if left_category_node:
                    left_ctx = list() # self, parent, children
                    left_ctx.append(left_category_node.name)
                    left_ctx.append(parent_category_node.name)
                    left_ctx.extend([child.name for child in left_category_node.get_children()])
                    left_ctx = "-".join(left_ctx)
                else:
                    left_ctx = "None"
                
                # right_ctx
                if right_category_node:
                    right_ctx = list() # self, parent, children
                    right_ctx.append(right_category_node.name)
                    right_ctx.append(parent_category_node.name)
                    right_ctx.extend([child.name for child in right_category_node.get_children()])
                    right_ctx = "-".join(right_ctx)
                else:
                    right_ctx = "None"

                # right_contains_VP
                right_contains_VP = False
                if right_category_node:
                    if right_category_node.name == "VP" or right_category_node.name == "S":
                        right_contains_VP = True
                    else:
                        for node in right_category_node.get_descendants():
                            if node.name == "VP" or node.name == "S":
                                right_contains_VP = True
                                break

            except:
                self_category = "NONE_TREE"
                parent_category = "NONE_TREE"
                left_category = "NONE_TREE"
                right_category = "NONE_TREE"
                conn_ctx = "NONE_TREE"
                parent_ctx = "NONE_TREE"
                left_ctx = "NONE_TREE"
                right_ctx = "NONE_TREE"
                right_contains_VP = False

            conn_feats = list()
            # Z. Lin
            conn_feats.append(Feature.get_feature_by_feat(self.cpos_dict, cpos))
            conn_feats.append(Feature.get_feature_by_feat(self.prev_conn_dict, prev+"|"+conn))
            conn_feats.append(Feature.get_feature_by_feat(self.prevpos_dict, prevpos))
            conn_feats.append(Feature.get_feature_by_feat(self.prevpos_cpos_dict, prevpos+"|"+cpos))
            conn_feats.append(Feature.get_feature_by_feat(self.conn_next_dict, conn+"|"+next))
            conn_feats.append(Feature.get_feature_by_feat(self.nextpos_dict, nextpos))
            conn_feats.append(Feature.get_feature_by_feat(self.cpos_nextpos_dict, cpos+"|"+nextpos))
            conn_feats.append(Feature.get_feature_by_feat(self.cparent_to_root_path_dict, cparent_to_root_path))
            conn_feats.append(Feature.get_feature_by_feat(self.compressed_cparent_to_root_path_dict, compressed_cparent_to_root_path))
            # pitler
            conn_feats.append(Feature.get_feature_by_feat(self.self_category_dict, self_category))
            conn_feats.append(Feature.get_feature_by_feat(self.parent_category_dict, parent_category))
            conn_feats.append(Feature.get_feature_by_feat(self.left_category_dict, left_category))
            conn_feats.append(Feature.get_feature_by_feat(self.right_category_dict, right_category))
            conn_feats.append(Feature.get_feature_by_list([int(right_contains_VP)]))
            # conn-syn
            conn_feats.append(Feature.get_feature_by_feat(self.conn_self_category_dict, conn+"|"+self_category))
            conn_feats.append(Feature.get_feature_by_feat(self.conn_parent_category_dict, conn+"|"+parent_category))
            conn_feats.append(Feature.get_feature_by_feat(self.conn_left_category_dict, conn+"|"+left_category))
            conn_feats.append(Feature.get_feature_by_feat(self.conn_right_category_dict, conn+"|"+right_category))
            # sync-syn
            conn_feats.append(Feature.get_feature_by_feat(self.self_category_parent_category_dict, self_category+"|"+parent_category))
            conn_feats.append(Feature.get_feature_by_feat(self.self_category_right_category_dict, self_category+"|"+right_category))
            conn_feats.append(Feature.get_feature_by_feat(self.self_category_left_category_dict, self_category+"|"+left_category))
            conn_feats.append(Feature.get_feature_by_feat(self.parent_category_left_category_dict, parent_category+"|"+left_category))
            conn_feats.append(Feature.get_feature_by_feat(self.parent_category_right_category_dict, parent_category+"|"+right_category))
            conn_feats.append(Feature.get_feature_by_feat(self.left_category_right_category_dict, left_category+"|"+right_category))
            # mine
            conn_feats.append(Feature.get_feature_by_feat(self.conn_lower_dict, conn.lower()))
            conn_feats.append(Feature.get_feature_by_feat(self.conn_dict, conn))
            conn_feats.append(Feature.get_feature_by_feat_list(self.cparent_to_root_path_node_name_dict, cparent_to_root_path_node_names))
            conn_feats.append(Feature.get_feature_by_feat(self.conn_right_ctx_dict, conn+"|"+right_ctx))
            conn_feats.append(Feature.get_feature_by_feat(self.conn_parent_ctx_dict, conn+"|"+parent_ctx))
            # merge
            conn_feats = Feature.merge_features(conn_feats, 
                "%d|%d|%s" % (sent_idx, conn_idx, ",".join([str(idx) for idx in indices])))

            doc_conn_feats.append(conn_feats)
        return doc_conn_feats

    def _classify_connectives(self, doc_parsed_result, doc_conn_feats):
        # write features to a file
        if len(doc_conn_feats) == 0:
            return list()
        names = [x.name for x in doc_conn_feats]
        feats = sparse.vstack(list(map(lambda x: x.to_csr(), doc_conn_feats)))
        pred = self.conn_model.predict(feats)
        return list(zip(names, pred))


######################################################
#########    Argument Position Classifier    #########
######################################################
class ArgumentPositionClassifier:
    def __init__(self, **kw):
        try:
            discourse_path = os.path.join(os.path.dirname(__file__), "discourse")
        except:
            discourse_path = os.path.join("aser", "extract", "discourse")

        self.conn_part_dict = {"start": 0, "middle": 1, "end": 2}

        for feat in [
            "conn", "cpos", 
            "prev1", "prev1pos", "prev1_conn", "prev1pos_cpos", 
            "prev2", "prev2pos", "prev2_conn", "prev2pos_cpos", 
            "next1pos_cpos", "next2", "conn_to_root_path"]:
            feat_file = feat + "feat_file"
            with open(kw.get(feat_file, os.path.join(discourse_path, "argpos_feats", feat + ".txt")), "r") as f:
                x_dict = dict()
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    if line:
                        x_dict[line] = idx
                setattr(self, feat + "_dict", x_dict)

        argpos_model_file = kw.get("argpos_model_file", os.path.join(discourse_path, "argpos_classifier.pkl"))
        with open(argpos_model_file, "rb") as f:
            self.argpos_model = pickle.load(f)

    # def help_func(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
    #     doc_argpos_feats = self._generate_argument_position_features(doc_parsed_result, doc_connectives, syntax_tree_cache)

    #     return doc_connectives, doc_argpos_feats
    
    def classify(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        SS_connectives, PS_connectives = list(), list()
        if len(doc_connectives) == 0:
            return SS_connectives, PS_connectives

        doc_argpos_feats = self._generate_argument_position_features(doc_parsed_result, doc_connectives, syntax_tree_cache)
        doc_argpos_labels = self._classify_argument_positions(doc_parsed_result, doc_argpos_feats)
        for c, l in zip(doc_connectives, doc_argpos_labels):
            if l[1] == 1:
                PS_connectives.append(c)
            else:
                SS_connectives.append(c)
        return SS_connectives, PS_connectives

    def _generate_argument_position_features(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if syntax_tree_cache is None:
            syntax_tree_cache = dict()

        doc_argpos_feats = list()
        for conn_idx, connective in enumerate(doc_connectives):
            sent_idx, indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx]
            sent_len = len(sent_parsed_result["tokens"])

            if sent_idx in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx]
            else:
                syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])

            # conn
            conn = " ".join([sent_parsed_result["tokens"][idx] for idx in indices])
            conn_part = "middle"
            if indices[0]/sent_len <= 0.2:
                conn_part = "start"
            elif indices[0]/sent_len >= 0.8:
                conn_part = "end"
            cpos = "_".join([sent_parsed_result["pos_tags"][idx] for idx in indices])

            # prev1
            prev1_sent_idx, prev1_idx = get_prev_token_index(doc_parsed_result, sent_idx, indices[0])
            if prev1_sent_idx != -1:
                prev1, prev1pos = doc_parsed_result[prev1_sent_idx]["tokens"][prev1_idx], doc_parsed_result[prev1_sent_idx]["pos_tags"][prev1_idx]
            else:
                prev1, prev1pos = "NONE", "NONE"

            # prev2
            if prev1_sent_idx != -1:
                prev2_sent_idx, prev2_idx = get_prev_token_index(doc_parsed_result, prev1_sent_idx, prev1_idx)
                if prev2_sent_idx != -1:
                    prev2, prev2pos = doc_parsed_result[prev2_sent_idx]["tokens"][prev2_idx], doc_parsed_result[prev2_sent_idx]["pos_tags"][prev2_idx]
                else:
                    prev2, prev2pos = "NONE", "NONE"
            else:
                prev2, prev2pos = "NONE", "NONE"

            # next1
            next1_sent_idx, next1_idx = get_next_token_index(doc_parsed_result, sent_idx, indices[-1])
            if next1_sent_idx != -1:
                next1, next1pos = doc_parsed_result[next1_sent_idx]["tokens"][next1_idx], doc_parsed_result[next1_sent_idx]["pos_tags"][next1_idx]
            else:
                next1, next1pos = "NONE", "NONE"
            
            # next2
            if next1_sent_idx != -1:
                next2_sent_idx, next2_idx = get_next_token_index(doc_parsed_result, next1_sent_idx, next1_idx)
                if next2_sent_idx != -1:
                    next2, next2pos = doc_parsed_result[next2_sent_idx]["tokens"][next2_idx], doc_parsed_result[next2_sent_idx]["pos_tags"][next2_idx]
                else:
                    next2, next2pos = "NONE", "NONE"
            else:
                next2, next2pos = "NONE", "NONE"

            # conn_to_root_path
            try:
                conn_to_root_path = list()
                for idx in indices:
                    node = syntax_tree.get_leaf_node_by_token_index(idx)
                    path = syntax_tree.get_node_path_to_root(node)
                    conn_to_root_path.append(path)
                conn_to_root_path = "&".join(conn_to_root_path)
            except:
                conn_to_root_path = "NONE_TREE"

            argpos_feats = list()
            argpos_feats.append(Feature.get_feature_by_feat(self.conn_dict, conn))
            argpos_feats.append(Feature.get_feature_by_feat(self.conn_part_dict, conn_part))
            argpos_feats.append(Feature.get_feature_by_feat(self.cpos_dict, cpos))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev1_dict, prev1))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev1pos_dict, prev1pos))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev1_conn_dict, prev1+"|"+conn))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev1pos_cpos_dict, prev1pos+"|"+cpos))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev2_dict, prev2))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev2pos_dict, prev2pos))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev2_conn_dict, prev2+"|"+conn))
            argpos_feats.append(Feature.get_feature_by_feat(self.prev2pos_cpos_dict, prev2pos+"|"+cpos))
            argpos_feats.append(Feature.get_feature_by_feat(self.next1pos_cpos_dict, cpos+"|"+next1pos))
            argpos_feats.append(Feature.get_feature_by_feat(self.next2_dict, next2))
            argpos_feats.append(Feature.get_feature_by_feat(self.conn_to_root_path_dict, conn_to_root_path))
            # merge
            argpos_feats = Feature.merge_features(argpos_feats, 
                "%d|%d|%s" % (sent_idx, conn_idx, ",".join([str(idx) for idx in indices])))

            doc_argpos_feats.append(argpos_feats)
        return doc_argpos_feats
    
    def _classify_argument_positions(self, doc_parsed_result, doc_argpos_feats):
        # write features to a file
        if len(doc_argpos_feats) == 0:
            return list()
        names = [x.name for x in doc_argpos_feats]
        feats = sparse.vstack(list(map(lambda x: x.to_csr(), doc_argpos_feats)))
        pred = self.argpos_model.predict(feats)
        return list(zip(names, pred))


######################################################
#############    SS Argument Extractor    ############
######################################################
class SSArgumentExtractor:
    def __init__(self, **kw):
        try:
            discourse_path = os.path.join(os.path.dirname(__file__), "discourse")
        except:
            discourse_path = os.path.join("aser", "extract", "discourse")

        conn_category_feat_file = kw.get("conn_category_feat_file", "")
        if conn_category_feat_file:
            if conn_category_feat_file.endswith(".json"):
                with open(conn_category_feat_file, "r") as f:
                    self.conn_category_mapping = json.load(f)
            elif conn_category_feat_file.endswith(".txt"):
                with open(conn_category_feat_file, "r") as f:
                    self.conn_category_mapping = dict()
                    for line in f:
                        line = line.split("#")
                        self.conn_category_mapping[line[0].strip()] = line[1].strip()
            else:
                raise KeyError("Error: %s Not Found." % (conn_category_feat_file))
        else:
            if os.path.exists(os.path.join(discourse_path, "feats", "conn_category.json")):
                with open(os.path.join(discourse_path, "feats", "conn_category.json"), "r") as f:
                    self.conn_category_mapping = json.load(f)
            elif os.path.exists(os.path.join(discourse_path, "feats", "conn_category.txt")):
                with open(os.path.join(discourse_path, "feats", "conn_category.txt"), "r") as f:
                    self.conn_category_mapping = dict()
                    for line in f:
                        line = line.split("#")
                        self.conn_category_mapping[line[0].strip()] = line[1].strip()
                with open(os.path.join(discourse_path, "feats", "conn_category.json"), "w") as f:
                    json.dump(self.conn_category_mapping, f)
            else:
                raise KeyError("Error: %s Not Found." % (conn_category_feat_file))
                

        self.conn_category_dict = {"subordinator": 0, "coordinator": 1, "adverbial": 2}
        self.conn_nt_position_dict = {"right": 0, "left": 1}
        
        for feat in [
            "conn", "conn_lower", "nt_ctx", 
            "conn_nt_path", "conn_nt_path_left_number"]:
            feat_file = feat + "feat_file"
            with open(kw.get(feat_file, os.path.join(discourse_path, "ss_arg_feats", feat + ".txt")), "r") as f:
                x_dict = dict()
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    if line:
                        x_dict[line] = idx
                setattr(self, feat + "_dict", x_dict)

        ss_arg_model_file = kw.get("ss_arg_model_file", os.path.join(discourse_path, "ss_arg_classifier.pkl"))
        with open(ss_arg_model_file, "rb") as f:
            self.ss_arg_model = pickle.load(f)

    # def help_func(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
    #     if syntax_tree_cache is None:
    #         syntax_tree_cache = dict()
        
    #     parallel_connectives, non_parallel_connectives = self._divide_connectives_parallel(doc_connectives)

    #     doc_ss_arg_conns, doc_ss_arg_feats = list(), list()
    #     for conn_idx, connective in enumerate(non_parallel_connectives):
    #         sent_idx, conn_indices = connective["sent_idx"], connective["indices"]
    #         sent_parsed_result = doc_parsed_result[sent_idx]

    #         if sent_idx in syntax_tree_cache:
    #             syntax_tree = syntax_tree_cache[sent_idx]
    #         else:
    #             syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])
            
    #         conn = " ".join([sent_parsed_result["tokens"][idx] for idx in conn_indices])
    #         conn_lower = conn.lower()
    #         conn_category = self.conn_category_mapping[conn_lower]
    #         cpos = "_".join([sent_parsed_result["pos_tags"][idx] for idx in conn_indices])
    #         try:
    #             conn_node = syntax_tree.get_self_category_node_by_token_indices(conn_indices)
    #         except BaseException as e:
    #             print(sent_parsed_result)
    #             raise e
            
    #         left_number, right_number = 0, 0
    #         if conn_node.up:
    #             children = conn_node.up.get_children()
    #             for child_idx, child in enumerate(children):
    #                 if conn_node == child:
    #                     left_number, right_number = child_idx, len(children)-1-child_idx
    #                     break
            
    #         constituents = self._get_constituents(connective, syntax_tree)
    #         constituents.sort(key=lambda x: x["indices"][0])

    #         for nt_idx, constituent in enumerate(constituents):
    #             constituent_node = constituent["node"]
    #             parent_constituent_node, left_constituent_node, right_constituent_node = constituent_node.up, None, None
    #             if parent_constituent_node:
    #                 children = parent_constituent_node.get_children()
    #                 for child_idx, child in enumerate(children):
    #                     if constituent_node == child:
    #                         if child_idx > 0:
    #                             left_constituent_node = children[child_idx-1]
    #                         if child_idx < len(children) - 1:
    #                             right_constituent_node = children[child_idx+1]
    #                         break
                
    #             # nt_ctx
    #             nt_ctx = list() # self, parent, left, right
    #             nt_ctx.append(constituent_node.name)
    #             nt_ctx.append(parent_constituent_node.name if parent_constituent_node else "NULL")
    #             nt_ctx.append(left_constituent_node.name if left_constituent_node else "NULL")
    #             nt_ctx.append(right_constituent_node.name if right_constituent_node else "NULL")
    #             nt_ctx = "-".join(nt_ctx)

    #             # conn_nt_path
    #             conn_nt_path = syntax_tree.get_node_to_node_path(conn_node, constituent_node)
    #             # conn_nt_path_left_number
    #             conn_nt_path_left_number = conn_nt_path + (":>1" if left_number > 1 else ":<=1")

    #             # conn_nt_position
    #             conn_nt_position = syntax_tree.get_relative_position(conn_node, constituent_node)

    #             ss_arg_feats = list()
    #             ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_dict, conn))
    #             ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_lower_dict, conn_lower))
    #             ss_arg_feats.append(Feature.get_feature_by_feat(self.nt_ctx_dict, nt_ctx))
    #             ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_nt_path_dict, conn_nt_path))
    #             ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_nt_path_left_number_dict, conn_nt_path_left_number))
    #             ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_category_dict, conn_category))
    #             ss_arg_feats.append(Feature.get_feature_by_list([left_number]))
    #             ss_arg_feats.append(Feature.get_feature_by_list([right_number]))
    #             ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_nt_position_dict, conn_nt_position))

    #             # merge
    #             ss_arg_feats = Feature.merge_features(ss_arg_feats, 
    #                 "%d|%d|%s" % (sent_idx, conn_idx, ",".join([str(idx) for idx in constituent["indices"]])))

    #             ss_arg_conn = copy(connective)
    #             ss_arg_conn["nt_indices"] = constituent["indices"]

    #             doc_ss_arg_conns.append(ss_arg_conn)
    #             doc_ss_arg_feats.append(ss_arg_feats)
    #     return doc_ss_arg_conns, doc_ss_arg_feats
            
    def extract(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if len(doc_connectives) == 0:
            return doc_connectives
            
        parallel_connectives, non_parallel_connectives = self._divide_connectives_parallel(doc_connectives)

        # parallel_connectives
        if len(parallel_connectives) > 0:
            parallel_connectives = self._extract_parallel_arguments(doc_parsed_result, parallel_connectives)

        # non_parallel_connectives
        if len(non_parallel_connectives) > 0:
            non_parallel_connectives = self._extract_constituent_arguments(doc_parsed_result, non_parallel_connectives, syntax_tree_cache)

        len_parallel = len(parallel_connectives)
        len_non_parallel = len(non_parallel_connectives)
        p_idx, np_idx = 0, 0
        connectives = list()
        while p_idx < len_parallel and np_idx < len_non_parallel:
            p_sent_idx, p_indices = parallel_connectives[p_idx]["sent_idx"], parallel_connectives[p_idx]["indices"]
            np_sent_idx, np_indices = non_parallel_connectives[np_idx]["sent_idx"], non_parallel_connectives[np_idx]["indices"]
            if p_sent_idx < np_sent_idx:
                connectives.append(parallel_connectives[p_idx])
                p_idx += 1
                continue
            elif p_sent_idx > np_sent_idx:
                connectives.append(non_parallel_connectives[np_idx])
                np_idx += 1
                continue
            else:
                if p_indices[0] < np_indices[0]:
                    connectives.append(parallel_connectives[p_idx])
                    p_idx += 1
                    continue
                elif p_indices[0] > np_indices[0]:
                    connectives.append(non_parallel_connectives[np_idx])
                    np_idx += 1
                    continue
                else:
                    if p_indices[-1] <= np_indices[-1]:
                        connectives.append(parallel_connectives[p_idx])
                        p_idx += 1
                        continue
                    else:
                        connectives.append(non_parallel_connectives[np_idx])
                        np_idx += 1
                        continue
        connectives.extend(parallel_connectives[p_idx:])
        connectives.extend(non_parallel_connectives[np_idx:])
        
        return connectives

    def _divide_connectives_parallel(self, SS_connectives):
        parallel_connectives, non_parallel_connectives = list(), list()
        for connective in SS_connectives:
            indices = connective["indices"]
            parallel = False
            for idx in range(len(indices)-1):
                if indices[idx+1] - indices[idx] > 1:
                    parallel = True
                    break
            if parallel:
                parallel_connectives.append(connective)
            else:
                non_parallel_connectives.append(connective)

        return parallel_connectives, non_parallel_connectives

    def _extract_constituent_arguments(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        doc_ss_arg_feats = self._generate_constituent_argument_features(doc_parsed_result, doc_connectives, syntax_tree_cache)
        doc_ss_arg_labels = self._classify_constituent_arguments(doc_parsed_result, doc_ss_arg_feats)
        
        # place indices into two args
        doc_conn_args = [[list(), list()] for _ in range(len(doc_connectives)) ] # [[Arg1, Arg2], ...]
        for feats_name, label in doc_ss_arg_labels:
            if label == 1:
                sent_idx, conn_idx, nt_indices = feats_name.split("|")
                conn_idx = int(conn_idx)
                nt_indices = [int(idx) for idx in nt_indices.split(",")]
                doc_conn_args[conn_idx][0].extend(nt_indices)
            elif label == 2:
                sent_idx, conn_idx, nt_indices = feats_name.split("|")
                conn_idx = int(conn_idx)
                nt_indices = [int(idx) for idx in nt_indices.split(",")]
                doc_conn_args[conn_idx][1].extend(nt_indices)
        
        # merge args
        for connective, conn_args in zip(doc_connectives, doc_conn_args):
            sent_idx = connective["sent_idx"]
            sent_parsed_result = doc_parsed_result[sent_idx]
            conn_args[0].sort()
            conn_args[1].sort()

            if len(conn_args[0]) == 0 or len(conn_args[1]) == 0:
                continue

            # arg1
            if len(conn_args[0]) == 1:
                connective["arg1"] = {
                    "sent_idx": sent_idx,
                    "indices": conn_args[0]
                }
            elif len(conn_args[0]) > 1:
                arg1 = list()
                for arg_idx in range(0, len(conn_args[0])-1):
                    arg1.append(conn_args[0][arg_idx])
                    all_punc = True
                    for t_idx in range(conn_args[0][arg_idx]+1, conn_args[0][arg_idx+1]):
                        if sent_parsed_result["tokens"][t_idx] not in PUNCTUATION_SET:
                            all_punc = False
                            break
                    if all_punc:
                        arg1.extend(range(conn_args[0][arg_idx]+1, conn_args[0][arg_idx+1]))
                arg1.append(conn_args[0][-1])
                connective["arg1"] = {
                    "sent_idx": sent_idx, 
                    "indices": strip_punctuation(sent_parsed_result, arg1)}
                
            # arg2
            conn_args[1].sort()
            if len(conn_args[1]) == 1:
                connective["arg2"] = {
                    "sent_idx": sent_idx,
                    "indices": conn_args[1]
                }
            elif len(conn_args[1]) > 1:
                arg2 = list()
                for arg_idx in range(0, len(conn_args[1])-1):
                    arg2.append(conn_args[1][arg_idx])
                    all_punc = True
                    for t_idx in range(conn_args[1][arg_idx]+1, conn_args[1][arg_idx+1]):
                        if sent_parsed_result["tokens"][t_idx] not in PUNCTUATION_SET:
                            all_punc = False
                            break
                    if all_punc:
                        arg2.extend(range(conn_args[1][arg_idx]+1, conn_args[1][arg_idx+1]))
                arg2.append(conn_args[1][-1])
                connective["arg2"] = {
                    "sent_idx": sent_idx, 
                    "indices": strip_punctuation(sent_parsed_result, arg2)}
        
        return doc_connectives
    
    def _extract_parallel_arguments(self, doc_parsed_result, doc_connectives):
        for connective in doc_connectives:
            sent_idx = connective["sent_idx"]
            sent_parsed_result = doc_parsed_result[sent_idx]

            clauses = self._get_parallel_clauses(sent_parsed_result, connective)
            if len(clauses) == 2:
                connective["arg1"] = {
                    "sent_idx": sent_idx, 
                    "indices": clauses[0]}
                connective["arg2"] = {
                    "sent_idx": sent_idx, 
                    "indices": clauses[1]}
        return doc_connectives

    def _get_parallel_clauses(self, sent_parsed_result, connective):
        indices = connective["indices"]
        sent_len = len(sent_parsed_result["tokens"])

        conn_idx1, conn_idx2 = indices[-1], indices[-1]
        for conn_idx in range(0, len(indices)-1):
            if indices[conn_idx]+1 < indices[conn_idx+1]:
                conn_idx1 = conn_idx
                break
        arg1 = strip_punctuation(sent_parsed_result, list(range(conn_idx1+1, conn_idx2)))
        arg2 = strip_punctuation(sent_parsed_result, list(range(conn_idx2+1, sent_len)))

        clauses = list()
        if len(arg1):
            clauses.append(arg1)
        if len(arg2):
            clauses.append(arg2)
        return clauses
    
    def _get_constituents(self, connective, syntax_tree):
        constituents = list()
        indices = connective["indices"]

        if syntax_tree.tree:
            constituent_nodes = list()
            if len(indices) == 1:
                conn_node = syntax_tree.get_leaf_node_by_token_index(indices[0]).up
            else:
                conn_node = syntax_tree.get_common_ancestor_by_token_indices(indices)

                conn_leaves = set([syntax_tree.get_leaf_node_by_token_index(idx) for idx in indices])
                children = conn_node.get_children()
                for child in children:
                    leaves = set(child.get_leaves())
                    if len(conn_leaves & leaves) == 0:
                        constituent_nodes.append(child)
            curr = conn_node
            while not curr.is_root():
                constituent_nodes.extend(syntax_tree.get_siblings(curr))
                curr = curr.up

            leaves = syntax_tree.tree.get_leaves()
            for node in constituent_nodes:
                constituents.append({
                    "sent_idx": connective["sent_idx"], "connective": connective["connective"],
                    "syntax_tree": syntax_tree, "node": node,
                    "indices": sorted([leaves.index(leaf) for leaf in node.get_leaves()])})
        return constituents

    def _generate_constituent_argument_features(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if syntax_tree_cache is None:
            syntax_tree_cache = dict()

        doc_ss_arg_feats = list()
        for conn_idx, connective in enumerate(doc_connectives):
            sent_idx, conn_indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx]

            if sent_idx in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx]
            else:
                syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])
            
            conn = " ".join([sent_parsed_result["tokens"][idx] for idx in conn_indices])
            conn_lower = conn.lower()
            conn_category = self.conn_category_mapping[conn_lower]
            cpos = "_".join([sent_parsed_result["pos_tags"][idx] for idx in conn_indices])
            try:
                conn_node = syntax_tree.get_self_category_node_by_token_indices(conn_indices)
            except BaseException as e:
                print(sent_parsed_result)
                continue
            
            left_number, right_number = 0, 0
            if conn_node.up:
                children = conn_node.up.get_children()
                for child_idx, child in enumerate(children):
                    if conn_node == child:
                        left_number, right_number = child_idx, len(children)-1-child_idx
                        break
            
            constituents = self._get_constituents(connective, syntax_tree)
            constituents.sort(key=lambda x: x["indices"][0])

            for nt_idx, constituent in enumerate(constituents):
                constituent_node = constituent["node"]
                parent_constituent_node, left_constituent_node, right_constituent_node = constituent_node.up, None, None
                if parent_constituent_node:
                    children = parent_constituent_node.get_children()
                    for child_idx, child in enumerate(children):
                        if constituent_node == child:
                            if child_idx > 0:
                                left_constituent_node = children[child_idx-1]
                            if child_idx < len(children) - 1:
                                right_constituent_node = children[child_idx+1]
                            break
                
                # nt_ctx
                nt_ctx = list() # self, parent, left, right
                nt_ctx.append(constituent_node.name)
                nt_ctx.append(parent_constituent_node.name if parent_constituent_node else "NULL")
                nt_ctx.append(left_constituent_node.name if left_constituent_node else "NULL")
                nt_ctx.append(right_constituent_node.name if right_constituent_node else "NULL")
                nt_ctx = "-".join(nt_ctx)

                # conn_nt_path
                conn_nt_path = syntax_tree.get_node_to_node_path(conn_node, constituent_node)
                # conn_nt_path_left_number
                conn_nt_path_left_number = conn_nt_path + (":>1" if left_number > 1 else ":<=1")

                # conn_nt_position
                conn_nt_position = syntax_tree.get_relative_position(conn_node, constituent_node)

                ss_arg_feats = list()
                ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_dict, conn))
                ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_lower_dict, conn_lower))
                ss_arg_feats.append(Feature.get_feature_by_feat(self.nt_ctx_dict, nt_ctx))
                ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_nt_path_dict, conn_nt_path))
                ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_nt_path_left_number_dict, conn_nt_path_left_number))
                ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_category_dict, conn_category))
                ss_arg_feats.append(Feature.get_feature_by_list([left_number]))
                ss_arg_feats.append(Feature.get_feature_by_list([right_number]))
                ss_arg_feats.append(Feature.get_feature_by_feat(self.conn_nt_position_dict, conn_nt_position))

                # merge
                ss_arg_feats = Feature.merge_features(ss_arg_feats, 
                    "%d|%d|%s" % (sent_idx, conn_idx, ",".join([str(idx) for idx in constituent["indices"]])))

                doc_ss_arg_feats.append(ss_arg_feats)
        return doc_ss_arg_feats

    def _classify_constituent_arguments(self, doc_parsed_result, doc_ss_arg_feats):
        # write features to a file
        if len(doc_ss_arg_feats) == 0:
            return list()
        names = [x.name for x in doc_ss_arg_feats]
        feats = sparse.vstack(list(map(lambda x: x.to_csr(), doc_ss_arg_feats)))
        pred = self.ss_arg_model.predict(feats)
        return list(zip(names, pred))


######################################################
#############    PS Argument Extractor    ############
######################################################
class PSArgumentExtractor:
    def __init__(self, **kw):
        try:
            discourse_path = os.path.join(os.path.dirname(__file__), "discourse")
        except:
            discourse_path = os.path.join("aser", "extract", "discourse")

        conn_category_feat_file = kw.get("conn_category_feat_file", "")
        if conn_category_feat_file:
            if conn_category_feat_file.endswith(".json"):
                with open(conn_category_feat_file, "r") as f:
                    self.conn_category_mapping = json.load(f)
            elif conn_category_feat_file.endswith(".txt"):
                with open(conn_category_feat_file, "r") as f:
                    self.conn_category_mapping = dict()
                    for line in f:
                        line = line.split("#")
                        self.conn_category_mapping[line[0].strip()] = line[1].strip()
            else:
                raise KeyError("Error: %s Not Found." % (conn_category_feat_file))
        else:
            if os.path.exists(os.path.join(discourse_path, "feats", "conn_category.json")):
                with open(os.path.join(discourse_path, "feats", "conn_category.json"), "r") as f:
                    self.conn_category_mapping = json.load(f)
            elif os.path.exists(os.path.join(discourse_path, "feats", "conn_category.txt")):
                with open(os.path.join(discourse_path, "feats", "conn_category.txt"), "r") as f:
                    self.conn_category_mapping = dict()
                    for line in f:
                        line = line.split("#")
                        self.conn_category_mapping[line[0].strip()] = line[1].strip()
                with open(os.path.join(discourse_path, "feats", "conn_category.json"), "w") as f:
                    json.dump(self.conn_category_mapping, f)
            else:
                raise KeyError("Error: %s Not Found." % (conn_category_feat_file))
                
        self.conn_category_dict = {"subordinator": 0, "coordinator": 1, "adverbial": 2}

        for feat in [
           "verb_lemma", "clause_first", "clause_last", 
           "prev_clause_first", "conn_lower"]:
            feat_file = feat + "feat_file"
            with open(kw.get(feat_file, os.path.join(discourse_path, "ps_arg1_feats", feat + ".txt")), "r") as f:
                x_dict = dict()
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    if line:
                        x_dict[line] = idx
                setattr(self, feat + "_dict1", x_dict)
        
        for feat in [
            "clause_production_rule", 
            "clause_first", "clause_first_prev_last_parse_path", 
            "next", "conn_to_root_path", "conn", 
            "prev", "clause_last_next", 
            "conn_lower", "conn_conn_ctx", 
            "compressed_cparent_to_root_path", "cpos", "cparent_to_root_path_node_name"]:
            feat_file = feat + "feat_file"
            with open(kw.get(feat_file, os.path.join(discourse_path, "ps_arg2_feats", feat + ".txt")), "r") as f:
                x_dict = dict()
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    if line:
                        x_dict[line] = idx
                setattr(self, feat + "_dict2", x_dict)
                
        ps_arg1_model_file = kw.get("ps_arg1_model_file", os.path.join(discourse_path, "ps_arg1_classifier.pkl"))
        with open(ps_arg1_model_file, "rb") as f:
            self.ps_arg1_model = pickle.load(f)
        ps_arg2_model_file = kw.get("ps_arg2_model_file", os.path.join(discourse_path, "ps_arg2_classifier.pkl"))
        with open(ps_arg2_model_file, "rb") as f:
            self.ps_arg2_model = pickle.load(f)
            
        self.verb_pos = set(["VB", "VBD", "VBG", "VBN", "VBP", "VBZ"])
            
    def extract(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if syntax_tree_cache is None:
            syntax_tree_cache = dict()

        PS_connectives = [connective for connective in doc_connectives if connective["sent_idx"] > 0]
        self._extract_argument1s(doc_parsed_result, PS_connectives, syntax_tree_cache)
        self._extract_argument2s(doc_parsed_result, PS_connectives, syntax_tree_cache)
        
        return [connective for connective in PS_connectives if "arg1" in connective and "arg2" in connective]
    
    def _extract_argument1s(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        doc_arg1_feats = self._generate_argument1_features(doc_parsed_result, doc_connectives, syntax_tree_cache)
        doc_arg1_labels = self._classify_argument1s(doc_parsed_result, doc_arg1_feats)
        
        # place indices into two args
        doc_conn_arg1s = [list() for _ in range(len(doc_connectives))] # [[(Arg1_1, label_1), (Arg1_2, label_2)], ...]
        for feats_name, label in doc_arg1_labels:
            sent_idx, conn_idx, arg1_indices = feats_name.split("|")
            conn_idx = int(conn_idx)
            arg1_indices = [int(idx) for idx in arg1_indices.split(",")]
            doc_conn_arg1s[conn_idx].append((arg1_indices, label))
        
        # merge arg1s
        for connective, conn_arg1s in zip(doc_connectives, doc_conn_arg1s):
            if len(conn_arg1s) == 0:
                continue
            sent_idx, conn_indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx-1]
            sent_len = len(sent_parsed_result["tokens"])
            
            # arg1
            implicit_arg1 = strip_punctuation(sent_parsed_result, list(range(0, sent_len)))
            for arg1_indices, label in conn_arg1s:
                if label == 0:
                    parts = [list(), list()]
                    p_idx = 0
                    implicit_arg1_len = len(implicit_arg1)
                    arg1_len = len(arg1_indices)
                    for t_idx in implicit_arg1:
                        token = sent_parsed_result["tokens"][t_idx]
                        a_idx = bisect.bisect_left(arg1_indices, t_idx)
                        if a_idx < arg1_len and arg1_indices[a_idx] == t_idx:
                            p_idx = 1
                        else:
                            parts[p_idx].append(t_idx)
                    implicit_arg1 = strip_punctuation(sent_parsed_result, parts[0]) + \
                        strip_punctuation(sent_parsed_result, parts[1])
            if len(implicit_arg1) > 0:
                connective["arg1"] = {
                    "sent_idx": sent_idx-1, 
                    "indices": implicit_arg1}
            else:
                connective["arg1"] = {
                    "sent_idx": sent_idx-1, 
                    "indices": conn_arg1s[-1][0]} # default
        return doc_connectives
    
    def _extract_argument2s(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        doc_arg2_feats = self._generate_argument2_features(doc_parsed_result, doc_connectives, syntax_tree_cache)
        doc_arg2_labels = self._classify_argument2s(doc_parsed_result, doc_arg2_feats)
        
        # place indices into two args
        doc_conn_arg2s = [list() for _ in range(len(doc_connectives))] # [[(Arg2_1, label_1), (Arg2_2, label_2)], ...]
        for feats_name, label in doc_arg2_labels:
            sent_idx, conn_idx, arg2_indices = feats_name.split("|")
            conn_idx = int(conn_idx)
            arg2_indices = [int(idx) for idx in arg2_indices.split(",")]
            doc_conn_arg2s[conn_idx].append((arg2_indices, label))
        
        # merge arg2s
        for connective, conn_arg2s in zip(doc_connectives, doc_conn_arg2s):
            if len(conn_arg2s) == 0:
                continue

            sent_idx, conn_indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx]
            sent_len = len(sent_parsed_result["tokens"])
            
            # arg2
            implicit_arg2 = strip_punctuation(sent_parsed_result, list(range(0, conn_indices[0]))) + \
                strip_punctuation(sent_parsed_result, list(range(conn_indices[-1]+1, sent_len)))
            for arg2_indices, label in conn_arg2s:
                if label == 0:
                    parts = [list(), list()]
                    p_idx = 0
                    implicit_arg2_len = len(implicit_arg2)
                    arg2_len = len(arg2_indices)
                    for t_idx in implicit_arg2:
                        a_idx = bisect.bisect_left(arg2_indices, t_idx)
                        if a_idx < arg2_len and arg2_indices[a_idx] == t_idx:
                            p_idx = 1
                        else:
                            parts[p_idx].append(t_idx)
                    implicit_arg2 = strip_punctuation(sent_parsed_result, parts[0]) + \
                        strip_punctuation(sent_parsed_result, parts[1])
            if len(implicit_arg2) > 0:
                connective["arg2"] = {
                    "sent_idx": sent_idx, 
                    "indices": implicit_arg2}
            else:
                connective["arg2"] = {
                    "sent_idx": sent_idx, 
                    "indices": conn_arg2s[0][0]} # default
        return doc_connectives
    
    def _generate_argument1_features(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if syntax_tree_cache is None:
            syntax_tree_cache = dict()

        doc_arg1_feats = list()
        for conn_idx, connective in enumerate(doc_connectives):
            sent_idx, conn_indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx-1]
            sent_len = len(sent_parsed_result["tokens"])
            
            if sent_idx-1 in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx-1]
            else:
                syntax_tree = syntax_tree_cache[sent_idx-1] = SyntaxTree(sent_parsed_result["parse"])
                
            arg1_clauses = self._get_argument1_clauses(sent_parsed_result, connective, syntax_tree)
            
            if len(arg1_clauses) == 0:
                continue
            
            conn = " ".join([doc_parsed_result[sent_idx]["tokens"][idx] for idx in conn_indices])
            conn_lower = conn.lower()
            conn_category = self.conn_category_mapping[conn_lower]
            
            for clause_idx, clause in enumerate(arg1_clauses):
                # clause
                clause_first = sent_parsed_result["tokens"][clause[0]]
                clause_last = sent_parsed_result["tokens"][clause[-1]]

                # prev
                if clause[0] == 0:
                    prev = "NONE"
                else:
                    prev_sent_idx, prev_idx = get_prev_token_index(doc_parsed_result, sent_idx-1, clause[0], skip_tokens=CLAUSE_SEPARATOR_SET)
                    if prev_sent_idx == sent_idx-1:
                        if prev_idx+1 == clause[0]:
                            prev = sent_parsed_result["tokens"][prev_idx] # previous token not in clause seps
                        else:
                            prev = " ".join([sent_parsed_result["tokens"][idx] for idx in range(prev_idx+1, clause[0])]) # previous tokens in clause seps
                    elif prev_sent_idx+1 == sent_idx-1:
                        prev = " ".join([sent_parsed_result["tokens"][idx] for idx in range(0, clause[0])])
                    else:
                        prev = "NONE"
                
                # verb_lemma
                verb_lemmas = [sent_parsed_result["lemmas"][idx] for idx in clause if sent_parsed_result["pos_tags"][idx] in self.verb_pos]
                
                arg1_feats = list()
                arg1_feats.append(Feature.get_feature_by_feat_list(self.verb_lemma_dict1, verb_lemmas))
                arg1_feats.append(Feature.get_feature_by_feat(self.clause_first_dict1, clause_first))
                arg1_feats.append(Feature.get_feature_by_feat(self.clause_last_dict1, clause_last))
                arg1_feats.append(Feature.get_feature_by_feat(self.prev_clause_first_dict1, prev+"|"+clause_first))
                arg1_feats.append(Feature.get_feature_by_feat(self.conn_lower_dict1, conn_lower))
                arg1_feats.append(Feature.get_feature_by_feat(self.conn_category_dict, conn_category))

                arg1_feats = Feature.merge_features(arg1_feats, 
                    "%d|%d|%s" % (sent_idx, conn_idx, ",".join([str(idx) for idx in clause])))
                doc_arg1_feats.append(arg1_feats)
            
        return doc_arg1_feats

    def _generate_argument2_features(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if syntax_tree_cache is None:
            syntax_tree_cache = dict()

        doc_arg2_feats = list()
        for conn_idx, connective in enumerate(doc_connectives):
            sent_idx, conn_indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx]
            sent_len = len(sent_parsed_result["tokens"])
            
            if sent_idx in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx]
            else:
                syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])
                
            arg2_clauses = self._get_argument2_clauses(sent_parsed_result, connective, syntax_tree)
            
            if len(arg2_clauses) == 0:
                continue
            
            conn = " ".join([sent_parsed_result["tokens"][idx] for idx in conn_indices])
            conn_lower = conn.lower()
            conn_category = self.conn_category_mapping[conn_lower]
            cpos = "_".join([sent_parsed_result["pos_tags"][idx] for idx in conn_indices])
            conn_node = syntax_tree.get_self_category_node_by_token_indices(conn_indices)
            parent_node = conn_node.up
            if parent_node:
                parent_category = parent_node.name
                children = parent_node.get_children()
                left_node, right_node = None, None
                for child_idx, child in enumerate(children):
                    if conn_node == child:
                        if child_idx > 0:
                            left_node = children[child_idx-1]
                        if child_idx < len(children) - 1:
                            right_node = children[child_idx+1]
            else:
                left_node = None
                right_node = None

            # conn_ctx
            conn_ctx = list() # self, parent, left, right
            conn_ctx.append(conn_node.name)
            conn_ctx.append(parent_node.name if parent_node else "NULL")
            conn_ctx.append(left_node.name if left_node else "NULL")
            conn_ctx.append(right_node.name if right_node else "NULL")
            conn_ctx = "-".join(conn_ctx)

            for clause_idx, clause in enumerate(arg2_clauses):
                # clause
                clause_first = sent_parsed_result["tokens"][clause[0]]
                clause_last = sent_parsed_result["tokens"][clause[-1]]

                # prev
                if clause[0] == 0:
                    prev = "NONE"
                else:
                    prev_sent_idx, prev_idx = get_prev_token_index(doc_parsed_result, sent_idx, clause[0], skip_tokens=CLAUSE_SEPARATOR_SET)
                    if prev_sent_idx == sent_idx:
                        if prev_idx+1 == clause[0]:
                            prev = sent_parsed_result["tokens"][prev_idx] # previous token not in clause seps
                        else:
                            prev = " ".join([sent_parsed_result["tokens"][idx] for idx in range(prev_idx+1, clause[0])]) # previous tokens in clause seps
                    elif clause[0]-1 >= 0:
                        prev = " ".join([sent_parsed_result["tokens"][idx] for idx in range(0, clause[0])])
                    else:
                        prev = "NONE"

                # next
                if clause[-1] == len(sent_parsed_result["tokens"])-1:
                    next = "NONE"
                else:
                    next_sent_idx, next_idx = get_next_token_index(doc_parsed_result, sent_idx, clause[-1], skip_tokens=CLAUSE_SEPARATOR_SET)
                    if next_sent_idx == sent_idx:
                        if next_idx-1 == clause[-1]:
                            next = sent_parsed_result["tokens"][next_idx] # next token not in clause seps
                        else:
                            next = " ".join([sent_parsed_result["tokens"][idx] for idx in range(clause[-1]+1, next_idx)]) # next tokens in clause seps
                    elif clause[-1]+1 < len(sent_parsed_result["tokens"]):
                        next = " ".join([sent_parsed_result["tokens"][idx] for idx in range(clause[-1]+1, len(sent_parsed_result["tokens"]))])
                    else:
                        next = "NONE"

                # clause_first_node
                # prev_node
                try:
                    clause_first_node = syntax_tree.get_leaf_node_by_token_index(clause[0]).up
                    if clause_idx > 0:
                        prev_last_node = syntax_tree.get_leaf_node_by_token_index(arg2_clauses[clause_idx-1][-1]).up
                    else:
                        prev_last_node = None
                except:
                    clause_first_node = None
                    prev_last_node = None

                # clause_production_rules
                clause_production_rules = list()
                if syntax_tree.tree:
                    for node in syntax_tree.get_subtree_by_token_indices(clause).tree.traverse():
                        if not node.is_leaf():
                            clause_production_rules.append(
                                node.name + "-->" + " ".join([child.name for child in node.get_children()]))

                # conn_to_root_path
                # cparent_to_root_path
                try:
                    conn_to_root_paths = list()
                    cparent_to_root_paths = list()
                    for idx in conn_indices:
                        node = syntax_tree.get_leaf_node_by_token_index(idx)
                        path = syntax_tree.get_node_path_to_root(node)
                        conn_to_root_paths.append(path)
                        parent_node = node.up
                        path = syntax_tree.get_node_path_to_root(parent_node)
                        cparent_to_root_paths.append(path)
                    cparent_to_root_path_node_names = chain.from_iterable([path.split("-->") for path in cparent_to_root_paths])
                    conn_to_root_path = "&".join(conn_to_root_paths)
                    compressed_cparent_to_root_path = "&".join([get_compressed_path(path) for path in cparent_to_root_paths])
                except:
                    cparent_to_root_path_node_names = ["NONE_TREE"]
                    conn_to_root_path = "NONE_TREE"
                    compressed_cparent_to_root_path = "NONE_TREE"

                # clause_first_prev_last_parse_path
                try:
                    if prev_last_node:
                        clause_first_prev_last_parse_path = syntax_tree.get_node_to_node_path(clause_first_node, prev_last_node)
                    else:
                        clause_first_prev_last_parse_path = "NONE"
                except:
                    clause_first_prev_last_parse_path = "NONE_TREE"

                arg2_feats = list()
                arg2_feats.append(Feature.get_feature_by_feat_list(self.clause_production_rule_dict2, clause_production_rules))
                arg2_feats.append(Feature.get_feature_by_feat(self.clause_first_dict2, clause_first))
                arg2_feats.append(Feature.get_feature_by_feat(self.clause_first_prev_last_parse_path_dict2, clause_first_prev_last_parse_path))
                arg2_feats.append(Feature.get_feature_by_feat(self.next_dict2, next))
                arg2_feats.append(Feature.get_feature_by_feat(self.conn_to_root_path_dict2, conn_to_root_path))
                arg2_feats.append(Feature.get_feature_by_feat(self.conn_dict2, conn))
                arg2_feats.append(Feature.get_feature_by_feat(self.prev_dict2, prev))
                arg2_feats.append(Feature.get_feature_by_feat(self.clause_last_next_dict2, clause_last+"|"+next))
                arg2_feats.append(Feature.get_feature_by_feat(self.conn_lower_dict2, conn_lower))
                arg2_feats.append(Feature.get_feature_by_feat(self.conn_conn_ctx_dict2, conn+"|"+conn_ctx))
                arg2_feats.append(Feature.get_feature_by_feat(self.compressed_cparent_to_root_path_dict2, compressed_cparent_to_root_path))
                arg2_feats.append(Feature.get_feature_by_feat(self.cpos_dict2, cpos))
                arg2_feats.append(Feature.get_feature_by_feat_list(self.cparent_to_root_path_node_name_dict2, cparent_to_root_path_node_names))
                arg2_feats.append(Feature.get_feature_by_feat(self.conn_category_dict, conn_category))

                arg2_feats = Feature.merge_features(arg2_feats, 
                    "%d|%d|%s" % (sent_idx, conn_idx, ",".join([str(idx) for idx in clause])))
                doc_arg2_feats.append(arg2_feats)
        return doc_arg2_feats
    
    def _classify_argument1s(self, doc_parsed_result, doc_arg1_feats):
        # write features to a file
        if len(doc_arg1_feats) == 0:
            return list()
        names = [x.name for x in doc_arg1_feats]
        feats = sparse.vstack(list(map(lambda x: x.to_csr(), doc_arg1_feats)))
        pred = self.ps_arg1_model.predict(feats)
        return list(zip(names, pred))
    
    def _classify_argument2s(self, doc_parsed_result, doc_arg2_feats):
        # write features to a file
        if len(doc_arg2_feats) == 0:
            return list()
        names = [x.name for x in doc_arg2_feats]
        feats = sparse.vstack(list(map(lambda x: x.to_csr(), doc_arg2_feats)))
        pred = self.ps_arg2_model.predict(feats)
        return list(zip(names, pred))
        
    def _get_argument1_clauses(self, sent_parsed_result, connective, syntax_tree):
        return get_clauses(sent_parsed_result, syntax_tree)
        
    def _get_argument2_clauses(self, sent_parsed_result, connective, syntax_tree):
        return get_clauses(sent_parsed_result, syntax_tree, index_seps=set(connective["indices"]))

######################################################
###########    Explicit Sense Classifier    ##########
######################################################
class ExplicitSenseClassifier:
    def __init__(self, **kw):
        try:
            discourse_path = os.path.join(os.path.dirname(__file__), "discourse")
        except:
            discourse_path = os.path.join("aser", "extract", "discourse")

        conn_category_feat_file = kw.get("conn_category_feat_file", "")
        if conn_category_feat_file:
            if conn_category_feat_file.endswith(".json"):
                with open(conn_category_feat_file, "r") as f:
                    self.conn_category_mapping = json.load(f)
            elif conn_category_feat_file.endswith(".txt"):
                with open(conn_category_feat_file, "r") as f:
                    self.conn_category_mapping = dict()
                    for line in f:
                        line = line.split("#")
                        self.conn_category_mapping[line[0].strip()] = line[1].strip()
            else:
                raise KeyError("Error: %s Not Found." % (conn_category_feat_file))
        else:
            if os.path.exists(os.path.join(discourse_path, "feats", "conn_category.json")):
                with open(os.path.join(discourse_path, "feats", "conn_category.json"), "r") as f:
                    self.conn_category_mapping = json.load(f)
            elif os.path.exists(os.path.join(discourse_path, "feats", "conn_category.txt")):
                with open(os.path.join(discourse_path, "feats", "conn_category.txt"), "r") as f:
                    self.conn_category_mapping = dict()
                    for line in f:
                        line = line.split("#")
                        self.conn_category_mapping[line[0].strip()] = line[1].strip()
                with open(os.path.join(discourse_path, "feats", "conn_category.json"), "w") as f:
                    json.dump(self.conn_category_mapping, f)
            else:
                raise KeyError("Error: %s Not Found." % (conn_category_feat_file))

        with open(kw.get("exp_conn_file", os.path.join(discourse_path, "explicit_feats", "exp_conn.txt")), "r") as f:
            self.sorted_conn = list()
            for idx, line in enumerate(f):
                line = line.rstrip()
                if line:
                    self.sorted_conn.append(line)
            self.sorted_conn.sort()
                
        for feat in [
            "conn", "cpos", "prev_conn", "conn_lower", 
            "self_category", "parent_category", "left_category", "right_category", 
            "conn_lower_self_category", "conn_lower_parent_category", "conn_lower_left_category", "conn_lower_right_category", 
            "self_category_parent_category", "self_category_right_category", "self_category_left_category", 
            "parent_category_left_category", "parent_category_right_category", "left_category_right_category", 
            "conn_parent_ctx", "as_prev_conn", "as_prev_cpos", "when_prev_conn", "when_prev_cpos"]:
            feat_file = feat + "feat_file"
            with open(kw.get(feat_file, os.path.join(discourse_path, "explicit_feats", feat + ".txt")), "r") as f:
                x_dict = dict()
                for idx, line in enumerate(f):
                    line = line.rstrip()
                    if line:
                        x_dict[line] = idx
                setattr(self, feat + "_dict", x_dict)

        explicit_model_file = kw.get("explicit_model_file", os.path.join(discourse_path, "explicit_classifier.pkl"))
        with open(explicit_model_file, "rb") as f:
            self.explicit_model = pickle.load(f)
        self.predict_label_dict = {
            0: "None",
            1: "Precedence",         # "Temporal.Asynchronous.Precedence"
            2: "Succession",         # "Temporal.Asynchronous.Succession"
            3: "Synchrony",          # "Temporal.Synchrony"
            4: "Reason",             # "Contingency.Cause.Reason"
            5: "Result",             # "Contingency.Cause.Result"
            6: "Condition",          # "Contingency.Condition"
            7: "Contrast",           # "Comparison.Contrast"
            8: "Concession",         # "Comparison.Concession"
            9: "Conjunction",        # "Expansion.Conjunction"
            10: "Instantiation",     # "Expansion.Instantiation"
            11: "Restatement",       # "Expansion.Restatement"
            12: "Alternative",       # "Expansion.Alternative"
            13: "ChosenAlternative", # "Expansion.Alternative.Chosen alternative"
            14: "Exception"          # "Expansion.Exception"
        }

    # def help_func(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
    #     doc_explicit_feats = self._generate_explicit_features(doc_parsed_result, doc_connectives, syntax_tree_cache)

    #     return doc_connectives, doc_explicit_feats
        
    def classify(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if len(doc_connectives) == 0:
            return doc_connectives
            
        doc_explicit_feats = self._generate_explicit_features(doc_parsed_result, doc_connectives, syntax_tree_cache)
        doc_explicit_labels = self._classify_explicit(doc_parsed_result, doc_explicit_feats)
        for connective, label in zip(doc_connectives, doc_explicit_labels):
            connective["sense"] = label[1]
        return doc_connectives
    
    def _generate_explicit_features(self, doc_parsed_result, doc_connectives, syntax_tree_cache=None):
        if syntax_tree_cache is None:
            syntax_tree_cache = dict()

        doc_explicit_feats = list()
        for conn_idx, connective in enumerate(doc_connectives):
            sent_idx, indices = connective["sent_idx"], connective["indices"]
            sent_parsed_result = doc_parsed_result[sent_idx]
            sent_len = len(sent_parsed_result["tokens"])

            if sent_idx in syntax_tree_cache:
                syntax_tree = syntax_tree_cache[sent_idx]
            else:
                syntax_tree = syntax_tree_cache[sent_idx] = SyntaxTree(sent_parsed_result["parse"])
            
            # conn
            conn = " ".join([sent_parsed_result["tokens"][idx] for idx in indices])
            conn_lower = conn.lower()
            cpos = "_".join([sent_parsed_result["pos_tags"][idx] for idx in indices])

            # prev
            prev_sent_idx, prev_idx = get_prev_token_index(doc_parsed_result, sent_idx, indices[0])
            if prev_sent_idx != -1:
                prev = doc_parsed_result[prev_sent_idx]["tokens"][prev_idx]
            else:
                prev = "NONE"

            # Pitler
            try:
                category_node = syntax_tree.get_self_category_node_by_token_indices(indices)
                self_category = category_node.name
                parent_category_node = category_node.up
                if parent_category_node:
                    parent_category = parent_category_node.name
                    children = parent_category_node.get_children()
                    category_node_id = id(category_node)
                    left_category_node, right_category_node = None, None
                    for child_idx, child in enumerate(children):
                        if category_node_id == id(child):
                            if child_idx > 0:
                                left_category_node = children[child_idx-1]
                            if child_idx < len(children) - 1:
                                right_category_node = children[child_idx+1]
                    left_category = left_category_node.name if left_category_node else "NONE"
                    right_category = right_category_node.name if right_category_node else "NONE"
                else:
                    parent_category = "ROOT"
                    left_category = "NONE"
                    right_category = "NONE"

                # parent_ctx
                if parent_category_node:
                    parent_ctx = list() # self, parent, children
                    parent_ctx.append(parent_category_node.name)
                    parent_ctx.append(parent_category_node.up.name if parent_category_node.up else "NULL")
                    parent_ctx.extend([child.name for child in parent_category_node.get_children()])
                    parent_ctx = "-".join(parent_ctx)
                else:
                    parent_ctx = "None"
            except:
                self_category = "NONE_TREE"
                parent_category = "NONE_TREE"
                left_category = "NONE_TREE"
                right_category = "NONE_TREE"
                parent_ctx = "NONE_TREE"
                
            # as_prev_conn
            if conn == "as":
                prev_tokens = [sent_parsed_result["tokens"][idx] for idx in range(0, indices[0])]
                prev_connectives = self._extract_connectives_by_tokens(prev_tokens)
                prev_connectives.sort(key=lambda x: x["indices"][-1])
                if len(prev_connectives) > 0:
                    as_prev_conn = prev_connectives[-1]["connective"]
                    as_prev_cpos = " ".join([sent_parsed_result["pos_tags"][idx] for idx in prev_connectives[0]["indices"]])
                else:
                    as_prev_conn = "NULL"
                    as_prev_cpos = "NULL"
            else:
                as_prev_conn = "NOT_as"
                as_prev_cpos = "NOT_as"
                
            # when_prev_conn
            if conn == "when":
                prev_tokens = [sent_parsed_result["tokens"][idx] for idx in range(0, indices[0])]
                prev_connectives = self._extract_connectives_by_tokens(prev_tokens)
                prev_connectives.sort(key=lambda x: x["indices"][-1])
                if len(prev_connectives) > 0:
                    when_prev_conn = prev_connectives[-1]["connective"]
                    when_prev_cpos = " ".join([sent_parsed_result["pos_tags"][idx] for idx in prev_connectives[0]["indices"]])
                else:
                    when_prev_conn = "NULL"
                    when_prev_cpos = "NULL"
            else:
                when_prev_conn = "NOT_when"
                when_prev_cpos = "NOT_when"

            explicit_feats = list()
            # Z. Lin
            explicit_feats.append(Feature.get_feature_by_feat(self.conn_dict, conn))
            explicit_feats.append(Feature.get_feature_by_feat(self.cpos_dict, cpos))
            explicit_feats.append(Feature.get_feature_by_feat(self.prev_conn_dict, prev+"|"+conn))
            explicit_feats.append(Feature.get_feature_by_feat(self.conn_lower_dict, conn_lower))
            
            # pitler
            explicit_feats.append(Feature.get_feature_by_feat(self.self_category_dict, self_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.parent_category_dict, parent_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.left_category_dict, left_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.right_category_dict, right_category))
            
            # conn-syn
            explicit_feats.append(Feature.get_feature_by_feat(self.conn_lower_self_category_dict, conn_lower+"|"+self_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.conn_lower_parent_category_dict, conn_lower+"|"+parent_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.conn_lower_left_category_dict, conn_lower+"|"+left_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.conn_lower_right_category_dict, conn_lower+"|"+right_category))
            
            # sync-syn
            explicit_feats.append(Feature.get_feature_by_feat(self.self_category_parent_category_dict, self_category+"|"+parent_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.self_category_right_category_dict, self_category+"|"+right_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.self_category_left_category_dict, self_category+"|"+left_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.parent_category_left_category_dict, parent_category+"|"+left_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.parent_category_right_category_dict, parent_category+"|"+right_category))
            explicit_feats.append(Feature.get_feature_by_feat(self.left_category_right_category_dict, left_category+"|"+right_category))
            
            # mine
            explicit_feats.append(Feature.get_feature_by_feat(self.conn_parent_ctx_dict, conn+"|"+parent_ctx))
            explicit_feats.append(Feature.get_feature_by_feat(self.as_prev_conn_dict, as_prev_conn))
            explicit_feats.append(Feature.get_feature_by_feat(self.as_prev_cpos_dict, as_prev_cpos))
            explicit_feats.append(Feature.get_feature_by_feat(self.when_prev_conn_dict, when_prev_conn))
            explicit_feats.append(Feature.get_feature_by_feat(self.when_prev_cpos_dict, when_prev_cpos))
            
            # merge
            explicit_feats = Feature.merge_features(explicit_feats, 
                "%d|%d|%s" % (sent_idx, conn_idx, ",".join([str(idx) for idx in indices])))

            doc_explicit_feats.append(explicit_feats)
        return doc_explicit_feats
    
    def _extract_connectives_by_tokens(self, tokens):
        all_connectives = list()
        tokens = [t.lower() for t in tokens]
        for t_idx, token in enumerate(tokens):
            c_idx = bisect.bisect_left(self.sorted_conn, token)
            while c_idx < len(self.sorted_conn):
                conn = self.sorted_conn[c_idx]
                c_idx += 1
                if len(conn) < len(token):
                    break
                elif not conn.startswith(token):
                    break

                if ".." in conn:
                    conn_lists = [c.split() for c in conn.split("..")] # c1..c2..
                    if conn_lists[0][0] != token:
                        break
                    if len(conn_lists[0]) + t_idx <= len(tokens):
                        # check conn_lists[0]
                        match = True
                        for w_idx, c in enumerate(conn_lists[0]):
                            if tokens[w_idx + t_idx] != c:
                                match = False
                                break
                        if not match:
                            continue
                        indices = list(range(t_idx, t_idx+len(conn_lists[0])))

                        # check conn_lists[1]
                        for t_idx in index_from(tokens, conn_lists[1][0], start_from=t_idx):
                            match = False
                            if len(conn_lists[1]) + t_idx <= len(tokens):
                                match = True
                                for w_idx, c in enumerate(conn_lists[1]):
                                    if tokens[w_idx + t_idx] != c:
                                        match = False
                                        break
                                if match:
                                    all_connectives.append(
                                        {"connective": conn, "indices": indices + list(range(t_idx, t_idx+len(conn_lists[1])))})
                else:
                    conn_list = conn.split()
                    if conn_list[0] != token:
                        break
                    if len(conn_list) + t_idx <= len(tokens):
                        match = True
                        for w_idx, c in enumerate(conn_list):
                            if tokens[w_idx + t_idx] != c:
                                match = False
                                break
                        if match:
                            all_connectives.append(
                                {"connective": conn, "indices": list(range(t_idx, t_idx+len(conn_list)))})
        # filter shorter and duplicative conn
        all_connectives.sort(key=lambda x: (-len(x["indices"]), -x["indices"][0]))
        filtered_connectives = list()
        used_indices = set()
        for conn_indices in all_connectives:
            indices = conn_indices["indices"]
            # duplicated = len(set(indices) & used_indices) > 0
            duplicated = False
            for idx in indices:
                if idx in used_indices:
                    duplicated = True
                    break
            if not duplicated:
                used_indices.update(indices)
                filtered_connectives.append(conn_indices)
        return filtered_connectives
        
    def _classify_explicit(self, doc_parsed_result, doc_explicit_feats):
        # write features to a file
        if len(doc_explicit_feats) == 0:
            return list()
        names = [x.name for x in doc_explicit_feats]
        feats = sparse.vstack(list(map(lambda x: x.to_csr(), doc_explicit_feats)))
        pred = np.argmax(self.explicit_model.predict_proba(feats), axis=1)
        pred_label = [self.predict_label_dict[self.explicit_model.classes_[x]] for x in pred]
        return list(zip(names, pred_label))