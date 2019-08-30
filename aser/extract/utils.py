import os
import numpy as np
import collections
import random
from itertools import islice
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from aser.models.example import Training_Example, Two_Sentence_Training_Example
# from aser.models.activity import Activity

# class SA_Classifier:
#     def __init__(self, training_data=None, embedding_path='glove.txt', model_link=None):

#         self.embedding_dict = load_embedding(embedding_path)
#         if model_link is None:
#             self.model = Model()
#         else:
#             self.model = Model()
#         self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.2)
#         self.loss_func = torch.nn.CrossEntropyLoss()
#         if training_data is not None:
#             self.training_data = training_data
#             self.X_train, self.Y_train = self.tensorize_examples()
#         self.softmax = torch.nn.Softmax()

#     def load_data(self, current_data):
#         random.shuffle(current_data)
#         self.training_data = current_data
#         self.X_train, self.Y_train = self.tensorize_examples()

#     def tensorize_examples(self):
#         tensorized_example_list = list()
#         labels = list()
#         for t in self.training_data:
#             tensorized_example_list.append(t['example'].tensorize(self.embedding_dict))
#             if t['label'] == 1:
#                 labels.append(torch.LongTensor([1]))
#             else:
#                 labels.append(torch.LongTensor([0]))
#         # print(np.asarray(tensorized_example_list).shape)
#         return tensorized_example_list, labels

#     def train(self):
#         # print(self.X_train)
#         # print(self.X_train.shape)
#         # print(self.Y_train)
#         # print(self.Y_train.shape)
#         for tmp_iter in range(10):
#             print('Training iteration:', tmp_iter)
#             for i in range(len(self.X_train)):
#                 predict = self.model(self.X_train[i])
#                 # print(predict, predict.size())
#                 # print(self.Y_train[i].unsqueeze(0), self.Y_train[i].unsqueeze(0).size())
#                 tmp_loss = self.loss_func(predict, self.Y_train[i])
#                 self.optimizer.zero_grad()
#                 tmp_loss.backward()
#                 self.optimizer.step()

#     def predict(self, test_example):
#         # print(test_example.tensorize(self.embedding_dict))
#         tmp_predict = self.model(test_example.tensorize(self.embedding_dict))
#         predict_after_softmax = self.softmax(tmp_predict)
#         # print(predict_after_softmax)
#         # print(torch.max(tmp_predict, 1)[1].data.numpy().squeeze())
#         return predict_after_softmax.squeeze().data.numpy()[1]


# def check_activity_contain_token(edge_list, token):
#     for relation in edge_list:
#         if relation[0] == token or relation[1] == token:
#             return True
#     return False


def shrink_example(tmp_example):
    new_edges = list()
    for edge in tmp_example.sentence_parsed_relations:
        new_edge = list()
        if tmp_example.activity1.contain_tuple(edge[0]):
            new_edge.append('A')
        elif tmp_example.activity2.contain_tuple(edge[0]):
            new_edge.append('B')
        else:
            new_edge.append(edge[0][1])
        new_edge.append(edge[1])
        if tmp_example.activity1.contain_tuple(edge[2]):
            new_edge.append('A')
        elif tmp_example.activity2.contain_tuple(edge[2]):
            new_edge.append('B')
        else:
            new_edge.append(edge[2][1])
        if new_edge[0] != new_edge[2]:
            new_edges.append(tuple(new_edge))
    return new_edges


def find_edge(tmp_example, tmp_edge):
    shrinked_example = shrink_example(tmp_example)
    if tmp_edge in shrinked_example:
        return True
    else:
        return False


def verify_example_with_rule(tmp_example, tmp_rule):
    for tmp_edge in tmp_rule.positive_rules:
        if not find_edge(tmp_example, tmp_edge):
            return False
    for tmp_edge in tmp_rule.negative_rules:
        if find_edge(tmp_example, tmp_edge):
            return False
    return True


def select_positive_examples_with_rule(tmp_examples, rules):
    selected_positive_examples = list()
    # print('We are extracting positive examples from:', len(tmp_examples), 'examples')
    for tmp_example in tmp_examples:
        for rule in rules:
            if verify_example_with_rule(tmp_example, rule):
                selected_positive_examples.append({'example': tmp_example, 'label': 1})
                break
    # print('Number of extracted examples:', len(selected_positive_examples))
    return selected_positive_examples


def select_positive_examples_with_connectives(tmp_examples, connectives):
    selected_positive_examples = list()
    # print('We are extracting positive examples from:', len(tmp_examples), 'examples')
    for tmp_example in tmp_examples:
        for connective in connectives:
            if tmp_example.verify_connective(connective):
                selected_positive_examples.append({'example': tmp_example, 'label': 1})
                break
    # print('Number of extracted examples:', len(selected_positive_examples))
    return selected_positive_examples


def load_embedding(embedding_path):
    print("Loading word embeddings from {}...".format(embedding_path))
    default_embedding = np.zeros(300)
    embedding_dict = collections.defaultdict(lambda: default_embedding)
    with open(embedding_path) as tmp_f:
        for tmp_i, line in enumerate(tmp_f.readlines()):
            if tmp_i == 0:
                continue
            splits = line.split()
            try:
                assert len(splits) == 300 + 1
                word = splits[0]
                embedding = np.array([float(s) for s in splits[1:]])
                embedding_dict[word] = embedding
            except AssertionError:
                continue
    print("Done loading word embeddings.")
    return embedding_dict


def generate_sentence_from_parsed_result(parsed_result):
    all_tokens = list()
    for tmp_relation in parsed_result:
        if tmp_relation[0] not in all_tokens:
            all_tokens.append(tmp_relation[0])
        if tmp_relation[2] not in all_tokens:
            all_tokens.append(tmp_relation[2])
            # all_tokens[tmp_relation[2][1]] = tmp_relation[2][0]
    sorted_tokens = sorted(all_tokens, key=lambda tup: tup[0])
    tmp_sentence = list()
    for word in sorted_tokens:
        tmp_sentence.append(word[1])
    return tmp_sentence


def select_verbs_from_parsed_result(tmp_parsed_result):
    selected_verb_positions = list()
    for relation in tmp_parsed_result:
        if 'VB' in relation[0][2] and relation[0][0] not in selected_verb_positions:
            selected_verb_positions.append(relation[0][0])
        if 'VB' in relation[2][2] and relation[2][0] not in selected_verb_positions:
            selected_verb_positions.append(relation[2][0])
    return selected_verb_positions


def select_verbs_from_extracted_activities(tmp_extracted_activities):
    verb_counter = dict()
    for activity_type in tmp_extracted_activities:
        verb_counter[activity_type] = list()
        for activity in tmp_extracted_activities[activity_type]:
            for relation in activity.relations:
                if 'VB' in relation[0][2] and relation[0][0] not in verb_counter[activity_type]:
                    verb_counter[activity_type].append(relation[0][0])
                if 'VB' in relation[2][2] and relation[2][0] not in verb_counter[activity_type]:
                    verb_counter[activity_type].append(relation[2][0])
    return verb_counter


def get_selected_sizes_from_dist(num_dist, selected_num, min_num):
    selected_nums = [int(n / sum(num_dist) * selected_num) for n in num_dist]
    smoothed_nums = [max(min_num, n) if n else n for n in selected_nums]
    final_nums = [int(n / sum(smoothed_nums) * selected_num) for n in smoothed_nums]
    final_nums[0] = selected_num - sum(final_nums[1:])
    return final_nums


def get_time_str(t):
    import time
    ISFORMAT = "%Y-%m-%d %H:%M:%S"
    return time.strftime(ISFORMAT, time.localtime(t))


def fetch_lines(file_path, st, block_size):
    with open(file_path) as f:
        lines = list(islice(f, st, st + block_size))
    return lines


def fetch_lines_list(file_path, st_list, block_size):
    lines_list = []
    old_st = -1
    old_val = ""
    with open(file_path) as f:
        # print(st_list)
        for st in st_list:
            if st == old_st:
                lines_list.append(old_val)
                continue
            idx = st if old_st == -1 else st - old_st - 1
            # print(st, idx)
            lines = list(islice(f, idx, idx + block_size))
            lines_list.append(lines)
            old_st = st
            old_val = lines
    return lines_list


def over_sample(input_list, sample_num):
    sampled_list = []
    while len(sampled_list) < sample_num:
        sampled_list.extend(input_list)
    random.shuffle(sampled_list)
    if len(sampled_list) > sample_num:
        return sampled_list[:sample_num]
    else:
        return sampled_list

def check_and_mkdir(path, verbose=True):
    if not os.path.exists(path):
        os.mkdir(path)
        if verbose:
            print("mkdir ", path)