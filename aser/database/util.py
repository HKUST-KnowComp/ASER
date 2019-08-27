import hashlib

stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
             'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those'}


def get_length_ratio(example):
    e1_length = len(example['activity1']['tokens'])
    e2_length = len(example['activity2']['tokens'])
    r = (e1_length+e2_length) / len(example['sentence_tokens'])
    return r


def compute_overlap(w1, w2):
    w1_words = set(w1.split()) - stopwords
    w2_words = set(w2.split()) - stopwords
    Jaccard = len(w1_words & w2_words) / len(w1_words | w2_words)
    return Jaccard


def get_event_verbs(event):
    verbs = ' '.join([x[1].lower() for x in event['skeleton_words']
                      if x[2].startswith('VB')])
    return verbs


def get_event_skeleton_words_clean(event):
    skeleton_words_clean = ' '.join(
        [x[1].lower() for x in event['skeleton_words'] if x[1].lower() not in stopwords])
    return skeleton_words_clean


def get_event_skeleton_words(event):
    skeleton_words = ' '.join([x[1].lower() for x in event['skeleton_words']])
    return skeleton_words


def get_event_words(event):
    words = ' '.join([x[1].lower() for x in event['words']])
    return words


def generate_id(key):
    return hashlib.sha1(key.encode('utf-8')).hexdigest()


def merge_relations(relations1, relations2):
    if len(relations1) > len(relations2):
        relations1, relations2 = relations2, relations1
    new_relations = relations2.copy()
    for i in range(len(relations1)):
        flag = True
        insert_index = 0
        remove_index = -1
        for j in range(0, len(relations2)):
            if relations1[i][0] <= relations2[j][0]:
                insert_index += 1
            if relations1[i][1] == relations2[j][1]:
                if relations1[i][2] < relations2[j][2]:
                    remove_index = i
                else:
                    flag = False
                    break
        if flag:
            if remove_index == -1:
                new_relations.insert(insert_index, relations1[i])
            else:
                if remove_index < insert_index:
                    new_relations.insert(insert_index, relations1[i])
                    new_relations.pop(remove_index)
                else:
                    new_relations.pop(remove_index)
                    new_relations.insert(insert_index, relations1[i])
    return new_relations

def print_function(x, start_str='', end_str=''):
    if isinstance(x, list) and len(x) > 0:
        print(start_str, '[')
        for y in x:
            print_function(y, start_str=start_str+'\t')
        print(start_str, ']')
    elif isinstance(x, tuple) and len(x) > 0 and isinstance(x[0], float):
        print(start_str, '(')
        for y in x:
            print_function(y, start_str=start_str+'\t')
        print(start_str, ')')
    elif isinstance(x, dict):
        print(start_str, '{')
        for k, v in x.items():
            print(start_str+'\tkey: ', k)
            print(start_str+'\tvalue: ', v)
        print(start_str, '}')
    else:
        print(start_str, x)
