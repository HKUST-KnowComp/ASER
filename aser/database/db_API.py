from aser.database.util import *
from aser.database._kg_connection import *
from aser.database._kg_connection import _KG_Connection


class KG_Connection(_KG_Connection):
    def __init__(self, db_path, mode='cache'):
        super(KG_Connection, self).__init__(db_path, mode=mode)
        self.type = 'words'
        self.event_columns = ['_id', 'verbs', 'skeleton_words_clean',
                              'skeleton_words', 'words', 'pattern', 'frequency']
        self.event_column_types = ['PRIMARY KEY'] + ['TEXT'] * 5 + ['REAL']
        self.event_partial_cache['verbs'] = dict()
        self.event_partial_cache['skeleton_words_clean'] = dict()
        self.event_partial_cache['skeleton_words'] = dict()
        self.init()

    def get_partial_match_events(self, event, bys=['skeleton_words', 'skeleton_words_clean', 'verbs'],
                                 top_n=None, threshold=0.1, sort=True):
        return super(KG_Connection, self).get_partial_match_events(event, bys=bys, top_n=top_n, threshold=threshold, sort=sort)


def generate_event_id(event):
    words = get_event_words(event)
    _id = generate_id(words)
    return _id


def preprocess_event(event, pattern=None):
    verbs = get_event_verbs(event)
    skeleton_words_clean = get_event_skeleton_words_clean(event)
    skeleton_words = get_event_skeleton_words(event)
    words = get_event_words(event)
    _id = generate_id(words)
    if not pattern:
        pattern = 'unknown'
    return {'_id': _id, "verbs": verbs, "skeleton_words_clean": skeleton_words_clean,
            "skeleton_words": skeleton_words, 'words': words, 'pattern': pattern, 'frequency': 1.0}


def generate_relation_id(example):
    if isinstance(example, list) and len(example) == 2:
        try:
            event1_id = example[0]['event_id']
            event2_id = example[1]['event_id']
        except:
            event1_id = example[0]
            event2_id = example[1]
        _id = generate_id(event1_id + '$' + event2_id)
    else:
        event1_id = generate_event_id(example['activity1'])
        event2_id = generate_event_id(example['activity2'])
        _id = generate_id(event1_id + '$' + event2_id)
    return _id


def preprocess_relation(example):
    if 'event_pair_id' in example:
        result = {'_id': example['event_pair_id'],
                  "event1_id": example['event1_id'], "event2_id": example['event2_id']}
    else:
        event1_id = generate_event_id(example['activity1'])
        event2_id = generate_event_id(example['activity2'])
        _id = generate_id(event1_id + '$' + event2_id)
        result = {'_id': _id, "event1_id": event1_id, "event2_id": event2_id}
    for k in range(len(relation_senses)):
        result[relation_senses[k]] = 0.0
    result['Co_Occurrence'] = 1.0
    for r in example['relations']:
        result[r] = 1.0
    return result


def preprocess_example(example, corpus='', location=''):
    event1_id = generate_event_id(example['activity1'])
    event2_id = generate_event_id(example['activity2'])
    event_pair_id = generate_id(event1_id + '$' + event2_id)
    is_double = int('sentence1_tokens' in example)
    flag = 0 if len(example['relations']) > 0 else -1
    result = {'location': location, 'corpus': corpus, 'event_pair_id': event_pair_id,
              'event1_id': event1_id, 'event2_id': event2_id, 'relations': example['relations'], 'is_double': is_double, 'flag': flag}
    return result
