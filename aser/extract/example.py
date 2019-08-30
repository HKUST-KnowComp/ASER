from aser.extract.activity import Activity

Seed_Connectives = {
    'Precedence': [['before']],
    'Succession': [['after']],
    'Synchronous': [['meanwhile'], ['at', 'the', 'same', 'time']],
    'Reason': [['because']],
    'Result': [['so'], ['thus'], ['therefore']],
    'Condition': [['if']],
    'Contrast': [['but'], ['however']],
    'Concession': [['although']],
    'Conjunction': [['and'], ['also']],
    'Instantiation': [['for', 'example'], ['for', 'instance']],
    'Restatement': [['in', 'other', 'words']],
    'Alternative': [['or'], ['unless']],
    'ChosenAlternative': [['instead']],
    'Exception': [['except']],
}


class Training_Example:
    def __init__(self, initial_data):
        if initial_data is None:
            self.activity1 = Activity(None)
            self.activity2 = Activity(None)
            # self.sentence = ''
            self.sentence_parsed_relations = list()
            self.sentence_tokens = list()
        else:
            self.activity1 = Activity(initial_data['activity1'])
            self.activity2 = Activity(initial_data['activity2'])
            # self.sentence = initial_data['sentence']
            self.sentence_parsed_relations = initial_data['sentence_parsed_relations']
            self.sentence_tokens = initial_data['sentence_tokens']

    # def print(self):
    # print('sentence:', self.sentence)

    def to_dict(self):
        tmp_relations = self.predict_relations()
        activity1_dict = self.activity1.to_dict()
        activity1_dict['skeleton_tokens'] = [self.sentence_tokens[x[0] - 1] for x in activity1_dict['skeleton_words']]
        activity1_dict['tokens'] = [self.sentence_tokens[x[0] - 1] for x in activity1_dict['words']]
        activity2_dict = self.activity2.to_dict()
        activity2_dict['skeleton_tokens'] = [self.sentence_tokens[x[0] - 1] for x in activity2_dict['skeleton_words']]
        activity2_dict['tokens'] = [self.sentence_tokens[x[0] - 1] for x in activity2_dict['words']]
        return {'activity1': activity1_dict, 'activity2': activity2_dict, 'relations': tmp_relations,
                'sentence_parsed_relations': self.sentence_parsed_relations, 'sentence_tokens': self.sentence_tokens}

    def to_dict_with_temporal(self):
        tmp_relations = self.predict_relations()
        activity1_dict = self.activity1.to_dict()
        activity1_dict['skeleton_tokens'] = list()
        activity1_dict['tokens'] = list()
        for w_tuple in activity1_dict['skeleton_words']:
            if 'VB' in w_tuple[2]:
                activity1_dict['skeleton_tokens'].append(self.sentence_tokens[w_tuple[0] - 1])
            else:
                activity1_dict['skeleton_tokens'].append(w_tuple[1])
        for w_tuple in activity1_dict['words']:
            if 'VB' in w_tuple[2]:
                activity1_dict['tokens'].append(self.sentence_tokens[w_tuple[0] - 1])
            else:
                activity1_dict['tokens'].append(w_tuple[1])
        activity2_dict = self.activity2.to_dict()
        activity2_dict['skeleton_tokens'] = list()
        activity2_dict['tokens'] = list()
        for w_tuple in activity2_dict['skeleton_words']:
            if 'VB' in w_tuple[2]:
                activity2_dict['skeleton_tokens'].append(self.sentence_tokens[w_tuple[0] - 1])
            else:
                activity2_dict['skeleton_tokens'].append(w_tuple[1])
        for w_tuple in activity2_dict['words']:
            if 'VB' in w_tuple[2]:
                activity2_dict['tokens'].append(self.sentence_tokens[w_tuple[0] - 1])
            else:
                activity2_dict['tokens'].append(w_tuple[1])
        return {'activity1': activity1_dict, 'activity2': activity2_dict, 'relations': tmp_relations,
                'sentence_parsed_relations': self.sentence_parsed_relations, 'sentence_tokens': self.sentence_tokens}

    def to_activity_only_dict(self):
        return {'activity1': self.activity1.to_dict(), 'activity2': self.activity2.to_dict()}

    def to_activity_string(self):
        activity1_string = ''
        # print(self.sentence_tokens)
        # print(len(self.sentence_tokens))
        for w in self.activity1.words:
            # print(w)
            if self.sentence_tokens[w[0]-1] not in ["'s", "n't", "'m", "'S", "'M", 'na']:
                activity1_string += ' '
            activity1_string += self.sentence_tokens[w[0]-1]
            # activity1_string += w[1]
        if len(activity1_string) > 0:
            activity1_string = activity1_string[1:]
        # print('$')
        activity2_string = ''
        for w in self.activity2.words:
            # print(w)
            if self.sentence_tokens[w[0]-1] not in ["'s", "n't", "'m", "'S", "'M", 'na']:
                activity2_string += ' '
            activity2_string += self.sentence_tokens[w[0]-1]
            # activity2_string += w[1]
        if len(activity2_string) > 0:
            activity2_string = activity2_string[1:]
        return activity1_string, activity2_string

    def to_sentence_string(self):
        sentence_string = ''
        for word in self.sentence_tokens:
            sentence_string += ' '
            sentence_string += word
        return sentence_string[1:]

    def get_activity_average_position(self):
        return self.activity1.get_average_position(), self.activity2.get_average_position()

    def find_connective_position(self, connective_words):
        tmp_positions = list()
        for w in connective_words:
            tmp_positions.append(self.sentence_tokens.index(w))
        return sum(tmp_positions) / len(tmp_positions)

    def verify_contain_word(self, w):
        if w in self.sentence_tokens:
            return True
        else:
            return False

    def shrink_example(self):
        new_edges = list()
        for edge in self.sentence_parsed_relations:
            new_edge = list()
            if self.activity1.contain_tuple(edge[0]):
                new_edge.append('A')
            elif self.activity2.contain_tuple(edge[0]):
                new_edge.append('B')
            else:
                new_edge.append(edge[0][1])
            new_edge.append(edge[1])
            if self.activity1.contain_tuple(edge[2]):
                new_edge.append('A')
            elif self.activity2.contain_tuple(edge[2]):
                new_edge.append('B')
            else:
                new_edge.append(edge[2][1])
            if new_edge[0] != new_edge[2]:
                new_edges.append(tuple(new_edge))
        return new_edges

    def verify_connective(self, connective_words):
        connective_string = ''
        for word in connective_words:
            connective_string += ' '
            connective_string += word
        connective_string = connective_string[1:]
        sentence_string = self.to_sentence_string()
        if connective_string not in sentence_string:
            return False
        for w in connective_words:
            if not self.verify_contain_word(w):
                return False
        shrinked_edges = self.shrink_example()
        found_advcl = False
        for edge in shrinked_edges:
            if edge[0] == 'A' and edge[2] == 'B' and 'advcl' in edge[1]:
                found_advcl = True
        if not found_advcl:
            return False
        connective_position = self.find_connective_position(connective_words)
        e1_position, e2_position = self.get_activity_average_position()
        if 'instead' not in connective_words:
            if e1_position < connective_position < e2_position:
                return True
            else:
                return False
        else:
            if e1_position < e2_position < connective_position:
                return True
            else:
                return False

    def predict_relations(self):
        find_relations = list()
        for c_type in Seed_Connectives:
            for connective in Seed_Connectives[c_type]:
                if self.verify_connective(connective):
                    find_relations.append(c_type)
        return find_relations


class Two_Sentence_Training_Example:
    def __init__(self, initial_data):
        if initial_data is None:
            self.activity1 = Activity(None)
            self.activity2 = Activity(None)
            # self.sentence1 = ''
            # self.sentence2 = ''
            self.sentence1_parsed_relations = list()
            self.sentence2_parsed_relations = list()
            self.sentence1_tokens = list()
            self.sentence2_tokens = list()
        else:
            self.activity1 = Activity(initial_data['activity1'])
            self.activity2 = Activity(initial_data['activity2'])
            # self.sentence1 = initial_data['sentence1']
            # self.sentence2 = initial_data['sentence2']
            self.sentence1_parsed_relations = initial_data['sentence1_parsed_relations']
            self.sentence2_parsed_relations = initial_data['sentence2_parsed_relations']
            self.sentence1_tokens = initial_data['sentence1_tokens']
            self.sentence2_tokens = initial_data['sentence2_tokens']

    # def print(self):
    #     print('sentence1:', self.sentence1)

    def to_dict(self):
        tmp_relations = self.predict_relations()
        activity1_dict = self.activity1.to_dict()
        activity1_dict['skeleton_tokens'] = [self.sentence1_tokens[x[0] - 1] for x in activity1_dict['skeleton_words']]
        activity1_dict['tokens'] = [self.sentence1_tokens[x[0] - 1] for x in activity1_dict['words']]
        activity2_dict = self.activity2.to_dict()
        activity2_dict['skeleton_tokens'] = [self.sentence2_tokens[x[0] - 1] for x in activity2_dict['skeleton_words']]
        activity2_dict['tokens'] = [self.sentence2_tokens[x[0] - 1] for x in activity2_dict['words']]
        return {'activity1': activity1_dict, 'activity2': activity2_dict, 'relations': tmp_relations,
                'sentence1_parsed_relations': self.sentence1_parsed_relations,
                'sentence2_parsed_relations': self.sentence2_parsed_relations,
                'sentence1_tokens': self.sentence1_tokens,
                'sentence2_tokens': self.sentence2_tokens,
                'sentence_tokens': self.sentence1_tokens + ['$$'] + self.sentence2_tokens}

    def to_dict_with_temporal(self):
        tmp_relations = self.predict_relations()
        activity1_dict = self.activity1.to_dict()
        activity1_dict['skeleton_tokens'] = list()
        activity1_dict['tokens'] = list()
        for w_tuple in activity1_dict['skeleton_words']:
            if 'VB' in w_tuple[2]:
                activity1_dict['skeleton_tokens'].append(self.sentence1_tokens[w_tuple[0] - 1])
            else:
                activity1_dict['skeleton_tokens'].append(w_tuple[1])
        for w_tuple in activity1_dict['words']:
            if 'VB' in w_tuple[2]:
                activity1_dict['tokens'].append(self.sentence1_tokens[w_tuple[0] - 1])
            else:
                activity1_dict['tokens'].append(w_tuple[1])
        activity2_dict = self.activity2.to_dict()
        activity2_dict['skeleton_tokens'] = list()
        activity2_dict['tokens'] = list()
        for w_tuple in activity2_dict['skeleton_words']:
            if 'VB' in w_tuple[2]:
                activity2_dict['skeleton_tokens'].append(self.sentence2_tokens[w_tuple[0] - 1])
            else:
                activity2_dict['skeleton_tokens'].append(w_tuple[1])
        for w_tuple in activity2_dict['words']:
            if 'VB' in w_tuple[2]:
                activity2_dict['tokens'].append(self.sentence2_tokens[w_tuple[0] - 1])
            else:
                activity2_dict['tokens'].append(w_tuple[1])
        return {'activity1': activity1_dict, 'activity2': activity2_dict, 'relations': tmp_relations,
                'sentence1_parsed_relations': self.sentence1_parsed_relations,
                'sentence2_parsed_relations': self.sentence2_parsed_relations,
                'sentence1_tokens': self.sentence1_tokens,
                'sentence2_tokens': self.sentence2_tokens,
                'sentence_tokens': self.sentence1_tokens + ['$$'] + self.sentence2_tokens}

    def to_activity_only_dict(self):
        return {'activity1': self.activity1.to_dict(), 'activity2': self.activity2.to_dict()}

    def to_activity_string(self):
        # print(self.sentence1_tokens)
        # print(len(self.sentence1_tokens))
        # print(self.sentence2_tokens)
        # print(len(self.sentence2_tokens))
        activity1_string = ''
        for w in self.activity1.words:
            # print(w)
            if self.sentence1_tokens[w[0]-1] not in ["'s", "n't", "'m", "'S", "'M", 'na']:
                activity1_string += ' '
            activity1_string += self.sentence1_tokens[w[0]-1]
            # activity1_string += w[1]
        if len(activity1_string) > 0:
            activity1_string = activity1_string[1:]
        # print('$')
        activity2_string = ''
        for w in self.activity2.words:
            # print(w)
            if self.sentence2_tokens[w[0]-1] not in ["'s", "n't", "'m", "'S", "'M", 'na']:
                activity2_string += ' '
            activity2_string += self.sentence2_tokens[w[0]-1]
            # activity2_string += w[1]
        if len(activity2_string) > 0:
            activity2_string = activity2_string[1:]
        return activity1_string, activity2_string

    def to_sentence_string(self):
        sentence_string = ''
        for word in self.sentence1_tokens:
            sentence_string += ' '
            sentence_string += word
        for word in self.sentence2_tokens:
            sentence_string += ' '
            sentence_string += word
        return sentence_string[1:]

    def get_activity_average_position(self):
        return self.activity1.get_average_position(), self.activity2.get_average_position() + len(self.sentence1_tokens)

    def find_connective_position(self, connective_words):
        tmp_positions = list()
        for w in connective_words:
            if w in self.sentence1_tokens:
                tmp_positions.append(self.sentence1_tokens.index(w))
            else:
                if w in self.sentence2_tokens:
                    tmp_positions.append(self.sentence2_tokens.index(w) + len(self.sentence1_tokens))
        return sum(tmp_positions) / len(tmp_positions)

    def verify_contain_word(self, w):
        if w in self.sentence1_tokens or w in self.sentence2_tokens:
            return True
        else:
            return False

    def verify_connective(self, connective_words):
        connective_string = ''
        for word in connective_words:
            connective_string += ' '
            connective_string += word
        connective_string = connective_string[1:]
        sentence_string = self.to_sentence_string()
        if connective_string not in sentence_string:
            return False
        for w in connective_words:
            if not self.verify_contain_word(w):
                return False
        connective_position = self.find_connective_position(connective_words)
        e1_position, e2_position = self.get_activity_average_position()
        if 'instead' not in connective_words:
            if e1_position < connective_position < e2_position and e2_position - e1_position < 10:
                return True
            else:
                return False
        else:
            if e1_position < e2_position < connective_position and e2_position - e1_position < 10:
                return True
            else:
                return False

    def predict_relations(self):
        find_relations = list()
        for c_type in Seed_Connectives:
            for connective in Seed_Connectives[c_type]:
                if self.verify_connective(connective):
                    find_relations.append(c_type)
        return find_relations


