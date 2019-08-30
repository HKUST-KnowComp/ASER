class Activity:
    def __init__(self, initial_relations):
        if initial_relations is None:
            self.parsed_relations = list()
            self.skeleton_parsed_relations = list()
            self.skeleton_words = list()
            self.words = list()
        else:
            self.parsed_relations = initial_relations['parsed_relations']
            self.skeleton_parsed_relations = initial_relations['skeleton_parsed_relations']
            self.skeleton_words = initial_relations['skeleton_words']
            self.words = initial_relations['words']

    def remove_one_edge(self, edge):
        new_parsed_relations = list()
        new_skeleton_parsed_relations = list()
        for old_edge in self.parsed_relations:
            if old_edge[0][0] == edge[0][0] and old_edge[2][0] == edge[2][0]:
                continue
            new_parsed_relations.append(old_edge)
        for old_edge in self.skeleton_parsed_relations:
            if old_edge[0][0] == edge[0][0] and old_edge[2][0] == edge[2][0]:
                continue
            new_skeleton_parsed_relations.append(old_edge)
        self.parsed_relations = new_parsed_relations
        self.skeleton_parsed_relations = new_skeleton_parsed_relations
        self.find_skeleton_words()

    def find_skeleton_words(self):
        all_skeleton_words = list()
        for relation in self.skeleton_parsed_relations:
            if relation[0] not in all_skeleton_words:
                all_skeleton_words.append(relation[0])
            if relation[2] not in all_skeleton_words:
                all_skeleton_words.append(relation[2])
        self.skeleton_words = sorted(all_skeleton_words, key=lambda tup: tup[0])

        all_words = list()
        for relation in self.parsed_relations:
            if relation[0] not in all_words:
                all_words.append(relation[0])
            if relation[2] not in all_words:
                all_words.append(relation[2])
        self.words = sorted(all_words, key=lambda tup: tup[0])

    def contain_word(self, w):
        for relation in self.parsed_relations:
            if relation[0][1] == w or relation[2][1] == w:
                return True
        return False

    def contain_tuple(self, t):
        for relation in self.parsed_relations:
            if relation[0] == t or relation[2] == t:
                return True
        return False

    def to_string(self):
        all_words = list()
        for relation in self.parsed_relations:
            if relation[0] not in all_words:
                all_words.append(relation[0])
            if relation[2] not in all_words:
                all_words.append(relation[2])
        sorted_words = sorted(all_words, key=lambda tup: tup[0])
        generated_string = ''
        for tmp_word in sorted_words:
            generated_string += ' '
            generated_string += tmp_word[1]
        return generated_string[1:]

    def to_unlemmatize_string(self, original_sentence):
        all_words = list()
        for relation in self.parsed_relations:
            if relation[0] not in all_words:
                all_words.append(relation[0])
            if relation[2] not in all_words:
                all_words.append(relation[2])
        sorted_words = sorted(all_words, key=lambda tup: tup[0])
        generated_string = ''
        for tmp_word in sorted_words:
            generated_string += ' '
            generated_string += original_sentence['tokens'][tmp_word[0] - 1]
        return generated_string[1:]

    def to_dict(self, original_sentence=None):
        if original_sentence:
            original_skeleton_tokens = list()
            original_tokens = list()
            for w_tuple in self.skeleton_words:
                if 'VB' in w_tuple[2]:
                    original_skeleton_tokens.append(original_sentence['tokens'][w_tuple[0] - 1])
                else:
                    original_skeleton_tokens.append(w_tuple[1])
            for w_tuple in self.words:
                if 'VB' in w_tuple[2]:
                    original_tokens.append(original_sentence['tokens'][w_tuple[0] - 1])
                else:
                    original_tokens.append(w_tuple)
        else:
            original_skeleton_tokens = list()
            original_tokens = list()
            for w_tuple in self.skeleton_words:
                if 'VB' in w_tuple[2]:
                    original_skeleton_tokens.append(w_tuple[1])
            for w_tuple in self.words:
                original_tokens.append(w_tuple)
        return {'parsed_relations': self.parsed_relations, 'skeleton_parsed_relations': self.skeleton_parsed_relations,
                'skeleton_words': self.skeleton_words, 'words': self.words, 'skeleton_tokens': original_skeleton_tokens,
                'tokens': original_tokens}

    def get_average_position(self):
        positions = list()
        for t in self.words:
            positions.append(t[0])
        try:
            average_1_position = sum(positions) / len(positions)
        except:
            print(self.to_dict())
            average_1_position = sum(positions) / len(positions)
        return average_1_position
