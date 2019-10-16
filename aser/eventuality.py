import hashlib
import json
import pprint
from aser.extract.utils import sort_dependencies_position, extract_indices_from_dependencies


class Eventuality(object):
    def __init__(self, pattern=None, dependencies=None,
                 skeleton_dependencies=None,
                 sent_parsed_results=None):
        self.eid = None
        self.pattern = pattern
        self._dependencies = None
        self.words = None
        self.pos_tags = None
        self._skeleton_dependencies = None
        self._skeleton_words = None
        self._verbs = None
        self.raw_sent_mapping = None
        self.frequency = 1.0
        if pattern and dependencies and skeleton_dependencies and sent_parsed_results:
            self._construct(dependencies, skeleton_dependencies, sent_parsed_results)

    def __str__(self):
        repr_dict = {
            "eid": self.eid,
            "pattern": self.pattern,
            "dependencies": self.dependencies,
            "skeleton_dependencies": self.skeleton_dependencies,
            "skeleton_words": self.skeleton_words,
            "verbs": self.verbs,
            "words": self.words,
            "pos_tags":self.pos_tags
        }
        return pprint.pformat(repr_dict)

    def __repr__(self):
        return " ".join(self.words)

    @property
    def dependencies(self):
        return self._render_dependencies(self._dependencies)

    @property
    def _raw_dependencies(self):
        if not self.raw_sent_mapping:
            return self._dependencies
        new_dependencies = list()
        for governor, dep, dependent in self._dependencies:
            new_dependencies.append(
                (self.raw_sent_mapping[governor],
                 dep,
                 self.raw_sent_mapping[dependent])
            )
        return new_dependencies

    @property
    def raw_dependencies(self):
        if not self.raw_sent_mapping:
            return self.dependencies
        tmp_dependencies = self.dependencies
        new_dependencies = list()
        for governor, dep, dependent in tmp_dependencies:
            g_pos, g_word, g_tag = governor
            d_pos, d_word, d_tag = dependent
            new_dependencies.append(
                ((self.raw_sent_mapping[g_pos], g_word, g_tag),
                 dep,
                 (self.raw_sent_mapping[d_pos], d_word, d_tag))
            )
        return new_dependencies

    @property
    def skeleton_dependencies(self):
        dependencies = [self._dependencies[i] for i in self._skeleton_dependencies]
        return self._render_dependencies(dependencies)

    @property
    def skeleton_words(self):
        return [self.words[idx] for idx in self._skeleton_words]

    @property
    def skeleton_pos_tags(self):
        return [self.pos_tags[idx] for idx in self._skeleton_words]

    @property
    def verbs(self):
        return [self.words[idx] for idx in self._verbs]

    @property
    def position(self):
        """
        :return: this property returns average position of eventuality in a sentence.
                 this property only make sense when this eventuality are constructed while
                 extraction, instead of recovered from database.
        """
        positions = set()
        for governor, _, dependent in self._dependencies:
            positions.add(self.raw_sent_mapping[governor])
            positions.add(self.raw_sent_mapping[dependent])
        avg_position = sum(positions) / len(positions) if positions else 0.0
        return avg_position

    def _construct(self, dependencies, skeleton_dependencies, sent_parsed_results):
        word_indices = extract_indices_from_dependencies(dependencies)
        self.words = [sent_parsed_results["lemmas"][i].lower() for i in word_indices]
        self.pos_tags = [sent_parsed_results["pos_tags"][i] for i in word_indices]
        dependencies, raw2reset_idx, reset2raw_idx = sort_dependencies_position(
            dependencies, reset_position=True)
        self._dependencies = dependencies
        self.raw_sent_mapping = reset2raw_idx

        skeleton_word_indices = extract_indices_from_dependencies(skeleton_dependencies)
        self._skeleton_words = [raw2reset_idx[idx] for idx in skeleton_word_indices]

        _skeleton_dependencies, _, _ = sort_dependencies_position(
            skeleton_dependencies, reset_position=False)
        skeleton_dependency_indices = list()
        ptr = 0
        for i, dep in enumerate(dependencies):
            if ptr >= len(_skeleton_dependencies):
                break
            skeleton_dependency = _skeleton_dependencies[ptr][:]
            triple = (raw2reset_idx[skeleton_dependency[0]],
                      skeleton_dependency[1],
                      raw2reset_idx[skeleton_dependency[2]])
            if tuple(dep) == triple:
                skeleton_dependency_indices.append(i)
                ptr += 1
        self._skeleton_dependencies = skeleton_dependency_indices

        self._verbs = [i for i, tag in enumerate(self.pos_tags) if tag.startswith('VB')]

        self.eid = self._generate_eid(self.words)

    def encode(self, encoding="utf-8"):
        if encoding == "utf-8":
            msg = json.dumps(self.__dict__).encode("utf-8")
        elif encoding == "ascii":
            msg = json.dumps(self.__dict__).encode("ascii")
        else:
            msg = self.__dict__
        return msg

    def decode(self, msg, encoding="utf-8"):
        if encoding == "utf-8":
            decoded_dict = json.loads(msg.decode("utf-8"))
        elif encoding == "ascii":
            decoded_dict = json.loads(msg.decode("ascii"))
        else:
            decoded_dict = msg
        self.from_dict(decoded_dict)
        return self

    def to_dict(self):
        return self.__dict__

    def from_dict(self, d):
        for attr_name in self.__dict__:
            self.__setattr__(attr_name, d[attr_name])
        keys = self.raw_sent_mapping.keys()
        if not all([isinstance(key, int) for key in keys]):
            for key in keys:
                self.raw_sent_mapping[int(key)] = self.raw_sent_mapping.pop(key)
        return self

    def _render_dependencies(self, dependencies):
        edges = list()
        for governor_idx, dep, dependent_idx in dependencies:
            edge = ((governor_idx, self.words[governor_idx], self.pos_tags[governor_idx]),
                    dep,
                    (dependent_idx, self.words[dependent_idx], self.pos_tags[dependent_idx]))
            edges.append(edge)
        return edges

    def _generate_eid(self, words):
        key = ' '.join([x[0].lower() for x in words])
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def _filter_dependency_by_word_list(self, word_list, target="all"):
        position_mapping = dict()
        new_dependencies = list()
        for i, (governor, dep, dependent) in enumerate(self._dependencies):
            if target == "all":
                is_find_word = self.words[governor] in word_list or \
                               self.words[dependent] in word_list
            elif target == "governor":
                is_find_word = self.words[governor] in word_list
            elif target == "dependent":
                is_find_word = self.words[dependent] in word_list
            else:
                raise RuntimeError
            if is_find_word:
                continue
            position_mapping[i] = len(new_dependencies)
            new_dependencies.append([governor, dep, dependent])
        self._dependencies = new_dependencies
        new_skeleton_dependencies = list()
        for idx in self._skeleton_dependencies:
            if idx in position_mapping:
                new_skeleton_dependencies.append(position_mapping[idx])
        self._skeleton_dependencies = new_skeleton_dependencies

        new_word_indices = extract_indices_from_dependencies(
            self._dependencies)
        new_word_index_mapping = dict()
        for i in new_word_indices:
            new_word_index_mapping[i] = len(new_word_index_mapping)
        self.words = [self.words[i] for i in new_word_indices]
        self.pos_tags = [self.pos_tags[i] for i in new_word_indices]
        for dependency in self._dependencies:
            dependency[0] = new_word_index_mapping[dependency[0]]
            dependency[2] = new_word_index_mapping[dependency[2]]
        self._verbs = [new_word_index_mapping[i] for i in self._verbs 
                                if i in new_word_index_mapping]
        self._skeleton_words = [new_word_index_mapping[i] for i in self._skeleton_words
                                if i in new_word_index_mapping]

        if self.raw_sent_mapping:
            new_raw_sent_mapping = {new_word_index_mapping[key]: val
                                    for key, val in self.raw_sent_mapping.items()
                                    if key in new_word_index_mapping}
            self.raw_sent_mapping = new_raw_sent_mapping



class EventualityList(object):
    def __init__(self, eventualities=None):
        if eventualities:
            self.eventualities = eventualities
        else:
            self.eventualities = list()

    def append(self, eventuality):
        self.eventualities.append(eventuality)

    def extend(self, eventualities):
        self.eventualities.extend(eventualities)

    def encode(self, encoding="utf-8"):
        encoded_dict_list = list()
        for e in self.eventualities:
            encoded_dict_list.append(e.encode(encoding=None))
        if encoding == "utf-8":
            msg = json.dumps(encoded_dict_list).encode("utf-8")
        else:
            msg = encoded_dict_list
        return msg

    def decode(self, msg, encoding="utf-8"):
        if encoding == "utf-8":
            decoded_dict_list = json.loads(msg.decode("utf-8"))
        elif encoding == "ascii":
            decoded_dict_list = json.loads(msg.decode("ascii"))
        else:
            decoded_dict_list = msg
        self.eventualities = []
        for decoded_dict in decoded_dict_list:
            e = Eventuality()
            e.decode(decoded_dict, encoding="None")
            self.eventualities.append(e)
        
    def filter_by_frequency(self, lower_bound=None, upper_bound=None):
        if not lower_bound and not upper_bound:
            return
        if not lower_bound:
            lower_bound = 0.0
        if not upper_bound:
            upper_bound = float("inf")
        self.eventualities = list(filter(lambda e: e.frequency >= lower_bound and e.frequency <= upper_bound, self.eventualities))

    def __iter__(self):
        return self.eventualities.__iter__()

    def __str__(self):
        s = "[\n"
        s += "\n".join(e.__str__() for e in self.eventualities)
        s += '\n]'
        return s

    def __repr__(self):
        return self.__str__()

    def __setitem__(self, idx, item):
        self.eventualities[idx] = item

    def __getitem__(self, item):
        return self.eventualities[item]

    def __len__(self):
        return len(self.eventualities)