import hashlib
import json
import pprint
import bisect
from collections import Counter
from copy import copy
from aser.base import JsonSerializedObject
from aser.extract.utils import sort_dependencies_position, extract_indices_from_dependencies

class Eventuality(JsonSerializedObject):
    def __init__(self, pattern="unknown", dependencies=None,
                 skeleton_dependencies=None,
                 sent_parsed_result=None):
        super().__init__()
        self.eid = None
        self.pattern = pattern
        self._dependencies = None
        self.words = None
        self.pos_tags = None
        self._ners = None
        self._mentions = None
        self._skeleton_dependency_indices = None
        self._skeleton_indices = None
        self._verb_indices = None
        self.raw_sent_mapping = None
        self._phrase_segment_indices = None
        self.frequency = 1.0
        if dependencies and skeleton_dependencies and sent_parsed_result:
            self._construct(dependencies, skeleton_dependencies, sent_parsed_result)

    @staticmethod
    def generate_eid(eventuality):
        msg = json.dumps(
            [eventuality.dependencies, eventuality.words, eventuality.pos_tags])
        return hashlib.sha1(msg.encode('utf-8')).hexdigest()

    def update(self, x):
        if x is not None:
            if isinstance(x, float):
                self.frequency += x
            elif isinstance(x, Eventuality):
                if x.eid == self.eid:
                    if self._ners is not None and x._ners is not None:
                        for i, (ner, x_ner) in enumerate(zip(self._ners, x._ners)):
                            if isinstance(ner, str) and isinstance(x_ner, str) and ner == x_ner:
                                continue
                            if isinstance(ner, str):
                                self._ners[i] = Counter({ner: self.frequency})
                            if isinstance(x_ner, str):
                                x_ner = Counter({x_ner: x.frequency})
                            self._ners[i].update(x_ner)
                    if self._mentions is not None and x._mentions is not None:
                        for s_t, x_mention in x._mentions.items():
                            self._mentions[s_t] = x_mention
                    self.frequency += x.frequency

    def __len__(self):
        return len(self.words)

    def __str__(self):
        repr_dict = {
            "eid": self.eid,
            "pattern": self.pattern,
            "verbs": self.verbs,
            "skeleton_words": self.skeleton_words,
            "words": self.words,
            "skeleton_dependencies": self.skeleton_dependencies,
            "dependencies": self.dependencies,
            "pos_tags": self.pos_tags,
        }
        if self._ners is not None:
            repr_dict["ners"] = self._ners
        if self._mentions is not None:
            repr_dict["mentions"] = self._mentions
        repr_dict["frequency"] = self.frequency
        return pprint.pformat(repr_dict)

    def __repr__(self):
        return " ".join(self.words)

    @property
    def dependencies(self):
        return self._render_dependencies(self._dependencies)

    @property
    def ners(self):
        if self._ners is None:
            return None
        return [self._get_ner(idx) for idx in range(len(self._ners))]


    @property
    def mentions(self):
        if self._ners is None:
            return None
        _mentions = list()
        len_words = len(self.words)
        i = 0
        while i < len_words:
            ner = self._get_ner(i)
            if ner == "O":
                i += 1
                continue
            j = i + 1
            while j < len_words and self._get_ner(j) == ner:
                j += 1
            mention = self._mentions.get((i, j), 
                {"start": i,
                "end": j,
                "text": self.words[i:j],
                "ner": ner,
                "link": None,
                "entity": None})
            _mentions.append(mention)
            i = j
        return _mentions

    @property
    def _raw_dependencies(self):
        if self.raw_sent_mapping is None:
            return self._dependencies
        new_dependencies = list()
        for governor, dep, dependent in self._dependencies:
            new_dependencies.append((
                self.raw_sent_mapping[governor],
                dep,
                self.raw_sent_mapping[dependent]))
        return new_dependencies

    @property
    def raw_dependencies(self):
        if self.raw_sent_mapping is None:
            return self.dependencies
        tmp_dependencies = self.dependencies
        new_dependencies = list()
        for governor, dep, dependent in tmp_dependencies:
            g_pos, g_word, g_tag = governor
            d_pos, d_word, d_tag = dependent
            new_dependencies.append((
                (self.raw_sent_mapping[g_pos], g_word, g_tag),
                dep,
                (self.raw_sent_mapping[d_pos], d_word, d_tag)))
        return new_dependencies

    @property
    def skeleton_dependencies(self):
        dependencies = [self._dependencies[i] for i in self._skeleton_dependency_indices]
        return self._render_dependencies(dependencies)

    @property
    def skeleton_words(self):
        return [self.words[idx] for idx in self._skeleton_indices]

    @property
    def skeleton_pos_tags(self):
        return [self.pos_tags[idx] for idx in self._skeleton_indices]

    @property
    def skeleton_ners(self):
        if self._ners is None:
            return None
        return [self._get_ner(idx) for idx in self._skeleton_indices]

    @property
    def verbs(self):
        return [self.words[idx] for idx in self._verb_indices]

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

    @property
    def phrases(self):
        return [" ".join(self.words[st:end]) for st, end in self._phrase_segment_indices]

    @property
    def phrases_ners(self):
        _tmp_ners = list()

        for _range in self._phrase_segment_indices:
            _tmp_ners.append(self._ners[_range[0]])

        return _tmp_ners

    @property
    def skeleton_phrases(self):
        phrase_dict = dict()
        for _range in self._phrase_segment_indices:
            for i in range(*_range):
                phrase_dict[i] = _range

        _skeleton_tokens = list()
        for idx in self._skeleton_indices:
            st, end = phrase_dict[idx]
            _skeleton_tokens.append(" ".join(self.words[st:end]))

        return _skeleton_tokens

    @property
    def skeleton_phrases_ners(self):
        return self.skeleton_ners

    def _construct(self, dependencies, skeleton_dependencies, sent_parsed_result):
        word_indices = extract_indices_from_dependencies(dependencies)
        if sent_parsed_result["pos_tags"][word_indices[0]] == "IN":
            poped_idx = word_indices[0]
            for i in range(len(dependencies)-1, -1, -1):
                if dependencies[i][0] == poped_idx or \
                    dependencies[i][2] == poped_idx:
                    dependencies.pop(i)
            for i in range(len(skeleton_dependencies)-1, -1, -1):
                if skeleton_dependencies[i][0] == poped_idx or \
                    skeleton_dependencies[i][2] == poped_idx:
                    skeleton_dependencies.pop(i)
            word_indices.pop(0)
        len_words = len(word_indices)
        self.words = [sent_parsed_result["lemmas"][i].lower() for i in word_indices]
        self.pos_tags = [sent_parsed_result["pos_tags"][i] for i in word_indices]
        if "ners" in sent_parsed_result:
            self._ners = [sent_parsed_result["ners"][i] for i in word_indices]
        if "mentions" in sent_parsed_result:
            self._mentions = dict()
            for mention in sent_parsed_result["mentions"]:
                start_idx = bisect.bisect_left(word_indices, mention["start"])
                if not (start_idx < len_words and word_indices[start_idx] == mention["start"]):
                    continue
                end_idx = bisect.bisect_left(word_indices, mention["end"]-1)
                if not (end_idx < len_words and word_indices[end_idx] == mention["end"]-1):
                    continue
                mention = copy(mention)
                mention["start"] = start_idx
                mention["end"] = end_idx+1
                mention["text"] = " ".join(self.words[mention["start"]:mention["end"]])
                self._mentions[(mention["start"], mention["end"])] = mention
        dependencies, raw2reset_idx, reset2raw_idx = sort_dependencies_position(
            dependencies, reset_position=True)
        self._dependencies = dependencies
        self.raw_sent_mapping = reset2raw_idx

        skeleton_word_indices = extract_indices_from_dependencies(
            skeleton_dependencies)
        self._skeleton_indices = [raw2reset_idx[idx] for idx in skeleton_word_indices]

        _skeleton_dependencies, _, _ = sort_dependencies_position(
            skeleton_dependencies, reset_position=False)
        skeleton_dependency_indices = list()
        ptr = 0
        for i, dep in enumerate(dependencies):
            if ptr >= len(_skeleton_dependencies):
                break
            skeleton_dep = _skeleton_dependencies[ptr]
            skeleton_dep = (raw2reset_idx[skeleton_dep[0]], skeleton_dep[1], raw2reset_idx[skeleton_dep[2]])
            if dep == skeleton_dep:
                skeleton_dependency_indices.append(i)
                ptr += 1
        self._skeleton_dependency_indices = skeleton_dependency_indices

        self._verb_indices = [i for i, tag in enumerate(self.pos_tags) if tag.startswith('VB')]

        self._phrase_segment_indices = self._phrase_segment()

        self.eid = Eventuality.generate_eid(self)

    def to_dict(self, **kw):
        minimum = kw.get("minimum", False)
        if minimum:
            d = {
                "_dependencies": self._dependencies,
                "words": self.words,
                "pos_tags": self.pos_tags,
                "_ners": self._ners,
                "_mentions": self._mentions,
                "_verb_indices": self._verb_indices,
                "_skeleton_indices": self._skeleton_indices,
                "_skeleton_dependency_indices": self._skeleton_dependency_indices}
        else:
            d  = self.__dict__
        return d

    def encode(self, encoding="utf-8", **kw):
        d = self.to_dict(**kw)
        # key cannot be tuple
        if d["_mentions"]:
            d["_mentions"] = {str(k): v for k, v in d["_mentions"].items()}
        if encoding == "utf-8":
            msg = json.dumps(d).encode("utf-8")
        elif encoding == "ascii":
            msg = json.dumps(d).encode("ascii")
        else:
            msg = d
        return msg

    def decode(self, msg, encoding="utf-8", **kw):
        if encoding == "utf-8":
            decoded_dict = json.loads(msg.decode("utf-8"))
        elif encoding == "ascii":
            decoded_dict = json.loads(msg.decode("ascii"))
        else:
            decoded_dict = msg
        self.from_dict(decoded_dict, **kw)
        
        if self.raw_sent_mapping is not None:
            keys = list(self.raw_sent_mapping.keys())
            for key in keys:
                if not isinstance(key, int):
                    self.raw_sent_mapping[int(key)] = self.raw_sent_mapping.pop(key)
        if self._ners is not None:
            for i, ner in enumerate(self._ners):
                if not isinstance(ner, str):
                    self._ners[i] = Counter(ner)
        if self._mentions is not None:
            keys = list(self._mentions.keys())
            for key in keys:
                self._mentions[eval(key)] = self._mentions.pop(key)
        self._phrase_segment_indices = self._phrase_segment()
        return self

    def _render_dependencies(self, dependencies):
        edges = list()
        for governor_idx, dep, dependent_idx in dependencies:
            edge = ((governor_idx, self.words[governor_idx], self.pos_tags[governor_idx]),
                    dep,
                    (dependent_idx, self.words[dependent_idx], self.pos_tags[dependent_idx]))
            edges.append(edge)
        return edges

    def _get_ner(self, index):
        ner = "O"
        if not self.pos_tags[index].startswith('VB'):
            if isinstance(self._ners[index], str):
                ner = self._ners[index]
            else:
                for x in self._ners[index].most_common():
                    if x[0] != "O":
                        ner = x[0]
                        break
        return ner

    def _pos_compound_segment(self):
        tmp_compound_tuples = list()
        for governor_idx, dep, dependent_idx in self._dependencies:
            if dep.startswith("compound"):
                tmp_compound_tuples.append((governor_idx, dependent_idx))

        compound_tuples = list()
        i = 0
        while True:
            if i >= len(tmp_compound_tuples):
                break
            if i == len(tmp_compound_tuples) - 1:
                compound_tuples.append(tuple(sorted(tmp_compound_tuples[i])))
                break
            s1 = set(tmp_compound_tuples[i])
            while True:
                if i == len(tmp_compound_tuples) - 1:
                    i += 1
                    break
                i += 1
                s2 = set(tmp_compound_tuples[i])
                if s1 & s2:
                    s1 |= s2
                else:
                    break
            compound_tuples.append(tuple(sorted(s1)))

        segment_rst = list()
        ptr = 0
        i = 0
        while True:
            if i >= len(self.words):
                break
            if ptr < len(compound_tuples) and i == compound_tuples[ptr][0]:
                segment_rst.append((compound_tuples[ptr][0], compound_tuples[ptr][-1] + 1))
                i = compound_tuples[ptr][-1] + 1
                ptr += 1
            else:
                segment_rst.append((i, i + 1))
                i += 1
        return segment_rst

    def _ner_compound_segment(self):
        if self._ners is None:
            return [(i, i + 1) for i in range(len(self.words))]
        compound_tuples = list()
        for idx in self._skeleton_indices:
            ner = self._get_ner(idx)
            if ner != "O":
                st = idx - 1
                while st >= 0 and self._get_ner(st) == ner:
                    st -= 1

                end = idx + 1
                post_tokens = list()
                while end < len(self.words) and self._get_ner(end) == ner:
                    post_tokens.append(self.words[end])
                    end += 1
                compound_tuples.append((st, end))

        segment_rst = list()
        ptr = 0
        i = 0
        while True:
            if i >= len(self.words):
                break
            if ptr < len(compound_tuples) and i == compound_tuples[ptr][0]:
                segment_rst.append((compound_tuples[ptr][0], compound_tuples[ptr][-1]))
                i = compound_tuples[ptr][-1]
                ptr += 1
            else:
                segment_rst.append((i, i + 1))
                i += 1
        return segment_rst

    def _phrase_segment(self):
        return self._pos_compound_segment()