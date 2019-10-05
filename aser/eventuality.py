import hashlib
from aser.extract.utils import sort_dependencies_position, extract_tokens_from_dependencies


class Eventuality(object):
    def __init__(self, pattern=None, dependencies=None,
                 skeleton_dependencies=None):
        self.eid = None
        self.pattern = pattern
        self.dependencies = dependencies
        self.words = list()
        self.skeleton_dependencies = skeleton_dependencies
        self.skeleton_words = list()
        self.verbs = ''
        self.frequency = 0.0
        if pattern and dependencies and skeleton_dependencies:
            self._construct()

    def _construct(self):
        sort_dependencies_position(self.dependencies, fix_position=False)
        self.words = extract_tokens_from_dependencies(self.dependencies)
        sort_dependencies_position(self.skeleton_dependencies, fix_position=False)
        self.skeleton_words = extract_tokens_from_dependencies(self.skeleton_dependencies)
        self.verbs = ' '.join(
            [x[0].lower() for x in self.skeleton_words if x[1].startswith('VB')])
        self.eid = self._generate_eid(self.words)

    def _generate_eid(self, words):
        key = ' '.join([x[0].lower() for x in words])
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def to_dict(self):
        return self.__dict__

    def from_dict(self, d):
        for attr_name in self.__dict__:
            self.__setattr__(attr_name, d[attr_name])
        return self

    @property
    def position(self):
        """
        :return: this property returns average position of eventuality in a sentence.
                 this property only make sense when this eventuality are constructed while
                 extraction, instead of recovered from database.
        """
        positions = set()
        for governor, _, dependent in self.dependencies:
            positions.add(governor[0])
            positions.add(dependent[0])
        avg_position = sum(positions) / len(positions) if positions else 0.0
        return avg_position
