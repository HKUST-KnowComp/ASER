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
        self._construct()

    def _construct(self):
        sort_dependencies_position(self.dependencies, fix_position=False)
        self.words = extract_tokens_from_dependencies(self.dependencies)
        sort_dependencies_position(self.skeleton_dependencies, fix_position=False)
        self.skeleton_words = extract_tokens_from_dependencies(self.dependencies)
        self.verbs = ' '.join(
            [x[0].lower() for x in self.skeleton_words if x[1].startswith('VB')])
        self.eid = self._generate_eid(self.words)

    def _generate_eid(self, words):
        key = ' '.join([x[0].lower() for x in words])
        return hashlib.sha1(key.encode('utf-8')).hexdigest()

    def to_dict(self):
        rst = {
            "eid": self.eid,
            "pattern": self.pattern,
            "frequency": self.frequency,
            "dependencies": self.dependencies,
            "words": self.words,
            "skeleton_dependencies": self.skeleton_dependencies,
            "skeleton_words": self.skeleton_words,
            "verbs": self.verbs,

        }
        return rst