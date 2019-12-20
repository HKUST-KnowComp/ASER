import hashlib
from aser.extract.utils import PRONOUN_SET

def compute_overlap(w1, w2):
    w1_words = set(w1) - PRONOUN_SET
    w2_words = set(w2) - PRONOUN_SET
    Jaccard = len(w1_words & w2_words) / len(w1_words | w2_words)
    return Jaccard
