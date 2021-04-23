from ..extract.utils import PRONOUN_SET


def compute_overlap(w1, w2):
    """ Compute the overlap between two word list by Jaccard

    :param w1: one word list
    :type w1: List[str]
    :param w2: the other word list
    :type w2: List[str]
    :return: the Jaccard similarity
    :rtype: float
    """
    w1_words = set(w1) - PRONOUN_SET
    w2_words = set(w2) - PRONOUN_SET
    Jaccard = len(w1_words & w2_words) / len(w1_words | w2_words)
    return Jaccard

