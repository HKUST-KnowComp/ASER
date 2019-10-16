import hashlib

stopwords = {'i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
             'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their', 'theirs', 'themselves',
             'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those'}

def compute_overlap(w1, w2):
    w1_words = set(w1) - stopwords
    w2_words = set(w2) - stopwords
    Jaccard = len(w1_words & w2_words) / len(w1_words | w2_words)
    return Jaccard
