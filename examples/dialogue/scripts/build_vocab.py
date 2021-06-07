from collections import Counter
from dialogue.toolbox.vocab import Vocabulary, UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD, get_pretrained_embedding
import sys
import torch
from tqdm import tqdm
import json

WORD_VOCAB_SIZE = 15000
ASER_VOCAB_SIZE = 40000
ASER_EVENT_VOCAB_SIZE = 30000
OMCS_VOCAB_SIZE = 3000
OMCS_EVENT_VOCAB_SIZE = 3000
KNOWLY_VOCAB_SIZE = 45000
KNOWLY_EVENT_VOCAB_SIZE = 40000


def build_vocabs(counters):
    word_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD, BOS_WORD, EOS_WORD])
    word_vocab.build_from_counter(counters["word"], WORD_VOCAB_SIZE)

    aser_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    aser_vocab.build_from_counter(counters["aser"], ASER_VOCAB_SIZE)

    aser_event_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    aser_event_vocab.build_from_counter(counters["aser_event"], ASER_EVENT_VOCAB_SIZE)

    aser_rel_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    aser_rel_vocab.build_from_counter(counters["aser_relation"], 15)

    omcs_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    omcs_vocab.build_from_counter(counters["omcs"], OMCS_VOCAB_SIZE)

    omcs_event_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    omcs_event_vocab.build_from_counter(counters["omcs_event"], OMCS_EVENT_VOCAB_SIZE)

    omcs_rel_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    omcs_rel_vocab.build_from_counter(counters["omcs_relation"], 4)

    knowlywood_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    knowlywood_vocab.build_from_counter(counters["knowlywood"], KNOWLY_VOCAB_SIZE)

    knowlywood_event_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    knowlywood_event_vocab.build_from_counter(counters["knowlywood_event"], KNOWLY_EVENT_VOCAB_SIZE)

    knowlywood_rel_vocab = Vocabulary(special_tokens=[UNK_WORD, PAD_WORD])
    knowlywood_rel_vocab.build_from_counter(counters["knowlywood_relation"], 4)

    vocabs = {
        "word": word_vocab,
        "pre_word_emb": None,
        "aser": aser_vocab,
        "aser_event": aser_event_vocab,
        "aser_relation": aser_rel_vocab,
        "omcs": omcs_vocab,
        "omcs_event": omcs_event_vocab,
        "omcs_relation": omcs_rel_vocab,
        "knowlywood": knowlywood_vocab,
        "knowlywood_event": knowlywood_event_vocab,
        "knowlywood_relation": knowlywood_rel_vocab,
    }
    return vocabs


if __name__ == "__main__":
    word_counter = Counter()

    aser_counter = Counter()
    aser_event_counter = Counter()
    aser_rel_counter = Counter()

    omcs_counter = Counter()
    omcs_event_counter = Counter()
    omcs_rel_counter = Counter()

    knowlywood_counter = Counter()
    knowlywood_event_counter = Counter()
    knowlywood_rel_counter = Counter()

    with open(sys.argv[1]) as f:
        for line in tqdm(f):
            record = json.loads(line)
            word_counter.update(record["post"].lower().split())
            word_counter.update(record["response"].lower().split())

            aser_counter.update(record["aser_triples"])
            post_aser_events = []
            post_aser_rels = []
            for event_triple in record["aser_triples"]:
                e1, r, e2 = event_triple.split("$")
                post_aser_events.extend([e1, e2])
                post_aser_rels.append(r)
            aser_event_counter.update(post_aser_events)
            aser_rel_counter.update(post_aser_rels)

            omcs_counter.update(record["omcs_triples"])
            post_omcs_entities = []
            post_omcs_rels = []
            for fact in record["omcs_triples"]:
                e1, r, e2 = fact.split("$")
                post_omcs_entities.extend([e1, e2])
                post_omcs_rels.append(r)
            omcs_event_counter.update(post_omcs_entities)
            omcs_rel_counter.update(post_omcs_rels)

            knowlywood_counter.update(record["knowlywood_triples"])
            post_knowlywood_entities = []
            post_knowlywood_rels = []
            for fact in record["knowlywood_triples"]:
                e1, r, e2 = fact.split("$")
                post_knowlywood_entities.extend([e1, e2])
                post_knowlywood_rels.append(r)
            knowlywood_event_counter.update(post_knowlywood_entities)
            knowlywood_rel_counter.update(post_knowlywood_rels)

    print("Word counter size: ", len(word_counter))
    print("Aser counter size: ", len(aser_counter))
    print("Aser event counter size: ", len(aser_event_counter))
    print("Aser relation counter size: ", len(aser_rel_counter))
    print("Omcs counter size: ", len(omcs_counter))
    print("Omcs event size: ", len(omcs_event_counter))
    print("Omcs relation size: ", len(omcs_rel_counter))
    print("KnowlyWood counter size: ", len(knowlywood_counter))
    print("KnowlyWood event size: ", len(knowlywood_event_counter))
    print("KnowlyWood relation size: ", len(knowlywood_rel_counter))

    counters = {
                "word": word_counter,
                "aser": aser_counter,
                "aser_event": aser_event_counter,
                "aser_relation": aser_rel_counter,
                "omcs": omcs_counter,
                "omcs_event": omcs_event_counter,
                "omcs_relation": omcs_rel_counter,
                "knowlywood": knowlywood_counter,
                "knowlywood_event": knowlywood_event_counter,
                "knowlywood_relation": knowlywood_rel_counter
                }
    vocabs = build_vocabs(counters)
    torch.save(vocabs, sys.argv[2])
