import gc
import time
from tqdm import tqdm
from aser.client import ASERClient


if __name__ == "__main__":
    gc.collect()
    client = ASERClient(port=8000)

    # s1 = 'they yucca look leggy'
    # s2 = 'they african violet refuse to flower'
    s1 = 'I am hungry'
    s2 = 'I am in the kitchen'

    event2 = client.extract_eventualities(s2)
    e2 = client.get_exact_match_event(event2)
    gc.collect()

    st = time.time()
    for i in range(1000):
        client.extract_eventualities_struct(s1)
    print("`extract_eventualities_struct`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()

    st = time.time()
    for i in range(1000):
        event1 = client.extract_eventualities(s1)
    print("`extract_eventualities`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()

    st = time.time()
    for i in range(1000):
        e1 = client.get_exact_match_event(event1)
    print("`get_exact_match_event`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()

    st = time.time()
    for i in range(1000):
        rel = client.get_exact_match_relation(e1, e2)
    print("`get_exact_match_relation`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()

    st = time.time()
    for i in range(1000):
        related_events = client.get_related_events(e1)
    print("`get_related_events`: {:.2f}ms / call".format(
        (time.time() - st)))
