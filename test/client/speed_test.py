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

    event2 = client.extract_eventualities(s2, only_events=True)[0]
    gc.collect()

    st = time.time()
    for i in range(1000):
        event1 = client.extract_eventualities(s1, only_events=True)[0]
    print("`extract_eventualities`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()


    st = time.time()
    for i in range(1000):
        rel = client.predict_relation(event1, event2)
    print("`get_exact_match_relation`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()

    st = time.time()
    for i in range(1000):
        related_events = client.fetch_related_events(event1)
    print("`get_related_events`: {:.2f}ms / call".format(
        (time.time() - st)))
