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

    eventuality2 = client.extract_eventualities(s2, only_eventualities=True)[0]
    gc.collect()

    st = time.time()
    for i in range(1000):
        eventuality1 = client.extract_eventualities(s1, only_eventualities=True)[0]
    print("`extract_eventualities`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()


    st = time.time()
    for i in range(1000):
        rel = client.predict_relation(eventuality1, eventuality2)
    print("`get_exact_match_relation`: {:.2f}ms / call".format(
        (time.time() - st)))
    gc.collect()

    st = time.time()
    for i in range(1000):
        related_events = client.fetch_related_eventualities(eventuality1)
    print("`get_related_events`: {:.2f}ms / call".format(
        (time.time() - st)))
