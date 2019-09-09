import json
from multiprocessing import Pool
import traceback
import time
import numpy as np
from aser.client import ASERClient


ASER_PORT = 8000

s1 = "I am hungry"
s2 = "I am in the kitchen"

# s1 = 'they yucca look leggy'
# s2 = 'they african violet refuse to flower'

s3 = "I have no money"



# Helper Function `extract_eventualities`
def fn1(s, n_iter, ans, prefix):
    client = ASERClient(port=ASER_PORT)
    st = time.time()
    for _ in range(n_iter):
        try:
            tmp = client.extract_eventualities(s)
            assert json.dumps(tmp) == ans, "[{}] Wrong answer".format(prefix)
        except Exception:
            print("[{}] Wrong answer".format(prefix))
            print(traceback.format_exc())
            client.close()
            return
    end = time.time()
    duration = (end - st) / n_iter * 1000
    print("[{}] passed, {:.2f} ms / call".format(
        prefix, duration))
    client.close()
    return (end - st) / n_iter * 1000

# Helper Function `predict_relation`
def fn2(e1, e2, n_iter, ans, prefix):
    client = ASERClient(port=ASER_PORT)
    st = time.time()
    for _ in range(n_iter):
        try:
            tmp = client.predict_relation(e1, e2)
            assert json.dumps(tmp) == ans, "[{}] Wrong answer".format(prefix)
        except Exception:
            print("[{}] Wrong answer".format(prefix))
            print(traceback.format_exc())
            client.close()
            return

    end = time.time()
    print("[{}] passed, {} ms / call".format(
        prefix, (end - st) / n_iter * 1000))
    client.close()
    return (end - st) / n_iter * 1000


# Helper Function `fetch_related_events`
def fn3(e, n_iter, ans, prefix):
    client = ASERClient(port=ASER_PORT)
    st = time.time()
    for _ in range(n_iter):
        try:
            tmp = client.fetch_related_events(e)
            assert json.dumps(tmp) == ans, "[{}] Wrong answer".format(prefix)
        except Exception:
            print("[{}] Wrong answer".format(prefix))
            print(traceback.format_exc())
            client.close()
            return
    end = time.time()
    print("[{}] passed, {} ms / call".format(
        prefix, (end - st) / n_iter * 1000))
    client.close()
    return (end - st) / n_iter * 1000

def test_extract_eventualities():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing extract struct...")
    ans1 = json.dumps(client.extract_eventualities(s1))
    ans2 = json.dumps(client.extract_eventualities(s2))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    for i in range(n_workers // 2):
        r = pool.apply_async(fn1, args=(s1, n_iter, ans1, "`ext_event_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn1, args=(s2, n_iter, ans2, "`ext_event_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`ext_event_struct` {:.2f} ms / call in average".format(avg_t))

def test_predict_relation():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing match relation...")
    event1 = client.extract_eventualities(s1, only_events=True)[0]
    event2 = client.extract_eventualities(s2, only_events=True)[0]
    event3 = client.extract_eventualities(s3, only_events=True)[0]
    ans1 = json.dumps(client.predict_relation(event1, event2))
    ans2 = json.dumps(client.predict_relation(event1, event3))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    for i in range(n_workers // 2):
        r = pool.apply_async(fn2, args=(event1, event2, n_iter, ans1, "`predict_relation_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn2, args=(event1, event3, n_iter, ans2, "`predict_relation_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`match_relation` {:.2f} ms / call in average".format(avg_t))


def test_fetch_related_events():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing get related events...")
    event1 = client.extract_eventualities(s1, only_events=True)[0]
    event2 = client.extract_eventualities(s2, only_events=True)[0]
    ans1 = json.dumps(client.fetch_related_events(event1))
    ans2 = json.dumps(client.fetch_related_events(event2))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    for i in range(n_workers // 2):
        r = pool.apply_async(fn3, args=(event1, n_iter, ans1, "`fetch_related_events_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn3, args=(event2, n_iter, ans2, "`fetch_related_events_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`get_related_events` {:.2f} ms / call in average".format(avg_t))


def test_all_apis():
    client = ASERClient(port=ASER_PORT)
    event_info_1 = client.extract_eventualities(s1)
    event1 = event_info_1["eventualities"][0]
    event_info_2 = client.extract_eventualities(s2)
    event2 = event_info_2["eventualities"][0]
    event_info_3 = client.extract_eventualities(s3)
    event3 = event_info_3["eventualities"][0]
    rel1 = client.predict_relation(event1, event2)
    rel2 = client.predict_relation(event1, event3)
    related_events1 = client.fetch_related_events(event1)
    related_events2 = client.fetch_related_events(event2)
    client.close()

    st = time.time()
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    pool.apply_async(fn1, args=(s1, n_iter, json.dumps(event_info_1), "`ext_event_0`"))
    pool.apply_async(fn1, args=(s2, n_iter, json.dumps(event_info_2), "`ext_event_1`"))
    pool.apply_async(fn2, args=(event1, event2, n_iter, json.dumps(rel1), "`match_relation_2`"))
    pool.apply_async(fn2, args=(event1, event3, n_iter, json.dumps(rel2), "`match_relation_3`"))
    pool.apply_async(fn3, args=(event1, n_iter, json.dumps(related_events1), "`get_related_events_4`"))
    pool.apply_async(fn3, args=(event2, n_iter, json.dumps(related_events2), "`get_related_events_5`"))
    pool.close()
    pool.join()
    end = time.time()
    print("Overall duration: {:.2f} s".format(end - st))


if __name__ == "__main__":
    test_extract_eventualities()
    test_predict_relation()
    test_fetch_related_events()
    test_all_apis()