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




# Helper Function `extract_eventualities_struct`
def fn1(s, n_iter, ans, prefix):
    client = ASERClient(port=ASER_PORT)
    st = time.time()
    for _ in range(n_iter):
        try:
            tmp = client.extract_eventualities_struct(s)
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

# Helper Function `extract_eventualities`
def fn2(s, n_iter, ans, prefix):
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
    print("[{}] passed, {} ms / call".format(
        prefix, (end - st) / n_iter * 1000))
    client.close()
    return (end - st) / n_iter * 1000


# Helper Function `get_exact_match_event`
def fn3(event, n_iter, ans, prefix):
    client = ASERClient(port=ASER_PORT)
    st = time.time()
    for _ in range(n_iter):
        try:
            tmp = client.get_exact_match_event(event)
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


# Helper Function `get_exact_match_relation`
def fn4(e1, e2, n_iter, ans, prefix):
    client = ASERClient(port=ASER_PORT)
    st = time.time()
    for _ in range(n_iter):
        try:
            tmp = client.get_exact_match_relation(e1, e2)
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


# Helper Function `get_related_events`
def fn5(e, n_iter, ans, prefix):
    client = ASERClient(port=ASER_PORT)
    st = time.time()
    for _ in range(n_iter):
        try:
            tmp = client.get_related_events(e)
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


def test_extract_eventualities_struct():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing extract struct...")
    ans1 = json.dumps(client.extract_eventualities_struct(s1))
    ans2 = json.dumps(client.extract_eventualities_struct(s2))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    for i in range(n_workers // 2):
        r = pool.apply_async(fn1, args=(s1, n_iter, ans1, "`ext_event_struct_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn1, args=(s2, n_iter, ans2, "`ext_event_struct_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`ext_event_struct` {:.2f} ms / call in average".format(avg_t))


def test_extract_eventualities():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing extract event...")
    ans1 = json.dumps(client.extract_eventualities(s1))
    ans2 = json.dumps(client.extract_eventualities(s2))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    for i in range(n_workers // 2):
        r = pool.apply_async(fn2, args=(s1, n_iter, ans1, "`ext_event_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn2, args=(s2, n_iter, ans2, "`ext_event_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`ext_event` {:.2f} ms / call in average".format(avg_t))


def test_get_exact_match_event():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing match event...")
    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)
    ans1 = json.dumps(client.get_exact_match_event(event1))
    ans2 = json.dumps(client.get_exact_match_event(event2))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    # pool.apply_async(fn3, args=(event1, 500, ans1, "`match_event_1`({})".format(0),))
    for i in range(n_workers // 2):
        r = pool.apply_async(fn3, args=(event1, n_iter, ans1, "`match_event_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn3, args=(event2, n_iter, ans2, "`match_event_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`ext_struct` {:.2f} ms / call in average".format(avg_t))


def test_get_exact_match_relation():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing match relation...")
    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)
    event3 = client.extract_eventualities(s3)
    ans1 = json.dumps(client.get_exact_match_relation(event1, event2))
    ans2 = json.dumps(client.get_exact_match_relation(event1, event3))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    for i in range(n_workers // 2):
        r = pool.apply_async(fn4, args=(event1, event2, n_iter, ans1, "`match_relation_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn4, args=(event1, event3, n_iter, ans2, "`match_relation_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`match_relation` {:.2f} ms / call in average".format(avg_t))


def test_get_related_events():
    client = ASERClient(port=ASER_PORT)

    print("=" * 50)
    print("Testing get related events...")
    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)
    ans1 = json.dumps(client.get_related_events(event1))
    ans2 = json.dumps(client.get_related_events(event2))
    client.close()

    results_list = []
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    for i in range(n_workers // 2):
        r = pool.apply_async(fn5, args=(event1, n_iter, ans1, "`get_related_events_1`({})".format(i),))
        results_list.append(r)
        r = pool.apply_async(fn5, args=(event2, n_iter, ans2, "`get_related_events_2`({})".format(i),))
        results_list.append(r)
    pool.close()
    pool.join()
    avg_t = np.mean([r.get() for r in results_list]) / n_workers

    print("`get_related_events` {:.2f} ms / call in average".format(avg_t))


def test_all_apis():
    client = ASERClient(port=ASER_PORT)
    event_struct_1 = client.extract_eventualities_struct(s1)
    event_struct_2 = client.extract_eventualities_struct(s2)
    event1 = client.extract_eventualities(s1)
    event2 = client.extract_eventualities(s2)
    event3 = client.extract_eventualities(s3)
    e1 = client.get_exact_match_event(event1)
    e2 = client.get_exact_match_event(event2)
    e3 = client.get_exact_match_event(event3)
    rel1 = client.get_exact_match_relation(e1, e2)
    rel2 = client.get_exact_match_relation(e1, e3)
    related_events1 = client.get_related_events(e1)
    related_events2 = client.get_related_events(e2)
    client.close()

    st = time.time()
    n_workers = 10
    n_iter = 50
    pool = Pool(n_workers)
    pool.apply_async(fn1, args=(s1, n_iter, json.dumps(event_struct_1), "`ext_event_strcut_1`"))
    pool.apply_async(fn1, args=(s2, n_iter, json.dumps(event_struct_2), "`ext_event_strcut_2`"))
    pool.apply_async(fn2, args=(s1, n_iter, json.dumps(event1), "`ext_event_3`"))
    pool.apply_async(fn2, args=(s2, n_iter, json.dumps(event2), "`ext_event_4`"))
    pool.apply_async(fn3, args=(event1, n_iter, json.dumps(e1), "`match_event_5`"))
    pool.apply_async(fn3, args=(event2, n_iter, json.dumps(e2), "`match_event_6`"))
    pool.apply_async(fn4, args=(event1, event2, n_iter, json.dumps(rel1), "`match_relation_7`"))
    pool.apply_async(fn4, args=(event1, event3, n_iter, json.dumps(rel2), "`match_relation_8`"))
    pool.apply_async(fn5, args=(e1, n_iter, json.dumps(related_events1), "`get_related_events_9`"))
    pool.apply_async(fn5, args=(e2, n_iter, json.dumps(related_events2), "`get_related_events_10`"))
    pool.close()
    pool.join()
    end = time.time()
    print("Overall duration: {:.2f} s".format(end - st))


if __name__ == "__main__":
    test_extract_eventualities_struct()
    test_extract_eventualities()
    test_get_exact_match_event()
    # test_get_exact_match_relation()
    test_get_related_events()
    test_all_apis()