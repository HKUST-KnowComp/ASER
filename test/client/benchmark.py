import time
from multiprocessing import Pool
from aser.client import ASERClient


def fn1(s, iter_num):
    client = ASERClient(port=8000)
    for i in range(iter_num):
        client.extract_eventualities(s)
    client.close()

def fn2(e1, e2, iter_num):
    client = ASERClient(port=8000)
    for i in range(iter_num):
        client.predict_relation(e1, e2)
    client.close()

def fn3(e, iter_num):
    client = ASERClient(port=8000)
    for i in range(iter_num):
        client.fetch_related_eventualities(e)
    client.close()


if __name__ == "__main__":
    client = ASERClient(port=8000)

    # s1 = 'they yucca look leggy'
    # s2 = 'they african violet refuse to flower'
    s1 = 'I am hungry'
    s2 = 'I am in the kitchen'
    e1 = client.extract_eventualities(s1, only_eventualities=True)[0]
    e2 = client.extract_eventualities(s2, only_eventualities=True)[0]
    client.close()

    total_num = 10000
    n_workers = 1
    batch_size = total_num // n_workers

    st = time.time()
    pool = Pool(n_workers)
    for i in range(n_workers):
        pool.apply_async(fn1, args=(s2, batch_size,))
    pool.close()
    pool.join()
    print("`extract_event`: {:.2f} sent / s".format(total_num / (time.time() - st)))

    st = time.time()
    pool = Pool(n_workers)
    for i in range(n_workers):
        pool.apply_async(fn2, args=(e1, e2, batch_size,))
    pool.close()
    pool.join()
    print("`predict_relation`: {:.2f} pair / s".format(total_num / (time.time() - st)))

    st = time.time()
    pool = Pool(n_workers)
    for i in range(n_workers):
        pool.apply_async(fn3, args=(e1, batch_size,))
    pool.close()
    pool.join()
    print("`fetch_related_events`: {:.2f} event / s".format(total_num / (time.time() - st)))




