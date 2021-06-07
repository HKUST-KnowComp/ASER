from multiprocessing import Pool
import torch
from torch.autograd import Variable
import json


def batch_unpadding(_input, lens, right=True):
    """
    :param: _input: m x n tensor
    :param: lens:   list with size of m
    :param: right:  eliminate padding from right if true
    """
    if right:
        return torch.cat([x[:lens[i]] for i, x in enumerate(_input) if lens[i] > 0 ])
    else:
        return torch.cat([x[-lens[i]:] for i, x in enumerate(_input) if lens[i] > 0 ])


def dot2d(a, b, normalize=False):
    mm = torch.bmm(a, b.transpose(2, 1))
    if normalize:
        q1_norm = torch.norm(a, 2, dim=2, keepdim=True)
        q2_norm = torch.norm(b, 2, dim=2, keepdim=True).transpose(1, 2)
        norm = torch.bmm(q1_norm, q2_norm)
        mm = mm / (norm + 1e-8)
    return mm


def get_one_hot(lt, class_num):
    """
    :param lt: <torch.LongTensor>, bsize x seq_len
    :param voc_size: vocab_size
    :return: onehot: bsize x seq_len x vocab_size
    """
    bsize, seq_len = lt.size()
    one_hot_shape = bsize, seq_len, class_num
    onehot = Variable(lt.data.new(*one_hot_shape).zero_().float())
    onehot = onehot.scatter_(
        2, lt.unsqueeze(2), 1)
    return onehot


def get_num_parameters(model, trainable=False):
    if trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in model.parameters())


def kmax_pooling(x, dim, k, avg=False):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    res = x.gather(dim, index)
    if avg:
        return res.mean(dim=dim)
    else:
        return res


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]


def get_time_str(t):
    import time
    ISFORMAT = "%Y-%m-%d %H:%M:%S"
    return time.strftime(ISFORMAT, time.localtime(t))


def load_json_lines(raw_records):
    return [json.loads(record) for record in raw_records]


def load_json_lines_from_file_multicore(fname, n_workers=0):
    with open(fname) as f:
        raw_records = f.readlines()
    if not n_workers:
        return [json.loads(record) for record in raw_records]
    raw_record_chunks = list(chunks(raw_records, len(raw_records) // (n_workers - 1)))
    res_list = []
    pool = Pool(n_workers)
    for chunk in raw_record_chunks:
        res = pool.apply_async(load_json_lines, args=(chunk, ))
        res_list.append(res)
    pool.close()
    pool.join()
    results = [t for item in res_list for t in item.get()]
    return results


def padding_list(x, max_item_num, padding_val):
    if len(x) < max_item_num:
        return x + [padding_val for _ in range(max_item_num - len(x))]
    else:
        return x[:max_item_num]