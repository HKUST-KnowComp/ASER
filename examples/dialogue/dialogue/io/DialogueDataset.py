from multiprocessing import Pool
import os
import torch
import torch.utils.data
from tqdm import tqdm
import json
from dialogue.toolbox.utils import padding_list, chunks


def all_len_eq(_list, x):
    for item in _list:
        if len(item) != x:
            return False
    return True


class Batch(object):
    def __init__(self, data, opt):
        self.opt = opt
        self.enc_inps = None
        self.dec_inps = None
        self.dec_start_inps = None
        self.dec_tgts = None

        self.aser_lens = None
        self.aser_id_inps = None
        self.aser_triple_inps = None

        self.omcs_lens = None
        self.omcs_id_inps = None
        self.omcs_triple_inps = None

        self.knowly_lens = None
        self.knowly_id_inps = None
        self.knowly_triple_inps = None

        self.build_batch(data)

    def build_batch(self, data):
        pad_idx = self.opt.pad_idx
        bos_idx = self.opt.bos_idx
        eos_idx = self.opt.eos_idx
        posts_lens = [len(t[0]) + 1 for t in data]
        enc_inps = [padding_list(t[0] + [eos_idx], self.opt.max_post_length + 1, pad_idx) for t in data]
        self.enc_inps = (torch.LongTensor(enc_inps),
                         torch.LongTensor(posts_lens))
        assert all_len_eq(enc_inps, self.opt.max_post_length + 1), print(data)

        response_lens = [len(t[1]) + 1 for t in data]
        dec_inps = [padding_list([bos_idx] + t[1], self.opt.max_response_length + 1, pad_idx) for t in data]
        dec_tgts = [padding_list(t[1] + [eos_idx], self.opt.max_response_length + 1, pad_idx) for t in data]
        assert all_len_eq(dec_inps, self.opt.max_response_length + 1), print(data)
        assert all_len_eq(dec_tgts, self.opt.max_response_length + 1), print(data)
        self.dec_inps = (torch.LongTensor(dec_inps),
                         torch.LongTensor(response_lens))
        self.dec_start_inps = torch.LongTensor([[[bos_idx]] for _ in range(len(data))])
        self.dec_tgts = (torch.LongTensor(dec_tgts),
                         torch.LongTensor(response_lens))

        aser_lens = [len(t[2]) for t in data]
        self.aser_lens = torch.LongTensor(aser_lens)
        aser_id_inps = [padding_list(t[2], self.opt.max_aser_triples, pad_idx) for t in data]
        self.aser_id_inps = torch.LongTensor(aser_id_inps)
        aser_triple_inps = [padding_list(
            t[3], self.opt.max_aser_triples, [pad_idx, pad_idx, pad_idx]) for t in data]
        self.aser_triple_inps = torch.LongTensor(aser_triple_inps)

        omcs_lens = [len(t[4]) for t in data]
        self.omcs_lens = torch.LongTensor(omcs_lens)
        omcs_id_inps = [padding_list(t[4], self.opt.max_omcs_triples, pad_idx) for t in data]
        self.omcs_id_inps = torch.LongTensor(omcs_id_inps)
        omcs_triple_inps = [padding_list(
            t[5], self.opt.max_omcs_triples, [pad_idx, pad_idx, pad_idx]) for t in data]
        self.omcs_triple_inps = torch.LongTensor(omcs_triple_inps)

        knowly_lens = [len(t[6]) for t in data]
        self.knowly_lens = torch.LongTensor(knowly_lens)
        knowly_id_inps = [padding_list(t[6], self.opt.max_knowly_triples, pad_idx) for t in data]
        self.knowly_id_inps = torch.LongTensor(knowly_id_inps)
        knowly_triple_inps = [padding_list(
            t[7], self.opt.max_knowly_triples, [pad_idx, pad_idx, pad_idx]) for t in data]
        self.knowly_triple_inps = torch.LongTensor(knowly_triple_inps)

    def cuda(self):
        self.enc_inps = [t.cuda(async=True) for t in self.enc_inps]
        self.dec_inps = [t.cuda(async=True) for t in self.dec_inps]
        self.dec_start_inps = self.dec_start_inps.cuda()
        self.dec_tgts = [t.cuda(async=True) for t in self.dec_tgts]
        self.aser_lens = self.aser_lens.cuda(async=True)
        self.aser_id_inps = self.aser_id_inps.cuda(async=True)
        self.aser_triple_inps = self.aser_triple_inps.cuda(async=True)
        self.omcs_lens = self.omcs_lens.cuda(async=True)
        self.omcs_id_inps = self.omcs_id_inps.cuda(async=True)
        self.omcs_triple_inps = self.omcs_triple_inps.cuda(async=True)
        self.knowly_lens = self.knowly_lens.cuda(async=True)
        self.knowly_id_inps = self.knowly_id_inps.cuda(async=True)
        self.knowly_triple_inps = self.knowly_triple_inps.cuda(async=True)


class DialogueDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, vocabs, opt, data_cache_path=None):
        self.opt = opt
        with open(data_path) as f:
            raw_lines = f.readlines()

        if data_cache_path and os.path.exists(data_cache_path):
            self.data = torch.load(data_cache_path)
        else:
            if len(raw_lines) > 1000:
                res_list = []
                pool = Pool(20)
                for lines in chunks(raw_lines, len(raw_lines) // 20):
                    res = pool.apply_async(self.load_data, args=(lines, vocabs,))
                    res_list.append(res)
                pool.close()
                pool.join()

                posts = []
                responses = []
                aser_ids_list = []
                aser_triples_list = []
                omcs_ids_list = []
                omcs_triples_list = []
                knowly_ids_list = []
                knowly_triples_list = []
                for res in res_list:
                    records = res.get()
                    posts.extend(records[0])
                    responses.extend(records[1])
                    aser_ids_list.extend(records[2])
                    aser_triples_list.extend(records[3])
                    omcs_ids_list.extend(records[4])
                    omcs_triples_list.extend(records[5])
                    knowly_ids_list.extend(records[6])
                    knowly_triples_list.extend(records[7])
                self.data = [(posts[i], responses[i],
                              aser_ids_list[i], aser_triples_list[i],
                              omcs_ids_list[i], omcs_triples_list[i],
                              knowly_ids_list[i], knowly_triples_list[i]
                              )
                             for i in range(len(posts))]
            else:
                t = self.load_data(raw_lines, vocabs)
                self.data = [(t[0][i], t[1][i], t[2][i], t[3][i], t[4][i],
                              t[5][i], t[6][i], t[7][i])
                             for i in range(len(t[0]))]
            if data_cache_path:
                torch.save(self.data, data_cache_path)

    def load_data(self, lines, vocabs):
        posts = []
        responses = []
        aser_ids_list = []
        aser_triples_list = []
        omcs_ids_list = []
        omcs_triples_list = []
        knowly_ids_list = []
        knowly_triples_list = []
        for line in lines:
            record = json.loads(line)
            post_idx = [vocabs["word"].to_idx(t) for t in record["post"].lower().split()[:self.opt.max_post_length]]
            posts.append(post_idx)
            response_idx = [vocabs["word"].to_idx(t) for t in record["response"].lower().split()[:self.opt.max_response_length]]
            responses.append(response_idx)

            aser_ids_list.append(
                [vocabs["aser"].to_idx(t) for t in record["aser_triples"][:self.opt.max_aser_triples]])
            tmp = []
            for event_pair in record["aser_triples"][:self.opt.max_aser_triples]:
                e1, r, e2 = event_pair.split("$")
                tmp.append([vocabs["aser_event"].to_idx(e1),
                            vocabs["aser_relation"].to_idx(r),
                            vocabs["aser_event"].to_idx(e2)])
            aser_triples_list.append(tmp)

            omcs_ids_list.append(
                [vocabs["omcs"].to_idx(t) for t in record["omcs_triples"][:self.opt.max_omcs_triples]])
            tmp = []
            for event_pair in record["omcs_triples"][:self.opt.max_omcs_triples]:
                e1, r, e2 = event_pair.split("$")
                tmp.append([vocabs["omcs_event"].to_idx(e1),
                            vocabs["omcs_relation"].to_idx(r),
                            vocabs["omcs_event"].to_idx(e2)])
            omcs_triples_list.append(tmp)

            knowly_ids_list.append(
                [vocabs["knowlywood"].to_idx(t) for t in record["knowlywood_triples"][:self.opt.max_knowly_triples]])
            tmp = []
            for event_pair in record["knowlywood_triples"][:self.opt.max_knowly_triples]:
                e1, r, e2 = event_pair.split("$")
                tmp.append([vocabs["knowlywood_event"].to_idx(e1),
                            vocabs["knowlywood_relation"].to_idx(r),
                            vocabs["knowlywood_event"].to_idx(e2)])
            knowly_triples_list.append(tmp)

        return posts, responses, aser_ids_list, aser_triples_list,\
               omcs_ids_list, omcs_triples_list, knowly_ids_list, knowly_triples_list

    def __getitem__(self, index):
        """

        :param index: int
        :return:
        """
        return self.data[index], self.opt

    def __len__(self):
        return len(self.data)


def collate_fn(data):
    batch = Batch([t[0] for t in data], data[0][-1])
    return batch


class DialogueDatasetIterator(object):
    def __init__(self, file_name, vocabs, file_cache_path=None, epochs=None, batch_size=16,
                 is_train=True, n_workers=0,
                 use_cuda=True, opt=None):
        self.dataset = DialogueDataset(file_name, vocabs, opt, file_cache_path)
        self.epochs = epochs
        self.batch_size = batch_size
        self.n_workers = n_workers
        self.is_train = is_train
        self.use_cuda = use_cuda

    def __iter__(self):
        if self.is_train:
            i = 0
            while True:
                data_loader = torch.utils.data.DataLoader(
                    dataset=self.dataset,
                    batch_size=self.batch_size,
                    shuffle=True,
                    collate_fn=collate_fn,
                    num_workers=self.n_workers)
                for batch in data_loader:
                    if self.use_cuda:
                        batch.cuda()
                    yield batch
                i += 1
                if self.epochs and i == self.epochs:
                    break
        else:
            data_loader = torch.utils.data.DataLoader(
                dataset=self.dataset,
                batch_size=self.batch_size,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=self.n_workers)
            for batch in data_loader:
                if self.use_cuda:
                    batch.cuda()
                yield batch
