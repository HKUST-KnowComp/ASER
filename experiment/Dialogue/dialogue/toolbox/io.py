import torch
import torchtext


class OrderedIterator(torchtext.data.Iterator):
    """ Ordered Iterator Class
        https://github.com/OpenNMT/OpenNMT-py/blob/master/onmt/
        inputters/inputter.py
    """

    def create_batches(self):
        """ Create batches """
        if self.train:
            def _pool(data, random_shuffler):
                for p in torchtext.data.batch(data, self.batch_size * 100):
                    p_batch = torchtext.data.batch(
                        sorted(p, key=self.sort_key),
                        self.batch_size, self.batch_size_fn)
                    for b in random_shuffler(list(p_batch)):
                        yield b

            self.batches = _pool(self.data(), self.random_shuffler)
        else:
            self.batches = []
            for b in torchtext.data.batch(self.data(), self.batch_size,
                                          self.batch_size_fn):
                self.batches.append(sorted(b, key=self.sort_key))


def make_sequential_example(ex, seq_list, seq_len, prefix="feats", pad_word="<pad>"):
    paded_list = [[] for _ in range(seq_len - len(seq_list))]
    for i, seq in enumerate(seq_list + paded_list):
        setattr(ex, prefix + "_" + str(i), seq)
    setattr(ex, prefix + "_num", len(seq_list))


def make_sequential_field(fields, seq_len, prefix="feats", pad_word="<pad>", fix_length=None):
    for i in range(seq_len):
        name = prefix + "_" + str(i)
        fields[name] = torchtext.data.Field(
                    pad_token=pad_word,
                    fix_length=fix_length,
                    include_lengths=True,
                    use_vocab=True,
                    batch_first=True)
    fields[prefix + "_num"] = torchtext.data.Field(
        include_lengths=False,
        use_vocab=False,
        batch_first=True,
        sequential=False)


def get_tensor_of_sequential_field(batch, prefix="feats"):
    ret_list = []
    ret_len_list = []
    for name, val in batch.__dict__.items():
        if name.startswith(prefix) and name.rsplit("_", 1)[-1].isdigit():
            ret_list.append(val[0])
            ret_len_list.append(val[1])
    return ret_list, ret_len_list