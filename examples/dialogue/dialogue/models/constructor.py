import torch.nn as nn
from dialogue.models.aser2seq import ASEREncoderDecoder
from dialogue.models.omcs2seq import OMCSEncoderDecoder
from dialogue.models.knowly2seq import KnowlyEncoderDecoder
from dialogue.models.seq2seq import EncoderDecoder, AttnEncoderDecoder


def construct_model(opt, pre_word_emb=None):
    loss_fn = nn.NLLLoss(ignore_index=opt.meta.pad_idx, reduction="sum")
    if opt.meta.model == "seq2seq":
        model = EncoderDecoder(loss_fn=loss_fn,
                               opt=opt)
    elif opt.meta.model == "seq2seq_attn":
        model = AttnEncoderDecoder(loss_fn=loss_fn,
                                   opt=opt)
    elif opt.meta.model == "aser2seq":
        model = ASEREncoderDecoder(loss_fn=loss_fn,
                                   opt=opt)
    elif opt.meta.model == "omcs2seq":
        model = OMCSEncoderDecoder(loss_fn=loss_fn,
                                   opt=opt)
    elif opt.meta.model == "knowly2seq":
        model = KnowlyEncoderDecoder(loss_fn=loss_fn,
                                   opt=opt)
    else:
        raise NotImplementedError
    if pre_word_emb is not None and opt.meta.use_pre_word_emb:
        print("load pretrained embeddings...")
        model.load_pretrained_embedding(pre_word_emb)
    return model