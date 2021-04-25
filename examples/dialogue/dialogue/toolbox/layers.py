# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


def sequence_mask(lengths, max_len=None):
    """
    Creates a boolean mask from sequence lengths.
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    return (torch.arange(0, max_len)
            .type_as(lengths)
            .repeat(batch_size, 1)
            .lt(lengths.unsqueeze(1)))


class Attention(nn.Module):
    def __init__(self, input_size, activation=nn.Tanh(), method="dot"):
        super(Attention, self).__init__()
        self.attn_score = AttnScore(input_size=input_size,
                                    activation=activation,
                                    method=method)

    def forward(self, query, keys, q_lens=None, k_lens=None):
        """
        :param query: bsize x query_num x input_size
        :param keys:  bsize x key_num x input_size
        :param q_lens: bsize x query_num
        :param k_lens: bsize x key_num
        :return: bsize x 1 x input_size
        """
        attn_weights = self.attn_score(keys, query, k_lens, q_lens) # bsize x 1 x key_num
        contexts = attn_weights.bmm(keys) # bsize x 1 x input_size
        return contexts, attn_weights


class AttnScore(nn.Module):
    def __init__(self, input_size, activation=nn.Tanh(),
                 method="dot"):
        super(AttnScore, self).__init__()
        self.activation = activation
        self.input_size = input_size
        self.method = method
        if method == "general":
            self.linear = nn.Linear(input_size, input_size)
            init.uniform(self.linear.weight.data, -0.005, 0.005)
        elif method == "concat":
            self.linear_1 = nn.Linear(input_size*2, input_size)
            self.linear_2 = nn.Linear(input_size, 1)
            init.uniform(self.linear_1.weight.data, -0.005, 0.005)
            init.uniform(self.linear_2.weight.data, -0.005, 0.005)
        elif method == "tri_concat":
            self.linear = nn.Linear(input_size*3, 1)
            init.uniform(self.linear.weight.data, -0.005, 0.005)

    def forward(self, h1, h2, h1_lens=None, h2_lens=None, normalize=True):
        """
        :param h1: b x m x d
        :param h2: b x n x d
        :return: attn_weights: b x 1 x m
        """
        bsize, seq_l1, dim = h1.size()
        bsize, seq_l2, dim = h2.size()
        assert h1.size(-1) == self.input_size
        assert h2.size(-1) == self.input_size
        if self.method == "dot":
            align = h2.bmm(h1.transpose(1, 2))
        elif self.method == "general":
            align = h2.bmm(self.linear(h1).transpose(1, 2))
        elif self.method == "concat":
            h1 = h1.unsqueeze(1).repeat(1, seq_l2, 1, 1)
            h2 = h2.unsqueeze(2).repeat(1, 1, seq_l1, 1)
            align = self.linear_2(self.activation(
                self.linear_1(torch.cat([h1, h2], dim=3)))).squeeze(-1)
            align = F.softmax(align, dim=2)
        elif self.method == "tri_concat":
            h1 = h1.unsqueeze(1).repeat(1, seq_l2, 1, 1)
            h2 = h2.unsqueeze(2).repeat(1, 1, seq_l1, 1)
            align = self.linear(torch.cat([h1, h2, h1 * h2], dim=3)).squeeze(-1)
        if h1_lens is not None:
            mask = sequence_mask(h1_lens, max_len=seq_l1).unsqueeze(1)
            align.data.masked_fill_(1 - mask, -1e8)
        if normalize:
            attn_weights = F.softmax(align, dim=2)
        else:
            attn_weights = F.softmax(align, dim=2)
        return attn_weights


class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from
    "Attention is All You Need"
    :cite:`DBLP:journals/corr/VaswaniSPUJGKP17`.
    Similar to standard `dot` attention but uses
    multiple attention distributions simulataneously
    to select relevant items.
    .. mermaid::
       graph BT
          A[key]
          B[value]
          C[query]
          O[output]
          subgraph Attn
            D[Attn 1]
            E[Attn 2]
            F[Attn N]
          end
          A --> D
          C --> D
          A --> E
          C --> E
          A --> F
          C --> F
          D --> O
          E --> O
          F --> O
          B --> O
    Also includes several additional tricks.
    Args:
       head_count (int): number of parallel heads
       model_dim (int): the dimension of keys/values/queries,
           must be divisible by head_count
       dropout (float): dropout parameter
    """

    def __init__(self, head_count, model_dim, dropout=0.1):
        assert model_dim % head_count == 0
        self.dim_per_head = model_dim // head_count
        self.model_dim = model_dim

        super(MultiHeadedAttention, self).__init__()
        self.head_count = head_count

        self.linear_keys = nn.Linear(model_dim,
                                     head_count * self.dim_per_head)
        self.linear_values = nn.Linear(model_dim,
                                       head_count * self.dim_per_head)
        self.linear_query = nn.Linear(model_dim,
                                      head_count * self.dim_per_head)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.final_linear = nn.Linear(model_dim, model_dim)

    def forward(self, key, value, query, mask=None):
        """
        Compute the context vector and the attention vectors.
        Args:
           key (`FloatTensor`): set of `key_len`
                key vectors `[batch, key_len, dim]`
           value (`FloatTensor`): set of `key_len`
                value vectors `[batch, key_len, dim]`
           query (`FloatTensor`): set of `query_len`
                 query vectors  `[batch, query_len, dim]`
           mask: binary mask indicating which keys have
                 non-zero attention `[batch, query_len, key_len]`
        Returns:
           (`FloatTensor`, `FloatTensor`) :
           * output context vectors `[batch, query_len, dim]`
           * one of the attention vectors `[batch, query_len, key_len]`
        """

        batch_size = key.size(0)
        dim_per_head = self.dim_per_head
        head_count = self.head_count
        key_len = key.size(1)
        query_len = query.size(1)

        def shape(x):
            """  projection """
            return x.view(batch_size, -1, head_count, dim_per_head) \
                .transpose(1, 2)

        def unshape(x):
            """  compute context """
            return x.transpose(1, 2).contiguous() \
                    .view(batch_size, -1, head_count * dim_per_head)

        # 1) Project key, value, and query.
        key_up = shape(self.linear_keys(key))
        value_up = shape(self.linear_values(value))
        query_up = shape(self.linear_query(query))

        # 2) Calculate and scale scores.
        query_up = query_up / math.sqrt(dim_per_head)
        scores = torch.matmul(query_up, key_up.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask, -1e18)

        # 3) Apply attention dropout and compute context vectors.
        attn = self.softmax(scores)
        drop_attn = self.dropout(attn)
        context = unshape(torch.matmul(drop_attn, value_up))

        output = self.final_linear(context)
        # CHECK
        # batch_, q_len_, d_ = output.size()
        # aeq(q_len, q_len_)
        # aeq(batch, batch_)
        # aeq(d, d_)

        # Return one attn
        top_attn = attn \
            .view(batch_size, head_count,
                  query_len, key_len)[:, 0, :, :] \
            .contiguous()

        return output, top_attn


class Highway(nn.Module):
    def __init__(self, size, num_layers, f):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.linear = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.gate = nn.ModuleList([nn.Linear(size, size) for _ in range(num_layers)])

        self.f = f

    def forward(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x))
            transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine
            transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
         """

        for layer in range(self.num_layers):
            gate = F.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x


class LayerNorm(nn.Module):
    """
        Layer Normalization class
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class PositionwiseFeedForward(nn.Module):
    """ A two-layer Feed-Forward-Network with residual layer norm.
        Args:
            d_model (int): the size of input for the first-layer of the FFN.
            d_ff (int): the hidden layer size of the second-layer
                              of the FNN.
            dropout (float): dropout probability(0-1.0).
    """

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.layer_norm = LayerNorm(d_model)
        self.dropout_1 = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.dropout_2 = nn.Dropout(dropout)

    def forward(self, x):
        """
        Layer definition.
        Args:
            input: [ batch_size, input_len, model_dim ]
        Returns:
            output: [ batch_size, input_len, model_dim ]
        """
        inter = self.dropout_1(self.relu(self.w_1(self.layer_norm(x))))
        output = self.dropout_2(self.w_2(inter))
        return output + x


class SelfAttention(nn.Module):
    def __init__(self, input_size, activation=nn.Tanh(), method="dot"):
        super(SelfAttention, self).__init__()
        self.attn = Attention(input_size=input_size,
                              activation=activation,
                              method=method)

    def forward(self, h, h_lens):
        return self.attn(h, h, h_lens, h_lens)


class VariableSelfAttention(nn.Module):
    def __init__(self, input_size, activation=nn.Tanh(), method="dot"):
        super(VariableSelfAttention, self).__init__()
        query = torch.zeros(1, 1, input_size)
        stdv = 1. / math.sqrt(input_size)
        init.uniform_(query.data, -stdv, stdv)
        self.query = nn.Parameter(query, requires_grad=True)
        self.attn = Attention(input_size=input_size,
                              activation=activation,
                              method=method)

    def forward(self, h, h_lens=None):
        """
        :param h: bsize x seq_len x input_size
        :param h_lens: bsize x seq_len
        :return: bsize x 1 x input_size, bsize x 1 x seq_len
        """
        bsize, _, input_size = h.size()
        query = self.query.repeat(bsize, 1, 1)
        return self.attn(query, h, k_lens=h_lens)

    def forward_list(self, h_list, h_lens_list=None):
        """
        :param h_list: [h1, h2, ..., hn]
        :param h_lens_list: [h1_lens, h2_lens, ..., hn_lens]
        :return: ret: n x bsize x 1 x input_size
                attns: n x bsize x 1 x seq_len
        """
        h_num = len(h_list)
        bsize, seq_len, input_size = h_list[0].size()
        h_concat = torch.cat(h_list, dim=0)
        if h_lens_list:
            h_lens_concat = torch.cat(h_lens_list, dim=0)
        else:
            h_lens_concat = None
        ret, attns = self.forward(h_concat, h_lens_concat)
        ret = ret.view(h_num, bsize, 1, input_size)
        attns = attns.view(h_num, bsize, 1, seq_len)
        return ret, attns


class StucturedSelfAttention(nn.Module):
    """ Not tested """
    def __init__(self, input_size, hidden_size=64):
        super(StucturedSelfAttention, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            # nn.BatchNorm1d(self_linear_hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, inputs, input_lens=None):
        bsize, seq_len, input_size = inputs.size()
        if input_lens is not None:
            mask = sequence_mask(input_lens, max_len=seq_len).unsqueeze(2).repeat(
                1, 1, input_size).float()
            inputs = inputs * mask
        flatten_inputs = inputs.view(-1, input_size)
        out = self.fc(flatten_inputs)
        out = out.view(bsize, seq_len, -1)
        self_attn_weights = F.softmax(out, dim=1).transpose(1, 2)
        encoded = self_attn_weights.bmm(inputs)
        return encoded, self_attn_weights

    def encode_list(self, input_list, input_lens_list=None):
        input_num = len(input_list)
        if input_lens_list:
            input_mask_list = [sequence_mask(input_len) for input_len in input_lens_list]
            input_list = [(input_list[i] * input_mask_list[i]).unsqueeze(1) for i in range(len(input_list))]
        concat_inputs = torch.cat(input_list, dim=1)  # bsize x input_num x seq_len x input_size
        bsize, seq_len, input_size = concat_inputs.size()
        flatten_inputs = concat_inputs.view(-1, input_size)
        out = self.fc(flatten_inputs)
        out = out.view(bsize * input_num, seq_len, -1)
        self_attn_weights = F.softmax(out, dim=1).transpose(1, 2) # bsize * input_num x 1 x seq_len
        flatten_inputs = concat_inputs.view(bsize * input_num, seq_len, -1) # bsize * input_num x seq_len  x 1
        encoded = self_attn_weights.bmm(flatten_inputs).squeeze(-1) # bsize * input_num x 1
        encoded = encoded.view(bsize, input_num, 1)
        return encoded


class SortedLSTM(nn.Module):
    def __init__(self, **kw):
        super(SortedLSTM, self).__init__()
        self.lstm = nn.LSTM(**kw)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.lstm.weight_ih_l0)
        nn.init.constant_(self.lstm.bias_ih_l0, val=0)
        nn.init.orthogonal_(self.lstm.weight_hh_l0)
        nn.init.constant_(self.lstm.bias_hh_l0, val=0)

    def forward(self, inputs, input_lens=None, last_hidden=None):
        if input_lens is None:
            return self.lstm(inputs, last_hidden)
        inp_len = inputs.size(1)
        sorted_lens, sorted_idx = input_lens.sort(descending=True)
        _, original_idx = sorted_idx.sort(descending=False)
        sorted_inputs = inputs[sorted_idx]
        sorted_packed_embeds = pack_padded_sequence(
            sorted_inputs, sorted_lens.cpu().numpy(), batch_first=True)
        packed_encoder_output, (sorted_hidden, sorted_cell) = self.lstm(
            sorted_packed_embeds, last_hidden)
        sorted_encoder_output, _ = pad_packed_sequence(packed_encoder_output)
        sorted_encoder_output = sorted_encoder_output.transpose(0, 1)
        encoder_outputs = sorted_encoder_output[original_idx, :, :]
        bsize, outp_len, outp_dim = encoder_outputs.size()
        if outp_len < inp_len:
            app_shape = bsize, inp_len - outp_len, outp_dim
            app_tensor = encoder_outputs.data.new(*app_shape).zero_().float()
            app_tensor.data.add_(1e-7)
            encoder_outputs = torch.cat((encoder_outputs, app_tensor), dim=1)
        hidden = sorted_hidden[:, original_idx, :]
        cell = sorted_cell[:, original_idx, :]

        return encoder_outputs, (hidden, cell)

    def flatten_parameters(self):
        self.lstm.flatten_parameters()


class SortedGRU(nn.Module):
    def __init__(self, **kw):
        super(SortedGRU, self).__init__()
        self.gru = nn.GRU(**kw)

    def forward(self, inputs, input_lens, last_hidden=None):
        if input_lens is None:
            return self.gru(inputs, last_hidden)
        inp_len = inputs.size(1)
        sorted_lens, sorted_idx = input_lens.sort(descending=True)
        _, original_idx = sorted_idx.sort(descending=False)
        sorted_inputs = inputs[sorted_idx]
        sorted_packed_embeds = pack_padded_sequence(
            sorted_inputs, sorted_lens.cpu().numpy(), batch_first=True)
        packed_encoder_output, sorted_hidden = self.gru(
            sorted_packed_embeds, last_hidden)
        sorted_encoder_output, _ = pad_packed_sequence(packed_encoder_output)
        sorted_encoder_output = sorted_encoder_output.transpose(0, 1)
        encoder_outputs = sorted_encoder_output[original_idx, :, :]
        bsize, outp_len, outp_dim = encoder_outputs.size()
        if outp_len < inp_len:
            app_shape = bsize, inp_len - outp_len, outp_dim
            app_tensor = encoder_outputs.data.new(*app_shape).zero_().float()
            app_tensor.data.add_(1e-7)
            encoder_outputs = torch.cat((encoder_outputs, app_tensor), dim=1)
        hidden = sorted_hidden[:, original_idx, :]

        return encoder_outputs, hidden

    def flatten_parameters(self):
        self.gru.flatten_parameters()


