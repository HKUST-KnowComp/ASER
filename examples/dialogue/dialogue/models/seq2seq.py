import torch
import torch.nn as nn
import torch.nn.functional as F
from dialogue.models.base import BaseDeepModel
from dialogue.toolbox.beam import BeamSeqs
from dialogue.toolbox.layers import SortedGRU, Attention
from dialogue.toolbox.utils import batch_unpadding


class EncoderDecoder(BaseDeepModel):
    def __init__(self, loss_fn, opt):
        super(EncoderDecoder, self).__init__()
        rnn_hidden_size = opt.model.rnn_hidden_size
        self.encoder_embedding = nn.Embedding(opt.model.word_vocab_size, opt.model.word_embed_size)
        self.decoder_embedding = nn.Embedding(opt.model.word_vocab_size, opt.model.word_embed_size)
        self.encoder = SortedGRU(input_size=opt.model.word_embed_size,
                                 hidden_size=opt.model.rnn_hidden_size // 2,
                                 num_layers=opt.model.n_layers,
                                 batch_first=True,
                                 bidirectional=True)
        self.decoder = SortedGRU(input_size=opt.model.word_embed_size,
                                 hidden_size=opt.model.rnn_hidden_size,
                                 num_layers=opt.model.n_layers,
                                 batch_first=True,
                                 bidirectional=False)
        self.dropout = nn.Dropout(opt.model.dropout)
        self.fc = nn.Linear(rnn_hidden_size, opt.model.word_vocab_size)

        self.loss_fn = loss_fn
        self.rnn_hidden_size = opt.model.rnn_hidden_size
        self.n_layers = opt.model.n_layers
        self.use_cuda = opt.meta.use_cuda

    def encode(self, encoder_inputs, encoder_lens):
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(
            encoder_embeds, encoder_lens)
        return encoder_outputs, last_hidden

    def decode(self, last_hidden, decoder_inputs, decoder_lens=None):
        decoder_embeds = self.decoder_embedding(decoder_inputs)
        decoder_outputs, last_hidden = self.decoder(
            decoder_embeds, decoder_lens, last_hidden)
        decoder_outputs = self.dropout(decoder_outputs)
        decoder_outputs = self.fc(decoder_outputs)
        return decoder_outputs, last_hidden

    def forward(self, encoder_inputs, encoder_lens, decoder_inputs, decoder_lens):
        _, encoder_last_hidden = self.encode(
            encoder_inputs, encoder_lens)
        encoder_last_hidden = self._fix_hidden(encoder_last_hidden)
        decoder_outputs, _ = self.decode(encoder_last_hidden, decoder_inputs, decoder_lens)
        outputs = F.log_softmax(decoder_outputs, dim=2)
        return outputs

    def generate(self, encoder_inputs, encoder_lens,
                 decoder_start_input, max_len, beam_size=1, eos_val=None):
        _, encoder_last_hidden = self.encode(encoder_inputs, encoder_lens)
        encoder_last_hidden = self._fix_hidden(encoder_last_hidden)
        beamseqs = BeamSeqs(beam_size=beam_size)
        beamseqs.init_seqs(seqs=decoder_start_input[0], init_state=encoder_last_hidden)
        done = False
        for i in range(max_len):
            for j, (seqs, _, last_token, last_hidden) in enumerate(beamseqs.current_seqs):
                if beamseqs.check_and_add_to_terminal_seqs(j, eos_val):
                    if len(beamseqs.terminal_seqs) >= beam_size:
                        done = True
                        break
                    continue
                out, last_hidden = self.decode(last_hidden, last_token.unsqueeze(0))
                _output = F.log_softmax(out.squeeze(0), dim=1).squeeze(0)
                scores, tokens = _output.topk(beam_size * 2)
                for k in range(beam_size * 2):
                    score, token = scores.data[k], tokens[k]
                    token = token.unsqueeze(0)
                    beamseqs.add_token_to_seq(j, token, score, last_hidden)
            if done:
                break
            beamseqs.update_current_seqs()
        final_seqs = beamseqs.return_final_seqs()
        return final_seqs[0].unsqueeze(0)

    @staticmethod
    def _fix_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                            hidden[1:hidden.size(0):2]], 2)
        return hidden

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def run_batch(self, batch):
        enc_inps, enc_lens = batch.enc_inps
        dec_inps, dec_lens = batch.dec_inps
        dec_tgts, _ = batch.dec_tgts
        dec_probs = self.forward(
            encoder_inputs=enc_inps, encoder_lens=enc_lens,
            decoder_inputs=dec_inps, decoder_lens=dec_lens)

        decoder_probs_pack = dec_probs.view(-1, dec_probs.size(2))
        decoder_targets_pack = dec_tgts.view(-1)
        loss = self.loss_fn(decoder_probs_pack, decoder_targets_pack)
        decoder_probs_pack = batch_unpadding(dec_probs, dec_lens)
        decoder_targets_pack = batch_unpadding(dec_tgts, dec_lens)
        _, pred = decoder_probs_pack.max(1)
        num_correct = pred.eq(decoder_targets_pack).sum().item()
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words,
        }
        return result_dict

    def predict_batch(self, batch, max_len=20, beam_size=4, eos_val=0):
        enc_inps, enc_lens = batch.enc_inps
        dec_start_inps = batch.dec_start_inps
        preds = self.generate(enc_inps, enc_lens, dec_start_inps,
                             max_len, beam_size, eos_val).squeeze(2)
        preds = preds.data.cpu().numpy()
        return preds


class AttnEncoderDecoder(BaseDeepModel):
    def __init__(self, loss_fn, opt):
        super(AttnEncoderDecoder, self).__init__()
        rnn_hidden_size = opt.model.rnn_hidden_size

        self.encoder_embedding = nn.Embedding(opt.model.word_vocab_size, opt.model.word_embed_size)
        self.decoder_embedding = self.encoder_embedding
        self.encoder = SortedGRU(input_size=opt.model.word_embed_size,
                                 hidden_size=opt.model.rnn_hidden_size // 2,
                                 num_layers=opt.model.n_layers,
                                 batch_first=True,
                                 bidirectional=True)
        self.decoder = SortedGRU(input_size=opt.model.word_embed_size,
                                 hidden_size=opt.model.rnn_hidden_size,
                                 num_layers=opt.model.n_layers,
                                 batch_first=True,
                                 bidirectional=False)
        self.attn = Attention(input_size=opt.model.rnn_hidden_size, method=opt.model.attn_score_method)
        self.dropout = nn.Dropout(opt.model.dropout)
        self.concat = nn.Linear(rnn_hidden_size * 2, rnn_hidden_size)
        self.fc = nn.Linear(rnn_hidden_size, opt.model.word_vocab_size)

        self.loss_fn = loss_fn
        self.rnn_hidden_size = opt.model.rnn_hidden_size
        self.n_layers = opt.model.n_layers
        self.use_cuda = opt.meta.use_cuda

    def encode(self, encoder_inputs, encoder_lens):
        encoder_embeds = self.encoder_embedding(encoder_inputs)
        encoder_outputs, last_hidden = self.encoder(
            encoder_embeds, encoder_lens)
        return encoder_outputs, last_hidden

    def decode(self, encoder_outputs, encoder_lens,
               last_hidden, decoder_inputs, decoder_lens=None):
        decoder_embeds = self.decoder_embedding(decoder_inputs)
        decoder_outputs, last_hidden = self.decoder(
            decoder_embeds, decoder_lens, last_hidden)
        contexts, _ = self.attn(decoder_outputs, encoder_outputs,
                                q_lens=decoder_lens, k_lens=encoder_lens)

        outlayer_inputs = torch.cat([decoder_outputs, contexts], dim=2)
        outlayer_outputs = torch.tanh(self.concat(outlayer_inputs))
        decoder_outputs = self.fc(outlayer_outputs)
        # decoder_outputs = self.dropout(decoder_outputs)
        return decoder_outputs, last_hidden

    def forward(self, encoder_inputs, encoder_lens, decoder_inputs, decoder_lens):
        encoder_outputs, encoder_last_hidden = self.encode(
            encoder_inputs, encoder_lens)
        encoder_last_hidden = self._fix_hidden(encoder_last_hidden)
        decoder_outputs, _ = self.decode(
            encoder_outputs, encoder_lens, encoder_last_hidden, decoder_inputs, decoder_lens)
        outputs = F.log_softmax(decoder_outputs, dim=2)
        return outputs

    def generate(self, encoder_inputs, encoder_lens,
                 decoder_start_input, max_len, beam_size=1, eos_val=None):
        encoder_outputs, encoder_last_hidden = self.encode(encoder_inputs, encoder_lens)
        encoder_last_hidden = self._fix_hidden(encoder_last_hidden)
        beamseqs = BeamSeqs(beam_size=beam_size)
        beamseqs.init_seqs(seqs=decoder_start_input[0], init_state=encoder_last_hidden)
        done = False
        for i in range(max_len):
            for j, (seqs, _, last_token, last_hidden) in enumerate(beamseqs.current_seqs):
                if beamseqs.check_and_add_to_terminal_seqs(j, eos_val):
                    if len(beamseqs.terminal_seqs) >= beam_size:
                        done = True
                        break
                    continue
                out, last_hidden = self.decode(encoder_outputs, encoder_lens,
                                               last_hidden, last_token.unsqueeze(0))
                _output = F.log_softmax(out.squeeze(0), dim=1).squeeze(0)
                scores, tokens = _output.topk(beam_size * 2)
                for k in range(beam_size * 2):
                    score, token = scores.data[k], tokens[k]
                    token = token.unsqueeze(0)
                    beamseqs.add_token_to_seq(j, token, score, last_hidden)
            if done:
                break
            beamseqs.update_current_seqs()
        final_seqs = beamseqs.return_final_seqs()
        return final_seqs[0].unsqueeze(0)

    @staticmethod
    def _fix_hidden(hidden):
        # The encoder hidden is  (layers*directions) x batch x dim.
        # We need to convert it to layers x batch x (directions*dim).
        hidden = torch.cat([hidden[0:hidden.size(0):2],
                            hidden[1:hidden.size(0):2]], 2)
        return hidden

    def flatten_parameters(self):
        self.encoder.flatten_parameters()
        self.decoder.flatten_parameters()

    def run_batch(self, batch):
        enc_inps, enc_lens = batch.enc_inps
        dec_inps, dec_lens = batch.dec_inps
        dec_tgts, _ = batch.dec_tgts
        dec_probs = self.forward(
            encoder_inputs=enc_inps, encoder_lens=enc_lens,
            decoder_inputs=dec_inps, decoder_lens=dec_lens)

        decoder_probs_pack = dec_probs.view(-1, dec_probs.size(2))
        decoder_targets_pack = dec_tgts.view(-1)
        loss = self.loss_fn(decoder_probs_pack, decoder_targets_pack)
        decoder_probs_pack = batch_unpadding(dec_probs, dec_lens)
        decoder_targets_pack = batch_unpadding(dec_tgts, dec_lens)
        _, pred = decoder_probs_pack.max(1)
        num_correct = pred.eq(decoder_targets_pack).sum().item()
        num_words = pred.size(0)
        result_dict = {
            "loss": loss,
            "num_correct": num_correct,
            "num_words": num_words,
        }
        return result_dict

    def predict_batch(self, batch, max_len=20, beam_size=4, eos_val=0):
        enc_inps, enc_lens = batch.enc_inps
        dec_start_inps = batch.dec_start_inps
        preds = self.generate(enc_inps, enc_lens, dec_start_inps,
                             max_len, beam_size, eos_val).squeeze(2)
        preds = preds.data.cpu().numpy()
        return preds