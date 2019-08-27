import torch


class BeamSeqs(object):
    def __init__(self, beam_size):
        self.current_seqs = []
        self.new_seqs = []
        self.terminal_seqs = []
        self.beam_size = beam_size

    def init_seqs(self, seqs, init_state):
        latest_token = seqs[-1]
        init_score = 0
        self.current_seqs.append((seqs, init_score, latest_token, init_state))

    def add_token_to_seq(self, i, token, new_score, extra_info):
        seq, score, _, _ = self.current_seqs[i]
        seq = torch.cat([seq, token.unsqueeze(0)])
        self.new_seqs.append((seq, score + new_score, token, extra_info))

    def update_current_seqs(self):
        self.current_seqs = self.new_seqs
        self.current_seqs = [item for item in self.current_seqs if item is not None]
        if len(self.current_seqs) > self.beam_size:
            self.current_seqs = sorted(
                self.current_seqs,
                key=lambda x: x[1], # / (((5 + x[0].size(0)) ** 0.6) / 6 ** 0.6),
                reverse=True)[:self.beam_size]
        self.new_seqs = []

    def check_and_add_to_terminal_seqs(self, j, eos_val):
        tmp = self.current_seqs[j]
        seqs = tmp[0]
        if seqs[-1].data[0] == eos_val:
            if seqs.size(0) >= 5:
                self.terminal_seqs.append(self.current_seqs[j])
            self.current_seqs[j] = None
            return True
        else:
            return False

    def return_final_seqs(self):
        if len(self.terminal_seqs) == 0:
            return max(self.current_seqs, key=lambda x: x[1])# / (((5 + x[0].size(0)) ** 0.6) / 6 ** 0.6))
        return max(self.terminal_seqs, key=lambda x: x[1])# / (((5 + x[0].size(0)) ** 0.6) / 6 ** 0.6))