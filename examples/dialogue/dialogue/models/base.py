import torch
import torch.nn as nn


class BaseDeepModel(nn.Module):
    def __init__(self):
        super(BaseDeepModel, self).__init__()

    def load_pretrained_embedding(self, pretrained_embedding_matrix):
        tmp = torch.FloatTensor(pretrained_embedding_matrix.float())
        if self.use_cuda:
            tmp = tmp.cuda()
        self.encoder_embedding.weight.data.copy_(tmp)
        self.decoder_embedding.weight.data.copy_(tmp)

    def flatten_parameters(self):
        pass