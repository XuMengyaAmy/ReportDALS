import torch
from torch import nn
import copy
from models.containers import ModuleList
from ..captioning_model import CaptioningModel

from torch.autograd import Function

class GRLayer(Function):

    @staticmethod
    def forward(ctx, input):
        ctx.alpha=0.1

        return input.view_as(input)

    @staticmethod
    def backward(ctx, grad_outputs):
        output=grad_outputs.neg() * ctx.alpha
        return output

def grad_reverse(x):
    return GRLayer.apply(x)

# # original one
# class _ImageDA(nn.Module):
#     def __init__(self):
#         super(_ImageDA,self).__init__()
#         self.d_fc1 = nn.Linear(3 * 6* 512, 100)
#         self.bn1 = nn.BatchNorm1d(100)
#         self.reLu=nn.ReLU(inplace=False)
#         self.d_fc2 = nn.Linear( 100,2)  #  self.d_fc2 = nn.Linear( 100,2)
#         self.logsoft = nn.LogSoftmax(dim=1)

#     def forward(self,x):
#         x=grad_reverse(x)
#         x=self.reLu(self.d_fc1(x))
#         x=self.d_fc2(x)
#         return self.logsoft(x)

class _ImageDA(nn.Module):
    def __init__(self):
        super(_ImageDA,self).__init__()
        self.d_fc1 = nn.Linear(3 * 6* 512, 100)
        self.bn1 = nn.BatchNorm1d(100)
        self.reLu=nn.ReLU(inplace=False)
        self.d_fc3 = nn.Linear(100,32)
        self.d_fc2 = nn.Linear(32,3)  
        self.logsoft = nn.LogSoftmax(dim=1)

    def forward(self,x):
        x=grad_reverse(x)
        x=self.reLu(self.d_fc1(x))
        x=self.reLu(self.d_fc3(x))
        x=self.d_fc2(x)
        return self.logsoft(x)


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()
        self.domain_classifier = _ImageDA()

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, images, seq, *args):
        enc_output, mask_enc = self.encoder(images)  
        domain_output = self.domain_classifier(torch.flatten(enc_output, 1))  
        dec_output = self.decoder(seq, enc_output, mask_enc)
        return dec_output, domain_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device),
                None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                self.enc_output, self.mask_enc = self.encoder(visual)
                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(it, self.enc_output, self.mask_enc)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([copy.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))

        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
