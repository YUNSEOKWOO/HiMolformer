import torch
from torch import nn
import torch.nn.functional as F
# from pytorch_lightning.loggers import TensorBoardLogger
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from fast_transformers.masking import LengthMask as LM
from Molformer.rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
from fast_transformers.feature_maps import GeneralizedRandomFeatures
from functools import partial
import torch.optim as optim
import numpy as np
# from scipy.stats import pearsonr
# from sklearn.metrics import r2_score
from Molformer.utils import normalize_smiles

class LightningModule(pl.LightningModule):

    def __init__(self, config, tokenizer):
        super(LightningModule, self).__init__()

        self.config = config
        self.Molformer_config = config['Molformer']
        self.tokenizer=tokenizer
        self.min_loss = {
            self.config['target'] + "min_valid_loss": torch.finfo(torch.float32).max,
            self.config['target'] + "min_epoch": 0,
        }

        # Word embeddings layer
        n_vocab, d_emb = len(tokenizer.vocab), self.Molformer_config['n_embd']
        # input embedding stem
        builder = rotate_builder.from_kwargs(
            n_layers=self.Molformer_config['n_layer'],
            n_heads=self.Molformer_config['n_head'],
            query_dimensions=self.Molformer_config['n_embd']//self.Molformer_config['n_head'],
            value_dimensions=self.Molformer_config['n_embd']//self.Molformer_config['n_head'],
            feed_forward_dimensions=self.Molformer_config['n_embd'],
            attention_type='linear',
            feature_map=partial(GeneralizedRandomFeatures, n_dims=self.Molformer_config['num_feats']),
            activation='gelu',
            )
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, self.Molformer_config['n_embd'])
        self.drop = nn.Dropout(self.Molformer_config['d_dropout'])
        self.blocks = builder.get()
        self.lang_model = self.lm_layer(self.Molformer_config['n_embd'], n_vocab)
        self.train_config = config

    def forward(self, input_data):
        if self.config['inference_mode'] == True:
            idx, mask = input_data
        else:
            idx, mask, target = input_data
        token_embeddings = self.tok_emb(idx)  # each index maps to a (learnable) vector
        x = self.drop(token_embeddings)
        x = self.blocks(x, length_mask=LM(mask.sum(-1)))  # if you need length_mask

        input_mask_expanded = mask.unsqueeze(-1).expand(x.size()).float()
        sum_embeddings = torch.sum(x * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        loss_input = sum_embeddings / sum_mask
    
        return loss_input

    class lm_layer(nn.Module):
        def __init__(self, n_embd, n_vocab):
            super().__init__()
            self.embed = nn.Linear(n_embd, n_embd)
            self.ln_f = nn.LayerNorm(n_embd)
            self.head = nn.Linear(n_embd, n_vocab, bias=False)
        def forward(self, tensor):
            tensor = self.embed(tensor)
            tensor = F.gelu(tensor)
            tensor = self.ln_f(tensor)
            tensor = self.head(tensor)
            return tensor

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)