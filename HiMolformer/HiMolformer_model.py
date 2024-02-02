import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import re
import pytorch_lightning as pl
import os
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score
from HiMol.HiMol_model import GNN, GNN_graphpred
from Molformer.Molformer_model import LightningModule

class HiMolformer(pl.LightningModule):
    def __init__(self, config, tokenizer):
        super(HiMolformer, self).__init__()
        self.save_hyperparameters(config)
        self.config = config
        self.epoch_counter = 0
        self.HiMol = GNN_graphpred(
            num_layer=config['HiMol']['num_layer'], 
            emb_dim=config['HiMol']['emb_dim'], 
            JK=config['HiMol']['JK'], 
            drop_ratio=config['HiMol']['dropout'], 
            gnn_type=config['HiMol']['gnn_type']
        )
        if config['Molformer']['pretrained_path'] == '':
            print("# training from scratch")
            self.Molformer = LightningModule(config, tokenizer)
        else:
            print(f"# loaded pre-trained model from {config['Molformer']['pretrained_path']}")
            self.Molformer = LightningModule(config, tokenizer).load_from_checkpoint(config['Molformer']['pretrained_path'], strict=False, config=config, tokenizer=tokenizer, vocab=len(tokenizer.vocab))

        if config['pretrain_freezing'] == True:
            for param in self.Molformer.parameters():
                param.requires_grad = False

        self.HiMol_embedding = config['HiMol']['emb_dim']
        self.Molformer_embedding = config['Molformer']['n_embd']

        if config['emb_type'] == 'Molformer':
            self.embed_dim = self.Molformer_embedding
        elif config['emb_type'] == 'HiMol':
            self.embed_dim = self.HiMol_embedding
        elif config['emb_type'] == 'concat':
            self.embed_dim = self.HiMol_embedding + self.Molformer_embedding
        elif config['emb_type'] == 'cross_attention' or config['emb_type'] == 'cnn':
            self.embed_dim = [self.HiMol_embedding, self.Molformer_embedding]
        elif config['emb_type'] == 'sum' or config['emb_type'] == 'avg':
            assert config['HiMol']['emb_dim'] == config['Molformer']['n_embd'], \
                "HiMol embedding dimension and Molformer embedding dimension do not match."
            self.embed_dim = self.Molformer_embedding
        else:
            raise ValueError("Invalid embedding type.")

        if config['decoder'] == 'MLP':
            self.net = self.MLP_Net(
                self.embed_dim, dims=config['Molformer']['dims'], dropout=config['dropout'])
        elif config['decoder'] == 'cross_attention':
            self.net = self.cross_attention_Net(
                self.embed_dim, dims=config['Molformer']['dims'], dropout=config['dropout'])
        elif config['decoder'] == 'self_attention':
            self.net = self.self_attention_Net(
                self.embed_dim, dropout=config['dropout'])
        elif config['decoder'] == 'cnn':
            self.net = self.cnn_Net(
                self.embed_dim, dropout=config['dropout'])

        n_vocab, d_emb = len(tokenizer.vocab), config['Molformer']['n_embd']
        self.pos_emb = None
        self.tok_emb = nn.Embedding(n_vocab, config['Molformer']['n_embd'])
        self.drop = nn.Dropout(config['Molformer']['d_dropout'])
        self.min_loss = {
            self.config['target'] + "min_valid_loss": torch.finfo(torch.float32).max,
            self.config['target'] + "min_epoch": 0,
        }
        
        if config['loss'] == 'MAE':
            self.loss_fn = torch.nn.L1Loss()
        elif self.config['loss'] == 'MSE' or self.config['loss'] == 'RMSE':
            self.loss_fn = torch.nn.MSELoss()
        elif self.config['loss'] == 'BCE':
            self.loss_fn = torch.nn.BCELoss()
        self.apply(self._init_weights)  

    class MLP_Net(nn.Module):

        def __init__(self, smiles_embed_dim, dims, dropout=0.1):
            super().__init__()
            self.desc_skip_connection = True 
            self.fcs = []
            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, 1)

        def forward(self, smiles_emb):
            x_out = self.fc1(smiles_emb)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)
            if self.desc_skip_connection is True:
                x_out = x_out + smiles_emb
            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = z + x_out
            z = self.final(z)
            return z
        
    class cnn_Net(nn.Module):
        def __init__(self, smiles_embed_dim, dropout=0.1):
            super().__init__()
            self.cnn0 = nn.Conv1d(in_channels=2, out_channels=128, kernel_size=3, stride=1, padding=1)
            self.cnn1 = nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1)
            self.cnn2 = nn.Conv1d(in_channels=64, out_channels=32, kernel_size=3, stride=1, padding=1)
            self.fc1 = nn.Linear(32 * smiles_embed_dim[1], 256)
            self.fc2 = nn.Linear(256, 1)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

        def forward(self, smiles_emb):
            HiMol_embedding = smiles_emb[0].unsqueeze(1)  
            Molformer_embedding = smiles_emb[1].unsqueeze(1)  
            combined_embedding = torch.cat([HiMol_embedding, Molformer_embedding], dim=1)  
            x_out = self.cnn0(combined_embedding)
            x_out = self.cnn1(x_out)
            x_out = self.cnn2(x_out)
            x_out = x_out.view(x_out.size(0), -1)  
            x_out = self.relu(self.fc1(x_out))
            x_out = self.dropout(x_out)
            x_out = self.fc2(x_out)
            return x_out

    class self_attention_Net(nn.Module):
        def __init__(self, smiles_embed_dim, n_layers=3, dropout=0.1, n_heads=8):
            super().__init__()
            self.n_layers = n_layers
            self.desc_skip_connection = True
            self.attentions = nn.ModuleList([nn.MultiheadAttention(smiles_embed_dim, n_heads) for _ in range(n_layers)])
            self.norms = nn.ModuleList([nn.LayerNorm(smiles_embed_dim) for _ in range(n_layers)])
            self.fc1 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim, smiles_embed_dim)
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim, 1)

        def forward(self, smiles_emb):
            x_out = smiles_emb.unsqueeze(1)
            for i in range(self.n_layers):
                attn_output, _ = self.attentions[i](x_out, x_out, x_out)
                if self.desc_skip_connection:
                    x_out = attn_output + x_out
                x_out = self.norms[i](x_out)
            x_out = x_out.squeeze(1)
            x_out = self.fc1(x_out)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)
            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + x_out)
            else:
                z = self.final(z)
            return z
        
    class cross_attention_Net(nn.Module):

        def __init__(self, smiles_embed_dim, dropout=0.1, n_heads=8):
            super().__init__()
            self.desc_skip_connection = True 
            self.attention = nn.MultiheadAttention(smiles_embed_dim[1], n_heads)
            self.norm1 = nn.LayerNorm(smiles_embed_dim[1])
            self.fc1 = nn.Linear(smiles_embed_dim[1], smiles_embed_dim[1])
            self.dropout1 = nn.Dropout(dropout)
            self.relu1 = nn.GELU()
            self.fc2 = nn.Linear(smiles_embed_dim[1], smiles_embed_dim[1])
            self.dropout2 = nn.Dropout(dropout)
            self.relu2 = nn.GELU()
            self.final = nn.Linear(smiles_embed_dim[1], 1)

        def forward(self, smiles_emb):
            query1 = smiles_emb[1].unsqueeze(1)
            key1 = smiles_emb[0].unsqueeze(1)
            value1 = smiles_emb[0].unsqueeze(1)
            query2 = smiles_emb[0].unsqueeze(1)
            key2 = smiles_emb[1].unsqueeze(1)
            value2 = smiles_emb[1].unsqueeze(1)
            attn_out1, _ = self.attention(query1, key1, value1)
            if self.desc_skip_connection:
                x_out1 = attn_out1 + query1
            x_out1 = self.norm1(x_out1)
            attn_out2, _ = self.attention(query2, key2, value2)
            if self.desc_skip_connection:
                x_out2 = attn_out2 + query2  
            x_out2 = self.norm1(x_out2)
            x_out = x_out1 + x_out2
            attn_out, _ = self.attention(x_out, x_out, x_out)
            if self.desc_skip_connection:
                x_out = attn_out + x_out   
            x_out = self.norm1(x_out)
            x_out = self.fc1(x_out)
            x_out = self.dropout1(x_out)
            x_out = self.relu1(x_out)
            z = self.fc2(x_out)
            z = self.dropout2(z)
            z = self.relu2(z)
            if self.desc_skip_connection is True:
                z = self.final(z + query1 + query2)
            else:
                z = self.final(z)
            z = self.final(x_out)
            return z

    def get_loss(self, smiles_emb, target):
        z_pred = self.net.forward(smiles_emb).squeeze()
        target = target.float()
        if self.config['loss'] == 'BCE':
            return self.loss_fn(torch.sigmoid(z_pred), target), torch.sigmoid(z_pred), target
        else:
            return self.loss_fn(z_pred, target), z_pred, target
    
    def _init_weights(self, module):
        if module != self.Molformer:
            if isinstance(module, (nn.Linear, nn.Embedding)):
                print('init weight did!')
                module.weight.data.normal_(mean=0.0, std=0.02)
                if isinstance(module, nn.Linear) and module.bias is not None:
                    module.bias.data.zero_()
                    print('init bias did!')
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)

    def configure_optimizers(self):
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Conv3d)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding, torch.nn.BatchNorm1d, torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif fpn == 'net.attention.in_proj_weight':
                    no_decay.add(fpn)
                elif fpn == 'net.attentions.0.in_proj_weight':
                    no_decay.add(fpn)
                elif fpn == 'net.attentions.1.in_proj_weight':
                    no_decay.add(fpn)
                elif fpn == 'net.attentions.2.in_proj_weight':
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        if self.pos_emb != None:
            no_decay.add('pos_emb')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )
        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": 0.0},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        betas = (0.9, 0.99)
        print('betas are {}'.format(betas))
        learning_rate = self.config['lr_start'] * self.config['lr_multiplier']
        optimizer = optim.Adam(optim_groups, lr=learning_rate, betas=betas)
        return optimizer

    def training_step(self, batch, batch_idx):
        HiMol_batch, Molformer_batch = batch
        HiMol_input = HiMol_batch[0]
        Molformer_idx = Molformer_batch[0]
        Molformer_mask = Molformer_batch[1]
        targets = Molformer_batch[-1]

        loss = 0
        loss_tmp = 0

        if self.config['emb_type'] == 'Molformer':
            loss_input = self.Molformer(Molformer_batch)
        elif self.config['emb_type'] == 'HiMol':
            loss_input = self.HiMol(HiMol_batch)
        elif self.config['emb_type'] == 'concat':
            loss_input = torch.cat((self.HiMol(HiMol_batch)*self.config['HiMol_weight'], self.Molformer(Molformer_batch)*self.config['Molformer_weight']), dim=1)
        elif self.config['emb_type'] == 'sum':
            loss_input = self.HiMol(HiMol_batch)*self.config['HiMol_weight'] + self.Molformer(Molformer_batch)*self.config['Molformer_weight']
        elif self.config['emb_type'] == 'avg':
            loss_input = (self.HiMol(HiMol_batch)*self.config['HiMol_weight'] + self.Molformer(Molformer_batch)*self.config['Molformer_weight'])/2
        elif self.config['emb_type'] == 'cross_attention' or self.config['emb_type'] == 'cnn':
            loss_input = [self.HiMol(HiMol_batch), self.Molformer(Molformer_batch)]
        else:
            raise ValueError("Invalid embedding type.")

        if self.config['gaussian_noise'] == True:
            gaussian_noise_np = np.random.normal(0.0, 0.0001, loss_input.shape)
            gaussian_noise = torch.from_numpy(gaussian_noise_np).float()
            gaussian_noise = gaussian_noise.to(config['device'])
            loss_input = loss_input + gaussian_noise

        if self.config['emb_save'] == True and batch_idx == 0:
            self.epoch_counter += 1
            HiMol_embeddings = self.HiMol(HiMol_batch).detach().cpu().numpy()
            Molformer_embeddings = self.Molformer(Molformer_batch).detach().cpu().numpy()
            HiMolformer_embeddings = loss_input.detach().cpu().numpy()
            HiMol_embeddings_df = pd.DataFrame(HiMol_embeddings)
            Molformer_embeddings_df = pd.DataFrame(Molformer_embeddings)
            HiMolformer_embeddings_df = pd.DataFrame(HiMolformer_embeddings)
            HiMol_embeddings_df.insert(0, 'epoch', self.epoch_counter)
            Molformer_embeddings_df.insert(0, 'epoch', self.epoch_counter)
            HiMolformer_embeddings_df.insert(0, 'epoch', self.epoch_counter)
            HiMol_save_args = {'mode': 'a', 'index': False, 'header': not os.path.exists('embeddings/HiMol_embeddings.csv')}
            Molformer_save_args = {'mode': 'a', 'index': False, 'header': not os.path.exists('embeddings/Molformer_embeddings.csv')}
            HiMolformer_save_args = {'mode': 'a', 'index': False, 'header': not os.path.exists('embeddings/HiMolformer_embeddings.csv')}
            HiMol_embeddings_df.to_csv('embeddings/HiMol_embeddings.csv', **HiMol_save_args)
            Molformer_embeddings_df.to_csv('embeddings/Molformer_embeddings.csv', **Molformer_save_args)
            HiMolformer_embeddings_df.to_csv('embeddings/HiMolformer_embeddings.csv', **HiMolformer_save_args)
        if self.config['loss'] == 'MAE' or self.config['loss'] == 'MSE' or self.config['loss'] == 'BCE':
            loss, pred, actual = self.get_loss(loss_input, targets)
        elif self.config['loss'] == 'RMSE':
            loss_tmp, pred, actual = self.get_loss(loss_input, targets)
            loss = torch.sqrt(loss_tmp)
        self.log('train_loss', loss, on_step=True)
        return {"loss": loss}

    def validation_step(self, batch, batch_idx):
        HiMol_batch, Molformer_batch = batch
        HiMol_input = HiMol_batch[0]
        Molformer_idx = Molformer_batch[0]
        Molformer_mask = Molformer_batch[1]
        targets = Molformer_batch[-1]

        if self.config['emb_type'] == 'Molformer':
            loss_input = self.Molformer(Molformer_batch)
        elif self.config['emb_type'] == 'HiMol':
            loss_input = self.HiMol(HiMol_batch)
        elif self.config['emb_type'] == 'concat':
            loss_input = torch.cat((self.HiMol(HiMol_batch)*self.config['HiMol_weight'], self.Molformer(Molformer_batch)*self.config['Molformer_weight']), dim=1)
        elif self.config['emb_type'] == 'sum':
            loss_input = self.HiMol(HiMol_batch)*self.config['HiMol_weight'] + self.Molformer(Molformer_batch)*self.config['Molformer_weight']
        elif self.config['emb_type'] == 'avg':
            loss_input = (self.HiMol(HiMol_batch)*self.config['HiMol_weight'] + self.Molformer(Molformer_batch)*self.config['Molformer_weight'])/2
        elif self.config['emb_type'] == 'cross_attention' or self.config['emb_type'] == 'cnn':
            loss_input = [self.HiMol(HiMol_batch)/2, self.Molformer(Molformer_batch)/2]
        else:
            raise ValueError("Invalid embedding type.")

        if self.config['loss'] == 'MAE' or self.config['loss'] == 'MSE' or self.config['loss'] == 'BCE':
            loss, pred, actual = self.get_loss(loss_input, targets)
        elif self.config['loss'] == 'RMSE':
            loss_tmp, pred, actual = self.get_loss(loss_input, targets)
            loss = torch.sqrt(loss_tmp)

        self.log('val_loss', loss, on_step=False, on_epoch=True)
        
        return {"val_loss": loss, "pred": pred, "actual": actual}
    
    def validation_epoch_end(self, outputs):
        val_losses = [x['val_loss'] for x in outputs]
        avg_loss = torch.stack(val_losses).mean()
        print(f"avg_val_loss: {avg_loss.item()}")

        results_file = os.path.join(self.config['results_dir'], "validation_results.csv")
        with open(results_file, 'a') as f:
            f.write(f"{self.current_epoch},{avg_loss.item()}\n")

        return {"avg_val_loss": avg_loss}
    
    def test_step(self, batch, batch_idx):
        HiMol_batch, Molformer_batch = batch

        HiMol_input = HiMol_batch[0]
        Molformer_idx = Molformer_batch[0]
        Molformer_mask = Molformer_batch[1]
        targets = Molformer_batch[-1]

        if self.config['emb_type'] == 'Molformer':
            loss_input = self.Molformer(Molformer_batch)
        elif self.config['emb_type'] == 'HiMol':
            loss_input = self.HiMol(HiMol_batch)
        elif self.config['emb_type'] == 'concat':
            loss_input = torch.cat((self.HiMol(HiMol_batch)*self.config['HiMol_weight'], self.Molformer(Molformer_batch)*self.config['Molformer_weight']), dim=1)
        elif self.config['emb_type'] == 'sum':
                    loss_input = self.HiMol(HiMol_batch)*self.config['HiMol_weight'] + self.Molformer(Molformer_batch)*self.config['Molformer_weight']
        elif self.config['emb_type'] == 'avg':
            loss_input = (self.HiMol(HiMol_batch)*self.config['HiMol_weight'] + self.Molformer(Molformer_batch)*self.config['Molformer_weight'])/2
        elif self.config['emb_type'] == 'cross_attention' or self.config['emb_type'] == 'cnn':
            loss_input = [self.HiMol(HiMol_batch)/2, self.Molformer(Molformer_batch)/2]
        else:
            raise ValueError("Invalid embedding type.")

        if self.config['loss'] == 'BCE':
            pred_probs = torch.sigmoid(self.net.forward(loss_input)).squeeze()
            pred = (pred_probs > 0.5).float()
            targets = targets.float()

            acc = accuracy_score(targets.cpu().numpy(), pred.cpu().numpy())
            auc = roc_auc_score(targets.cpu().numpy(), pred_probs.cpu().numpy())
            precision = precision_score(targets.cpu().numpy(), pred.cpu().numpy(), average='binary')
            f1 = f1_score(targets.cpu().numpy(), pred.cpu().numpy(), average='binary')

            print('Test Metrics - Accuracy: {}, AUC: {}, Precision: {}, F1: {}'.format(acc, auc, precision, f1))

            self.log('test_acc', torch.tensor(acc), on_step=False, on_epoch=True)
            self.log('test_auc', torch.tensor(auc), on_step=False, on_epoch=True)
            self.log('test_precision', torch.tensor(precision), on_step=False, on_epoch=True)
            self.log('test_f1', torch.tensor(f1), on_step=False, on_epoch=True)

            return {'test_acc': torch.tensor(acc), 'test_auc': torch.tensor(auc), 'test_precision': torch.tensor(precision), 'test_f1': torch.tensor(f1)}

        else:
            pred = self.net.forward(loss_input).squeeze()
            mae_loss = torch.nn.L1Loss()(pred, targets.float())
            mse_loss = torch.nn.MSELoss()(pred, targets.float())
            rmse_loss = torch.sqrt(mse_loss)

            print('Test Losses - MAE: {}, MSE: {}, RMSE: {}'.format(mae_loss, mse_loss, rmse_loss))

            self.log('test_mae_loss', mae_loss, on_step=False, on_epoch=True)
            self.log('test_mse_loss', mse_loss, on_step=False, on_epoch=True)
            self.log('test_rmse_loss', rmse_loss, on_step=False, on_epoch=True)

            return {"test_mae_loss": mae_loss, "test_mse_loss": mse_loss, "test_rmse_loss": rmse_loss, "pred": pred, "actual": targets}


    def test_epoch_end(self, outputs):

        if self.config['loss'] == 'BCE':
            avg_test_acc = torch.stack([x['test_acc'] for x in outputs]).mean()
            avg_test_auc = torch.stack([x['test_auc'] for x in outputs]).mean()
            avg_test_precision = torch.stack([x['test_precision'] for x in outputs]).mean()
            avg_test_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()

            test_results_file = os.path.join(self.config['results_dir'], f"test_results.csv")

            with open(test_results_file, 'a', encoding='utf-8') as f:
                f.write(f"Average Test Accuracy: {avg_test_acc.item()}\n")
                f.write(f"Average Test AUC: {avg_test_auc.item()}\n")  
                f.write(f"Average Test Precision: {avg_test_precision.item()}\n")
                f.write(f"Average Test F1 Score: {avg_test_f1.item()}\n")

            best_model_score = self.trainer.checkpoint_callback.best_model_score
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            best_epoch = extract_epoch_from_checkpoint(best_model_path)

            with open(test_results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nBest Model Path: {best_model_path}\n")
                f.write(f"Target: {self.config['target']}\n")
                f.write(f"embed: {self.config['emb_type']}_{self.config['decoder']}{self.config['HiMol']['emb_dim']}_freezing_{self.config['pretrain_freezing']}\n")
                f.write(f"Train_valid_loss: {self.config['loss']}\n")
                f.write(f"Best Validation Score: {best_model_score}\n")
                f.write(f"Best Epoch: {best_epoch}\n")

            test_results = {
                "avg_test_acc": avg_test_acc.item(),
                "avg_test_auc": avg_test_auc.item(),
                "avg_test_precision": avg_test_precision.item(),
                "avg_test_f1": avg_test_f1.item(),
                "best_epoch": best_epoch,
                "train_valid_loss": self.config['loss']
            }

            return test_results

        else:
            avg_test_mae_loss = torch.stack([x['test_mae_loss'] for x in outputs]).mean()
            avg_test_mse_loss = torch.stack([x['test_mse_loss'] for x in outputs]).mean()
            avg_test_rmse_loss = torch.stack([x['test_rmse_loss'] for x in outputs]).mean()

            all_preds = torch.cat([x['pred'] for x in outputs])
            all_actuals = torch.cat([x['actual'] for x in outputs])

            all_preds_np = all_preds.detach().cpu().numpy()
            all_actuals_np = all_actuals.detach().cpu().numpy()

            test_results_file = os.path.join(self.config['results_dir'], f"test_results.csv")

            header = "Actual,Predicted"
            test_results_data = np.vstack((all_actuals_np, all_preds_np)).T
            
            np.savetxt(test_results_file, test_results_data, delimiter=",", header=header, comments='', fmt='%s')

            best_model_score = self.trainer.checkpoint_callback.best_model_score
            best_model_path = self.trainer.checkpoint_callback.best_model_path
            best_epoch = extract_epoch_from_checkpoint(best_model_path)

            with open(test_results_file, 'a', encoding='utf-8') as f:
                f.write(f"\nBest Model Path: {best_model_path}\n")
                f.write(f"Target: {self.config['target']}\n")
                f.write(f"embed: {self.config['emb_type']}_{self.config['decoder']}{self.config['HiMol']['emb_dim']}_freezing_{self.config['pretrain_freezing']}\n")
                f.write(f"Train_valid_loss: {self.config['loss']}\n")
                f.write(f"Best Validation Score: {best_model_score}\n")
                f.write(f"Best Epoch: {best_epoch}\n")
                f.write(f"Average Test MAE Loss: {avg_test_mae_loss.item()}\n")
                f.write(f"Average Test MSE Loss: {avg_test_mse_loss.item()}\n")
                f.write(f"Average Test RMSE Loss: {avg_test_rmse_loss.item()}\n")

            avg_test_mae_loss = torch.stack([x['test_mae_loss'] for x in outputs]).mean()
            avg_test_mse_loss = torch.stack([x['test_mse_loss'] for x in outputs]).mean()
            avg_test_rmse_loss = torch.stack([x['test_rmse_loss'] for x in outputs]).mean()

            test_results = {
                "avg_test_mae_loss": avg_test_mae_loss.item(),
                "avg_test_mse_loss": avg_test_mse_loss.item(),
                "avg_test_rmse_loss": avg_test_rmse_loss.item(),
                "best_epoch": best_epoch,
                "train,valid_loss": self.config['loss']
            }

            return test_results
        
def extract_epoch_from_checkpoint(checkpoint_path):

    match = re.search(r'epoch=(\d+)', checkpoint_path)
    if match:
        return int(match.group(1))
    else:
        return None