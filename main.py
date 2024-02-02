import yaml
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error, mean_absolute_error
import random
import time
import datetime
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
import os
import pandas as pd
from Molformer.tokenizer.tokenizer import MolTranBertTokenizer
from HiMolformer_splitter import scaffold_split, random_split
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score, f1_score
from HiMolformer_dataset import CombinedDataModule
from HiMolformer_model import HiMolformer
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  
    random.seed(seed)  
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(config):
    set_seed(config['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pos_emb_type = 'rot'

    if config['inference_mode'] == True:
        splitted_data = config['inference_data_path']
    else:
        if config['split'] == 'scaffold':
            splitted_data = scaffold_split(config)
        elif config['split'] == 'random':
            splitted_data = random_split(config)

    tokenizer = MolTranBertTokenizer('Molformer/bert_vocab.txt')
    data_module = CombinedDataModule(config, splitted_data)

    if config['inference_mode'] == True:
        tokenizer = MolTranBertTokenizer('Molformer/bert_vocab.txt')
        checkpoint_path = config['inference_checkpoint']
        if not checkpoint_path:
            raise ValueError("checkpoint_path is empty. Please provide a valid checkpoint path.")
        model = HiMolformer.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer, config=config)
        model.to(device)
        inference_data_loader = data_module.inference_dataloader()
        model.eval()

        results = []
        with torch.no_grad():
            for batch in inference_data_loader:
                HiMol_batch, Molformer_batch = batch

                HiMol_batch = HiMol_batch.to(device)
                Molformer_batch = [tensor.to(device) for tensor in Molformer_batch]

                if config['emb_type'] == 'Molformer':
                    loss_input = model.Molformer(Molformer_batch)
                elif config['emb_type'] == 'HiMol':
                    loss_input = model.HiMol(HiMol_batch)
                elif config['emb_type'] == 'concat':
                    loss_input = torch.cat([model.HiMol(HiMol_batch), model.Molformer(Molformer_batch)], dim=1)
                elif config['emb_type'] == 'sum':
                    loss_input = model.HiMol(HiMol_batch) + model.Molformer(Molformer_batch)
                elif config['emb_type'] == 'avg':
                    loss_input = (model.HiMol(HiMol_batch) + model.Molformer(Molformer_batch)) / 2
                elif config['emb_type'] == 'cross_attention' or config['emb_type'] == 'cnn':
                    loss_input = [model.HiMol(HiMol_batch) / 2, model.Molformer(Molformer_batch) / 2]
                else:
                    raise ValueError("Invalid embedding type.")

                output = model.net(loss_input).squeeze()
                if config['loss'] != 'BCE':
                    output = torch.clamp(output, min=0, max=100)
                results.extend(output.cpu().numpy().flatten())

        results_df = pd.DataFrame({'prediction': results})
        results_dir = 'result/inference'
        results_file = os.path.join(results_dir, f"{config['target']}_{config['loss']}_{config['emb_type']}{config['HiMol']['emb_dim']}_inference_results.csv")

        inference_data_path = f"{config['inference_data_path']}/raw/{config['inference_data_name']}.csv"
        inference_df = pd.read_csv(inference_data_path)
        required_columns = inference_df[['smiles', config['target']]]
        results_df = pd.concat([required_columns, results_df], axis=1)

        target_values = results_df[config['target']]
        prediction_values = results_df['prediction']

        if config['loss'] == 'BCE':
            prediction_tensor = torch.tensor(prediction_values.to_numpy())
            prob_predictions = torch.sigmoid(prediction_tensor).numpy()
            binary_predictions = (prob_predictions >= 0.5).astype(int)
            results_df['prediction'] = binary_predictions
            results_df['prob_predictions'] = prob_predictions
            acc = accuracy_score(target_values, binary_predictions)
            auc = roc_auc_score(target_values, prob_predictions)
            f1 = f1_score(target_values, binary_predictions)
            print('ACC: {}'.format(acc))
            print('AUC: {}'.format(auc))
            print('F1: {}'.format(f1))

        else:
            mae = mean_absolute_error(target_values, prediction_values)
            mse = mean_squared_error(target_values, prediction_values)
            rmse = np.sqrt(mse)
            print('MAE: {}'.format(mae))
            print('MSE: {}'.format(mse))
            print('RMSE: {}'.format(rmse))

        results_df.to_csv(results_file, index=False)
        print(f"Inference results saved to {results_file}")

    else:
        current_datetime = datetime.datetime.now()
        formatted_date = current_datetime.strftime('%Y%m%d')
        checkpoints_folder = config['checkpoints_folder']
        checkpoint_root = os.path.join(checkpoints_folder, config['loss'] + '_' + config['target'] + '_' + config['emb_type'] + str(config['HiMol']['emb_dim']) + '_' + config['decoder'] + '_freeze' + str(config['pretrain_freezing']) + '_seed' + str(config['seed']))
        # config['checkpoint_root'] = checkpoint_root
        os.makedirs(checkpoint_root, exist_ok=True)
        checkpoint_dir = os.path.join(checkpoint_root, "models")
        results_dir = os.path.join(checkpoint_root, "results")
        config['results_dir'] = results_dir
        config['checkpoint_dir'] = checkpoint_dir
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_path = os.path.join(checkpoints_folder, config['target'])
        checkpoint_callback = pl.callbacks.ModelCheckpoint(
            period=1, 
            save_last=True,
            dirpath=checkpoint_dir, 
            filename='checkpoint-{epoch}',
            verbose=True)

        tokenizer = MolTranBertTokenizer('Molformer/bert_vocab.txt')
        
        model = HiMolformer(config, tokenizer)

        last_checkpoint_file = os.path.join(checkpoint_dir, "last.ckpt")
        resume_from_checkpoint = None
        if config['train_checkpoint']:
            resume_from_checkpoint = config['train_checkpoint']
            print(f"starting training from : {resume_from_checkpoint}")
        elif os.path.isfile(last_checkpoint_file):
            print(f"resuming training from : {last_checkpoint_file}")
            resume_from_checkpoint = last_checkpoint_file
        else:
            print(f"training from scratch")

        early_stop_callback = EarlyStopping(
            monitor='val_loss',
            min_delta=0.00,    
            patience=15,       
            verbose=True,      
            mode='min'         
            )

        trainer = pl.Trainer(
        max_epochs=config['epochs'],
        default_root_dir=checkpoint_root,
        gpus=1,
        resume_from_checkpoint=resume_from_checkpoint,
        checkpoint_callback=checkpoint_callback,
        num_sanity_val_steps=0,
        callbacks=[early_stop_callback]
        )

        tic = time.perf_counter()
        trainer.fit(model, data_module)
        toc = time.perf_counter()
        print('Training Time: {}'.format(toc - tic))

        checkpoint_path = trainer.checkpoint_callback.best_model_path
        if checkpoint_path:
            print(f"Loading best model from: {checkpoint_path}")
            model = HiMolformer.load_from_checkpoint(checkpoint_path, tokenizer=tokenizer, config=config)

        trainer.test(model, data_module.test_dataloader())
        test_results = trainer.test(model, data_module.test_dataloader())
        return test_results[0] 

def append_to_file(filename, line):
    with open(filename, "a") as f:
        
        f.write(line + "\n")

def read_config_file(filepath):
    with open(filepath, 'r') as file:
        config = yaml.safe_load(file)
    return config

if __name__ == "__main__":
    config_path = 'config.yaml'
    config = read_config_file(config_path)
    main(config)