import torch
import os
import pytorch_lightning as pl
# from pytorch_lightning.utilities import rank_zero_warn, rank_zero_only, seed
from Molformer.tokenizer.tokenizer import MolTranBertTokenizer
from fast_transformers.masking import LengthMask as LM
from Molformer.rotate_attention.rotate_builder import RotateEncoderBuilder as rotate_builder
import pandas as pd
# from scipy.stats import pearsonr
from torch.utils.data import DataLoader
# from sklearn.metrics import r2_score
from Molformer.utils import normalize_smiles

def get_dataset(data_root, filename, dataset_len, aug=None, target=None):
    df = pd.read_csv(os.path.join(data_root, filename))
    print("Length of dataset:", len(df))
    if dataset_len:
        df = df.head(dataset_len)
        print("Warning entire dataset not used:", len(df))
    dataset = PropertyPredictionDataset(df, target, aug)
    return dataset

class PropertyPredictionDataset(torch.utils.data.Dataset):
    def __init__(self, df, target=None, tokenizer=MolTranBertTokenizer('Molformer/bert_vocab.txt'), aug=True):
        self.df = df
        self.target = target
        all_smiles = df["smiles"].tolist()
        self.original_smiles = []
        self.original_canonical_map = {
            smi: normalize_smiles(smi, canonical=True, isomeric=False) for smi in all_smiles
        }

        self.tokenizer = MolTranBertTokenizer('Molformer/bert_vocab.txt')
        if target:
            all_measures = df[target].tolist()
            self.measure_map = {all_smiles[i]: all_measures[i] for i in range(len(all_smiles))}

        for i in range(len(all_smiles)):
            smi = all_smiles[i]
            if smi in self.original_canonical_map.keys():
                self.original_smiles.append(smi)

        print(f"Embeddings not found for {len(all_smiles) - len(self.original_smiles)} molecules")

        self.aug = aug
        self.is_measure_available = "measure" in df.columns

    def __getitem__(self, index):
        original_smiles = self.original_smiles[index]
        canonical_smiles = self.original_canonical_map[original_smiles]
        if self.target:
            return canonical_smiles, self.measure_map[original_smiles]
        else:
            return canonical_smiles

    def __len__(self):
        return len(self.original_smiles)

class PropertyPredictionDataModule(pl.LightningDataModule):
    def __init__(self, config, splitted_data):
        super(PropertyPredictionDataModule, self).__init__()
        self.config = config
        self.Molformer_config = config['Molformer']
        self.splitted_data = splitted_data
        self.tokenizer = MolTranBertTokenizer('Molformer/bert_vocab.txt')

    def get_split_dataset_filename(self, split):
        return f"{self.config['split']}_{self.config['target']}_" + split + "_dataset.csv"

    def prepare_data(self):
        # print("Inside prepare_dataset")
        train_filename = PropertyPredictionDataModule.get_split_dataset_filename(self, "train")
        valid_filename = PropertyPredictionDataModule.get_split_dataset_filename(self, "valid")
        test_filename = PropertyPredictionDataModule.get_split_dataset_filename(self, "test")

        self.train_ds = get_dataset(
            os.path.join(self.splitted_data, 'raw'),
            train_filename,
            self.Molformer_config.get('train_dataset_length', None),
            self.Molformer_config.get('aug', None),
            target=self.config['target'],
        )

        self.val_ds = get_dataset(
            os.path.join(self.splitted_data, 'raw'),
            valid_filename,
            self.Molformer_config.get('eval_dataset_length', None),
            aug=False,
            target=self.config['target'],
        )

        self.test_ds = get_dataset(
            os.path.join(self.splitted_data, 'raw'),
            test_filename,
            self.Molformer_config.get('eval_dataset_length', None),
            aug=False,
            target=self.config['target'],
        )


    def prepare_inference_data(self):
        inference_filename = f"inference_dataset.csv"

        self.inference_ds = get_dataset(
            os.path.join(self.splitted_data, 'raw'),
            inference_filename,
            self.Molformer_config.get('eval_dataset_length', None),
            aug=False,
            target=None,
        )

    def collate(self, batch):
        tokens = self.tokenizer.batch_encode_plus([smile[0] for smile in batch], padding=True, add_special_tokens=True)
        if self.config['inference_mode'] == True:
            return torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask'])
        else:
            return (torch.tensor(tokens['input_ids']), torch.tensor(tokens['attention_mask']), torch.tensor([smile[1] for smile in batch]))
 
    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.config['batch_size'],
            num_workers=self.config['num_workers'],
            shuffle=False,
            collate_fn=self.collate,
        )