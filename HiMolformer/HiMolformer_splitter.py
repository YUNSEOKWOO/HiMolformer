import deepchem as dc
import pandas as pd
import numpy as np
import os

from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold

def scaffold_split(config):
    df = pd.read_csv(os.path.join(config['data_root'], config['raw_filename']))

    def get_scaffold(smiles):
        mol = Chem.MolFromSmiles(smiles)
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=mol)
        return scaffold

    # Scaffold column
    df['scaffold'] = df['smiles'].apply(get_scaffold)

    Ys = df[[config['target']]].values
    Xs = np.zeros((len(df), 1))  # 'X' is not actually used, but is needed to create the DeepChem dataset
    ids = df['smiles'].values
    dataset = dc.data.DiskDataset.from_numpy(X=Xs, y=Ys, w=np.ones((len(df), 1)), ids=ids)

    scaffoldsplitter = dc.splits.ScaffoldSplitter()
    train_dataset, valid_dataset, test_dataset = scaffoldsplitter.train_valid_test_split(
        dataset,
        frac_train=config['train_ratio'],
        frac_valid=config['valid_ratio'],
        frac_test=config['test_ratio'],
        seed=config['seed']
    )

    csv_root = f"{config['data_root']}/{config['split']}/{config['target']}"

    if not os.path.exists(os.path.join(csv_root, "raw")):
        os.makedirs(os.path.join(csv_root, "raw"))

    def save_dataset(dataset, df, filename):
        dataset_smiles = dataset.ids
        dataset_Y = dataset.y
        dataset_df = pd.DataFrame({'smiles': dataset_smiles, config['target']: dataset_Y.flatten()})
        dataset_df = dataset_df.merge(df[['smiles', 'scaffold']], on='smiles')
        dataset_df.to_csv(os.path.join(csv_root, f"raw/{filename}"), index=False)
        print(f"{filename} is generated")

    save_dataset(train_dataset, df, f"{config['split']}_{config['target']}_train_dataset.csv")
    save_dataset(valid_dataset, df, f"{config['split']}_{config['target']}_valid_dataset.csv")
    save_dataset(test_dataset, df, f"{config['split']}_{config['target']}_test_dataset.csv")

    return csv_root

def random_split(config):

    df = pd.read_csv(os.path.join(config['data_root'], config['raw_filename']))

    Ys = df[[config['target']]].values
    Xs = np.zeros((len(df), 1))
    ids = df['smiles'].values

    dataset = dc.data.DiskDataset.from_numpy(X=Xs, y=Ys, w=np.ones((len(df), 1)), ids=ids)

    randomsplitter = dc.splits.RandomSplitter()
    train_dataset, valid_dataset, test_dataset = randomsplitter.train_valid_test_split(
        dataset,
        frac_train=config['train_ratio'],
        frac_valid=config['valid_ratio'],
        frac_test=config['test_ratio'],
        seed=config['seed']
    )

    csv_root = f"{config['data_root']}/{config['split']}/{config['target']}"

    if not os.path.exists(os.path.join(csv_root, "raw")):
        os.makedirs(os.path.join(csv_root, "raw"))

    train_smiles = train_dataset.ids
    train_Y = train_dataset.y
    train_df = pd.DataFrame({'smiles': train_smiles, config['target']: train_Y.flatten()})
    train_df.to_csv(os.path.join(csv_root, f"raw/{config['split']}_{config['target']}_train_dataset.csv"), index=False)
    print(f"{config['split']}_{config['target']}_train_dataset.csv is generated")

    valid_smiles = valid_dataset.ids
    valid_Y = valid_dataset.y
    valid_df = pd.DataFrame({'smiles': valid_smiles, config['target']: valid_Y.flatten()})
    valid_df.to_csv(os.path.join(csv_root, f"raw/{config['split']}_{config['target']}_valid_dataset.csv"), index=False)
    print(f"{config['split']}_{config['target']}_valid_dataset.csv is generated")

    test_smiles = test_dataset.ids
    test_Y = test_dataset.y
    test_df = pd.DataFrame({'smiles': test_smiles, config['target']: test_Y.flatten()})
    test_df.to_csv(os.path.join(csv_root, f"raw/{config['split']}v_{config['target']}_test_dataset.csv"), index=False)
    print(f"{config['split']}_{config['target']}_test_dataset.csv is generated")

    return csv_root