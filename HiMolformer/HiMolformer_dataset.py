from torch_geometric.loader import DataLoader as GeoDataLoader
import numpy as np
import pytorch_lightning as pl
from HiMol.HiMol_dataset import MoleculeDataset
from Molformer.Molformer_dataset import PropertyPredictionDataModule
from torch.utils.data import Sampler, DataLoader

class SynchronizedLoader:
    def __init__(self, primary_loader, secondary_loader):
        self.primary_loader = primary_loader
        self.secondary_loader = secondary_loader
        self.length = min(len(primary_loader), len(secondary_loader))
        self.reset_iterators()
        
    def reset_iterators(self):
        self.primary_iterator = iter(self.primary_loader)
        self.secondary_iterator = iter(self.secondary_loader)
        self.length = min(len(self.primary_loader), len(self.secondary_loader))

    def __iter__(self):
        self.reset_iterators()
        return self

    def __next__(self):
        if self.length <= 0:
            raise StopIteration
        self.length -= 1
        return next(self.primary_iterator), next(self.secondary_iterator)


class SynchronizedSampler(Sampler):
    def __init__(self, data_source, seed):
        self.data_source = data_source
        self.seed = seed

    def __iter__(self):
        np.random.seed(self.seed)
        indices = np.random.permutation(len(self.data_source))
        return iter(indices.tolist())
    def __len__(self):
        return len(self.data_source)

class CombinedDataModule(pl.LightningDataModule):
    def __init__(self, config, splitted_data):
        super().__init__()
        self.config = config
        if self.config['inference_mode'] == True: 
            self.Molformer_data_module = PropertyPredictionDataModule(config, config['inference_data_path'])
            self.Molformer_data_module.prepare_inference_data()
            self.HiMol_inference_dataset = MoleculeDataset(config=config, root=config['inference_data_path'], split='inference')
        else:  
            self.Molformer_data_module = PropertyPredictionDataModule(config, splitted_data)
            self.Molformer_data_module.prepare_data()
            self.HiMol_train_dataset = MoleculeDataset(config=config, root=splitted_data, split='train', dataset=config['target'])
            self.HiMol_valid_dataset = MoleculeDataset(config=config, root=splitted_data, split='valid', dataset=config['target'])
            self.HiMol_test_dataset = MoleculeDataset(config=config, root=splitted_data, split='test', dataset=config['target'])

    def train_dataloader(self):
        if self.config['inference_mode'] == False:
            synchronized_sampler = SynchronizedSampler(self.HiMol_train_dataset, seed=self.config['seed'])
            HiMol_train_loader = GeoDataLoader(
                self.HiMol_train_dataset, 
                batch_size=self.config['batch_size'], 
                sampler=synchronized_sampler, 
                num_workers=self.config['num_workers']
            )
            Molformer_train_loader = DataLoader(
                self.Molformer_data_module.train_ds,
                batch_size=self.config['batch_size'],
                sampler=synchronized_sampler,
                num_workers=self.config['num_workers'],
                collate_fn=self.Molformer_data_module.collate
            )
            synchronized_train_loader = SynchronizedLoader(HiMol_train_loader, Molformer_train_loader)
            return synchronized_train_loader
        else:
            return None
    
    def val_dataloader(self):
        if self.config['inference_mode'] == False:
            HiMol_valid_loader = GeoDataLoader(
                self.HiMol_valid_dataset, 
                batch_size=self.config['batch_size']//2, 
                shuffle=False, 
                num_workers=self.config['num_workers']
            )
            Molformer_valid_loader = DataLoader(
                self.Molformer_data_module.val_ds,
                batch_size=self.config['batch_size']//2,
                shuffle=False,
                num_workers=self.config['num_workers'],
                collate_fn=self.Molformer_data_module.collate
            )
            synchronized_valid_loader = SynchronizedLoader(HiMol_valid_loader, Molformer_valid_loader)
            return synchronized_valid_loader
        else:
            return None
    
    def test_dataloader(self):
        if self.config['inference_mode'] == False:
            HiMol_test_loader = GeoDataLoader(
                self.HiMol_test_dataset, 
                batch_size=self.config['batch_size']//2, 
                shuffle=False, 
                num_workers=self.config['num_workers']
            )
            Molformer_test_loader = DataLoader(
                self.Molformer_data_module.test_ds,
                batch_size=self.config['batch_size']//2,
                shuffle=False,
                num_workers=self.config['num_workers'],
                collate_fn=self.Molformer_data_module.collate
            )
            synchronized_test_loader = SynchronizedLoader(HiMol_test_loader, Molformer_test_loader)
            return synchronized_test_loader
        else:
            return None

    def inference_dataloader(self):
        if self.config['inference_mode'] == True:
            HiMol_inference_loader = GeoDataLoader(
                self.HiMol_inference_dataset, 
                batch_size=self.config['batch_size']//2, 
                shuffle=False, 
                num_workers=self.config['num_workers']
            )
            Molformer_inference_loader = DataLoader(
                self.Molformer_data_module.inference_ds,
                batch_size=self.config['batch_size']//2,
                shuffle=False,
                num_workers=self.config['num_workers'],
                collate_fn=self.Molformer_data_module.collate
            )
            synchronized_inference_loader = SynchronizedLoader(HiMol_inference_loader, Molformer_inference_loader)
            return synchronized_inference_loader
        else:
            return None