import os
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from torch_geometric.data import Batch
import pandas as pd
import numpy as np
import networkx as nx
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import BRICS
from rdkit.Chem.rdMolDescriptors import GetMorganFingerprintAsBitVect
from itertools import repeat, product, chain
from HiMol.chemutils import get_mol, get_clique_mol


# allowable node and edge features
allowable_features = {
    'possible_atomic_num_list' : list(range(1, 119)),
    'possible_formal_charge_list' : [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5],
    'possible_chirality_list' : [
        Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
        Chem.rdchem.ChiralType.CHI_OTHER
    ],
    'possible_hybridization_list' : [
        Chem.rdchem.HybridizationType.S,
        Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
        Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
        Chem.rdchem.HybridizationType.SP3D2, Chem.rdchem.HybridizationType.UNSPECIFIED
    ],
    'possible_numH_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8],
    'possible_implicit_valence_list' : [0, 1, 2, 3, 4, 5, 6],
    'possible_degree_list' : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'possible_bonds' : [
        Chem.rdchem.BondType.SINGLE,
        Chem.rdchem.BondType.DOUBLE,
        Chem.rdchem.BondType.TRIPLE,
        Chem.rdchem.BondType.AROMATIC
    ],
    'possible_bond_dirs' : [ # only for double bond stereo information
        Chem.rdchem.BondDir.NONE,
        Chem.rdchem.BondDir.ENDUPRIGHT,
        Chem.rdchem.BondDir.ENDDOWNRIGHT
    ],
    'possible_bond_inring': [None, False, True]
}


def mol_to_graph_data_obj_simple(mol):
    """
    Converts rdkit mol object to graph Data object required by the pytorch
    geometric package. NB: Uses simplified atom and bond features, and represent
    as indices
    :param mol: rdkit mol object
    :return: graph data object with the attributes: x, edge_index, edge_attr
    """
    # atoms
    num_atom_features = 2   # atom type,  chirality tag
    atom_features_list = []
    for atom in mol.GetAtoms():

        atom_feature = [allowable_features['possible_atomic_num_list'].index(
                atom.GetAtomicNum())] + [allowable_features[
                                         'possible_degree_list'].index(atom.GetDegree())]


        atom_features_list.append(atom_feature)
    x_nosuper = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2   # bond type, bond direction
    if len(mol.GetBonds()) > 0: # mol has bonds
        edges_list = []
        edge_features_list = []
        for bond in mol.GetBonds():
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()

            edge_feature = [allowable_features['possible_bonds'].index(
                 bond.GetBondType())] + [allowable_features['possible_bond_inring'].index(
                 bond.IsInRing())]

            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index_nosuper = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr_nosuper = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index_nosuper = torch.empty((2, 0), dtype=torch.long)
        edge_attr_nosuper = torch.empty((0, num_bond_features), dtype=torch.long)

    num_atoms = x_nosuper.size(0)
    super_x = torch.tensor([[119, 0]]).to(x_nosuper.device)
    #add motif 
    cliques = motif_decomp(mol)
    num_motif = len(cliques)
    if num_motif > 0:
        motif_x = torch.tensor([[120, 0]]).repeat_interleave(num_motif, dim=0).to(x_nosuper.device)
        x = torch.cat((x_nosuper, motif_x, super_x), dim=0)

        motif_edge_index = []
        for k, motif in enumerate(cliques):
            motif_edge_index = motif_edge_index + [[i, num_atoms+k] for i in motif]
        motif_edge_index = torch.tensor(np.array(motif_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)

        super_edge_index = [[num_atoms+i, num_atoms+num_motif] for i in range(num_motif)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, motif_edge_index, super_edge_index), dim=1)

        motif_edge_attr = torch.zeros(motif_edge_index.size()[1], 2)
        motif_edge_attr[:,0] = 6 
        motif_edge_attr = motif_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)

        super_edge_attr = torch.zeros(num_motif, 2)
        super_edge_attr[:,0] = 5 
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, motif_edge_attr, super_edge_attr), dim = 0)

        num_part = (num_atoms, num_motif, 1)

    else:
        x = torch.cat((x_nosuper, super_x), dim=0)

        super_edge_index = [[i, num_atoms] for i in range(num_atoms)]
        super_edge_index = torch.tensor(np.array(super_edge_index).T, dtype=torch.long).to(edge_index_nosuper.device)
        edge_index = torch.cat((edge_index_nosuper, super_edge_index), dim=1)

        super_edge_attr = torch.zeros(num_atoms, 2)
        super_edge_attr[:,0] = 5 #bond type for self-loop edge
        super_edge_attr = super_edge_attr.to(edge_attr_nosuper.dtype).to(edge_attr_nosuper.device)
        edge_attr = torch.cat((edge_attr_nosuper, super_edge_attr), dim = 0)

        num_part = (num_atoms, 0, 1)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data



def motif_decomp(mol):
    n_atoms = mol.GetNumAtoms()
    if n_atoms == 1:
        return [[0]]

    cliques = []  
    breaks = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtom().GetIdx()
        a2 = bond.GetEndAtom().GetIdx()
        cliques.append([a1, a2])  

    res = list(BRICS.FindBRICSBonds(mol))  
    if len(res) != 0:
        for bond in res:
            if [bond[0][0], bond[0][1]] in cliques:
                cliques.remove([bond[0][0], bond[0][1]])
            else:
                cliques.remove([bond[0][1], bond[0][0]])
            cliques.append([bond[0][0]])
            cliques.append([bond[0][1]]) 

    for c in range(len(cliques) - 1):
        if c >= len(cliques):
            break
        for k in range(c + 1, len(cliques)):
            if k >= len(cliques):
                break
            if len(set(cliques[c]) & set(cliques[k])) > 0: 
                cliques[c] = list(set(cliques[c]) | set(cliques[k]))
                cliques[k] = []
        cliques = [c for c in cliques if len(c) > 0]
    cliques = [c for c in cliques if n_atoms> len(c) > 0]


    num_cli = len(cliques)
    ssr_mol = Chem.GetSymmSSSR(mol)
    for i in range(num_cli):
        c = cliques[i]
        cmol = get_clique_mol(mol, c)
        ssr = Chem.GetSymmSSSR(cmol)
        if len(ssr)>1: 
            for ring in ssr_mol:
                if len(set(list(ring)) & set(c)) == len(list(ring)):
                    cliques.append(list(ring))
            cliques[i]=[]
    
    cliques = [c for c in cliques if n_atoms> len(c) > 0]
    return cliques



def graph_data_obj_to_mol_simple(data_x, data_edge_index, data_edge_attr):
    """
    Convert pytorch geometric data obj to rdkit mol object. NB: Uses simplified
    atom and bond features, and represent as indices.
    :param: data_x:
    :param: data_edge_index:
    :param: data_edge_attr
    :return:
    """
    mol = Chem.RWMol()

    # atoms
    atom_features = data_x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        atomic_num = allowable_features['possible_atomic_num_list'][atomic_num_idx]
        chirality_tag = allowable_features['possible_chirality_list'][chirality_tag_idx]
        atom = Chem.Atom(atomic_num)
        atom.SetChiralTag(chirality_tag)
        mol.AddAtom(atom)

    # bonds
    edge_index = data_edge_index.cpu().numpy()
    edge_attr = data_edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        bond_type = allowable_features['possible_bonds'][bond_type_idx]
        bond_dir = allowable_features['possible_bond_dirs'][bond_dir_idx]
        mol.AddBond(begin_idx, end_idx, bond_type)
        # set bond direction
        new_bond = mol.GetBondBetweenAtoms(begin_idx, end_idx)
        new_bond.SetBondDir(bond_dir)

    return mol

def graph_data_obj_to_nx_simple(data):
    """
    Converts graph Data object required by the pytorch geometric package to
    network x data object. NB: Uses simplified atom and bond features,
    and represent as indices. NB: possible issues with recapitulating relative
    stereochemistry since the edges in the nx object are unordered.
    :param data: pytorch geometric Data object
    :return: network x object
    """
    G = nx.Graph()

    # atoms
    atom_features = data.x.cpu().numpy()
    num_atoms = atom_features.shape[0]
    for i in range(num_atoms):
        atomic_num_idx, chirality_tag_idx = atom_features[i]
        G.add_node(i, atom_num_idx=atomic_num_idx, chirality_tag_idx=chirality_tag_idx)
        pass

    # bonds
    edge_index = data.edge_index.cpu().numpy()
    edge_attr = data.edge_attr.cpu().numpy()
    num_bonds = edge_index.shape[1]
    for j in range(0, num_bonds, 2):
        begin_idx = int(edge_index[0, j])
        end_idx = int(edge_index[1, j])
        bond_type_idx, bond_dir_idx = edge_attr[j]
        if not G.has_edge(begin_idx, end_idx):
            G.add_edge(begin_idx, end_idx, bond_type_idx=bond_type_idx,
                       bond_dir_idx=bond_dir_idx)

    return G

def nx_to_graph_data_obj_simple(G):
    """
    Converts nx graph to pytorch geometric Data object. Assume node indices
    are numbered from 0 to num_nodes - 1. NB: Uses simplified atom and bond
    features, and represent as indices. NB: possible issues with
    recapitulating relative stereochemistry since the edges in the nx
    object are unordered.
    :param G: nx graph obj
    :return: pytorch geometric Data object
    """
    # atoms
    num_atom_features = 2  # atom type,  chirality tag
    atom_features_list = []
    for _, node in G.nodes(data=True):
        atom_feature = [node['atom_num_idx'], node['chirality_tag_idx']]
        atom_features_list.append(atom_feature)
    x = torch.tensor(np.array(atom_features_list), dtype=torch.long)

    # bonds
    num_bond_features = 2  # bond type, bond direction
    if len(G.edges()) > 0:  # mol has bonds
        edges_list = []
        edge_features_list = []
        for i, j, edge in G.edges(data=True):
            edge_feature = [edge['bond_type_idx'], edge['bond_dir_idx']]
            edges_list.append((i, j))
            edge_features_list.append(edge_feature)
            edges_list.append((j, i))
            edge_features_list.append(edge_feature)

        # data.edge_index: Graph connectivity in COO format with shape [2, num_edges]
        edge_index = torch.tensor(np.array(edges_list).T, dtype=torch.long)

        # data.edge_attr: Edge feature matrix with shape [num_edges, num_edge_features]
        edge_attr = torch.tensor(np.array(edge_features_list),
                                 dtype=torch.long)
    else:   # mol has no bonds
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, num_bond_features), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    return data

def get_gasteiger_partial_charges(mol, n_iter=12):
    """
    Calculates list of gasteiger partial charges for each atom in mol object.
    :param mol: rdkit mol object
    :param n_iter: number of iterations. Default 12
    :return: list of computed partial charges for each atom.
    """
    Chem.rdPartialCharges.ComputeGasteigerCharges(mol, nIter=n_iter,
                                                  throwOnParamFailure=True)
    partial_charges = [float(a.GetProp('_GasteigerCharge')) for a in
                       mol.GetAtoms()]
    return partial_charges

def create_standardized_mol_id(smiles):
    """

    :param smiles:
    :return: inchi
    """
    if check_smiles_validity(smiles):
        # remove stereochemistry
        smiles = AllChem.MolToSmiles(AllChem.MolFromSmiles(smiles),
                                     isomericSmiles=False)
        mol = AllChem.MolFromSmiles(smiles)
        if mol != None: # to catch weird issue with O=C1O[al]2oc(=O)c3ccc(cn3)c3ccccc3c3cccc(c3)c3ccccc3c3cc(C(F)(F)F)c(cc3o2)-c2ccccc2-c2cccc(c2)-c2ccccc2-c2cccnc21
            if '.' in smiles: # if multiple species, pick largest molecule
                mol_species_list = split_rdkit_mol_obj(mol)
                largest_mol = get_largest_mol(mol_species_list)
                inchi = AllChem.MolToInchi(largest_mol)
            else:
                inchi = AllChem.MolToInchi(mol)
            return inchi
        else:
            return
    else:
        return

class MoleculeDataset(InMemoryDataset):
    def __init__(self,
                 config,
                 split,
                 root,
                 #data = None,
                 #slices = None,
                 transform=None,
                 pre_transform=None,
                 pre_filter=None,
                 dataset=None,
                 empty=False):

        self.config = config
        self.split = split
        if config['inference_mode'] == True:
            self.filename = f"inference_dataset.csv"
        else:
            self.filename = f"{config['split']}_{config['target']}_{split}_dataset.csv"
        self.dataset = dataset
        self.target = config['target']
        self.root = root
        # self.config = config

        super(MoleculeDataset, self).__init__(root, transform, pre_transform,
                                                 pre_filter)
        self.transform, self.pre_transform, self.pre_filter = transform, pre_transform, pre_filter

        if not empty:
            self.data, self.slices = torch.load(os.path.join(self.processed_paths[0]))


    def get(self, idx):
        data = Data()
        for key in self.data.keys():
            item, slices = self.data[key], self.slices[key]
            s = list(repeat(slice(None), item.dim()))
            s[data.__cat_dim__(key, item)] = slice(slices[idx],
                                                    slices[idx + 1])
            data[key] = item[s]
        return data

    @property
    def raw_file_names(self):
        file_name_list = os.listdir(self.raw_dir)
        # assert len(file_name_list) == 1     # currently assume we have a
        # # single raw file
        return file_name_list

    @property
    def processed_file_names(self):
        return self.split + '_geometric_data_processed.pt'

    def download(self):
        raise NotImplementedError('Must indicate valid location of raw data. '
                                  'No download allowed')

    def _load_MLM_HLM_dataset(self, filename):
        """
        :param input_path:
        :return: list of smiles, list of rdkit mol obj, np.array containing the
        labels (regression task)
        """
        input_df = pd.read_csv(os.path.join(self.root, 'raw', filename), sep=',')
        smiles_list = input_df['smiles']
        rdkit_mol_objs_list = [AllChem.MolFromSmiles(s) for s in smiles_list]
        if self.config['inference_mode'] == False:
            labels = input_df[self.target]
            assert len(smiles_list) == len(rdkit_mol_objs_list)
            assert len(smiles_list) == len(labels)
            return smiles_list, rdkit_mol_objs_list, labels.values
        else:
            assert len(smiles_list) == len(rdkit_mol_objs_list)
            return smiles_list, rdkit_mol_objs_list
    
    def process(self):
        data_smiles_list = []
        data_list = []

        if self.dataset == 'MLM' or 'HLM':
            if self.config['inference_mode'] == True:
                # 인퍼런스 모드일 때 labels 없이 데이터 로드
                smiles_list, rdkit_mol_objs = self._load_MLM_HLM_dataset(self.filename)
            else:
                # 그 외 모드에서는 labels 포함하여 데이터 로드
                smiles_list, rdkit_mol_objs, labels = self._load_MLM_HLM_dataset(self.filename)

            for i in range(len(smiles_list)):
                rdkit_mol = rdkit_mol_objs[i]
                data = mol_to_graph_data_obj_simple(rdkit_mol)
                data.id = torch.tensor([i])  # mol의 인덱스

                if self.config['inference_mode'] == False:
                    # 인퍼런스 모드가 아닐 때만 레이블 설정
                    data.y = torch.tensor([labels[i]])

                data_list.append(data)
                data_smiles_list.append(smiles_list[i])
        else:
            raise ValueError('Invalid dataset name')
    
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        # write data_smiles_list in processed paths
        data_smiles_series = pd.Series(data_smiles_list)
        data_smiles_series.to_csv(os.path.join(self.processed_dir,
                                               f"{self.split}_smiles.csv"), index=False,
                                  header=False)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def check_smiles_validity(smiles):
    try:
        m = Chem.MolFromSmiles(smiles)
        if m:
            return True
        else:
            return False
    except:
        return False

def split_rdkit_mol_obj(mol):
    """
    Split rdkit mol object containing multiple species or one species into a
    list of mol objects or a list containing a single object respectively
    :param mol:
    :return:
    """
    smiles = AllChem.MolToSmiles(mol, isomericSmiles=True)
    smiles_list = smiles.split('.')
    mol_species_list = []
    for s in smiles_list:
        if check_smiles_validity(s):
            mol_species_list.append(AllChem.MolFromSmiles(s))
    return mol_species_list

def get_largest_mol(mol_list):
    """
    Given a list of rdkit mol objects, returns mol object containing the
    largest num of atoms. If multiple containing largest num of atoms,
    picks the first one
    :param mol_list:
    :return:
    """
    num_atoms_list = [len(m.GetAtoms()) for m in mol_list]
    largest_mol_idx = num_atoms_list.index(max(num_atoms_list))
    return mol_list[largest_mol_idx]

def create_all_datasets():
    #### create dataset
    downstream_dir = [
            'MLM',
            'HLM'
            ]

    for dataset_name in downstream_dir:
        print(dataset_name)
        root = "dataset/" + dataset_name
        os.makedirs(root + "/processed", exist_ok=True)
        dataset = MoleculeDataset(root, dataset=dataset_name)
        print(dataset)
