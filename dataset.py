import numpy as np
import torch
import torch.nn.functional as F
import h5py
from tqdm import tqdm

np.random.seed(0)

class ProteinNode:
    def __init__(self, ProteinID, ProteinLen, PrimarySeq, SecondarySeq, Profile, Repr):
        self.ProteinID = ProteinID
        self.ProteinLen = ProteinLen
        self.PrimarySeq = PrimarySeq
        self.SecondarySeq = SecondarySeq
        self.Profile = Profile
        self.Repr = Repr

def Load_H5PY_embedding(FilePath, isEightClass, dataset=None):
    SS = ['L', 'B', 'E', 'G', 'I', 'H', 'S', 'T', 'X']
    SS8_Dict = dict(zip(SS, range(len(SS))))
    SS3_Dict = {'L': 0, 'B': 1, 'E': 1, 'G': 2, 'I': 2, 'H': 2, 'S': 0, 'T': 0, 'X': 3}
    Standard_AAS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
    AA_Dict = {}
    Non_Standard_AAS = list('BJOUXZ')
    for i, AA in enumerate(Standard_AAS):
        AA_Dict.update({AA: i})
    for i, AA in enumerate(Non_Standard_AAS):
        AA_Dict.update({AA: 20})
    
    phys_dict = {0: [-0.350, -0.680, -0.677, -0.171, -0.170, 0.900, -0.476],
                 1: [-0.140, -0.329, -0.359, 0.508, -0.114, -0.652, 0.476],
                 2: [-0.213, -0.417, -0.281, -0.767, -0.900, -0.155, -0.635],
                 3: [-0.230, -0.241, -0.058, -0.696, -0.868, 0.900, -0.582],
                 4: [0.363, 0.373, 0.412, 0.646, -0.272, 0.155, 0.318],
                 5: [-0.900, -0.900, -0.900, -0.342, -0.179, -0.900, -0.900],
                 6: [0.384, 0.110, 0.138, -0.271, 0.195, -0.031, -0.106],
                 7: [0.900, -0.066, -0.009, 0.652, -0.186, 0.155, 0.688],
                 8: [-0.088, 0.066, 0.163, -0.889, 0.727, 0.279, -0.265],
                 9: [0.213, -0.066, -0.009, 0.596, -0.186, 0.714, -0.053],
                 10: [0.110, 0.066, 0.087, 0.337, -0.262, 0.652, -0.001],
                 11: [-0.213, -0.329, -0.243, -0.674, -0.075, -0.403, -0.529],
                 12: [0.247, -0.900, -0.294, 0.055, -0.010, -0.900, 0.106],
                 13: [-0.230, -0.110, -0.020, -0.464, -0.276, 0.528, -0.371],
                 14: [0.105, 0.373, 0.466, -0.900, 0.900, 0.528, -0.371],
                 15: [-0.337, -0.637, -0.544, -0.364, -0.265, -0.466, -0.212],
                 16: [0.402, -0.417, -0.321, -0.199, -0.288, -0.403, 0.212],
                 17: [0.677, -0.285, -0.232, 0.331, -0.191, -0.031, 0.900],
                 18: [0.479, 0.900, 0.900, 0.900, -0.209, 0.279, 0.529],
                 19: [0.363, 0.417, 0.541, 0.188, -0.274, -0.155, 0.476],
                 20: [0.0771, -0.1536, -0.0620, -0.0762, -0.1451, 0.0497, -0.0398]}
    
    f = h5py.File(FilePath + '.h5', 'r')
    if dataset == 'CB513':
        idxs = f['CB513_filtered_idxs'][()]
    elif dataset == 'CASP12':
        idxs = f['CASP12_filtered_idxs'][()]
    elif dataset == 'CASP13':
        idxs = f['CASP13_filtered_idxs'][()]
    elif dataset == 'CASP14':
        idxs = f['CASP14_filtered_idxs'][()]
    else:
        NumOfSamples = len(f.keys()) // 5
        idxs = range(NumOfSamples)

    Data = []
    for i in tqdm(idxs, desc='data_loading'):
        ProteinID = f['ID' + str(i)][()]
        ProteinID = ProteinID.decode()
        PrimarySeq = f['PS' + str(i)][()]
        PrimarySeq = PrimarySeq.decode()
        ProteinLen = len(PrimarySeq)
        PrimarySeq = [AA_Dict[e] for e in PrimarySeq]
        SecondarySeq = f['SS' + str(i)][()]
        SecondarySeq = SecondarySeq.decode()
        if isEightClass:
            SecondarySeq = [SS8_Dict[e] for e in SecondarySeq]
        else:
            SecondarySeq = [SS3_Dict[e] for e in SecondarySeq] 
        PrimarySeq = torch.tensor(PrimarySeq, dtype=torch.long)
        SecondarySeq = torch.tensor(SecondarySeq, dtype=torch.long)
        Profile = torch.from_numpy(f['Profile' + str(i)][()])
        phys_features = torch.zeros(size=(ProteinLen, 7), dtype=torch.float)
        for k in range(ProteinLen):
            phys_features[k, :] = torch.tensor(phys_dict[PrimarySeq[k].item()])
        
        Repr = torch.from_numpy(f['repr' + str(i)][()])
        if dataset or FilePath.split('/')[-1] in ['CASP12', 'CASP13', 'CASP14','CB513'] :    
            Profile = torch.cat([Profile, phys_features], dim=1)  
        Node = ProteinNode(ProteinID, ProteinLen, PrimarySeq, SecondarySeq, Profile, Repr)
        Data.append(Node)
    f.close()
    return Data


def CASP12(isEightClass):
    Data = Load_H5PY_embedding('data_1280/PISCES_Data', isEightClass, dataset='CASP12')
    train_list = Data[:-512]
    valid_list = Data[-512:]
    valid_list.sort(key=lambda Node: Node.ProteinLen, reverse=False)
    test_list = Load_H5PY_embedding('data_1280/CASP12', isEightClass)
    return train_list, valid_list, test_list


def CASP13(isEightClass):
    Data = Load_H5PY_embedding('data_1280/PISCES_Data', isEightClass, dataset='CASP13')
    train_list = Data[:-512]
    valid_list = Data[-512:]
    valid_list.sort(key=lambda Node: Node.ProteinLen, reverse=False)
    test_list = Load_H5PY_embedding('data_1280/CASP13', isEightClass)
    return train_list, valid_list, test_list


def CASP14(isEightClass):
    Data = Load_H5PY_embedding('data_1280/PISCES_Data', isEightClass, dataset='CASP14')
    train_list = Data[:-512]
    valid_list = Data[-512:]
    valid_list.sort(key=lambda Node: Node.ProteinLen, reverse=False)
    test_list = Load_H5PY_embedding('data_1280/CASP14', isEightClass)
    return train_list, valid_list, test_list

def CB513(isEightClass):
    Data = Load_H5PY_embedding('data_1280/PISCES_Data', isEightClass, dataset='CB513')
    train_list = Data[:-512]
    valid_list = Data[-512:]
    valid_list.sort(key=lambda Node: Node.ProteinLen, reverse=False)
    test_list = Load_H5PY_embedding('data_1280/CB513', isEightClass)
    return train_list, valid_list, test_list

def Test2016(isEightClass):
    train_list = Load_H5PY_embedding('data_1280/SPOT_1D_Train', isEightClass)
    valid_list = Load_H5PY_embedding('data_1280/SPOT_1D_Valid', isEightClass)
    test2016_list = Load_H5PY_embedding('data_1280/Test2016', isEightClass)
    return train_list, valid_list, test2016_list

def Test2018(isEightClass):
    train_list = Load_H5PY_embedding('data_1280/SPOT_1D_Train', isEightClass)
    valid_list = Load_H5PY_embedding('data_1280/SPOT_1D_Valid',isEightClass)
    test2018_list = Load_H5PY_embedding('data_1280/Test2018', isEightClass)
    return train_list, valid_list, test2018_list

def Load_embedding_DataSet(DataSet, isEightClass):
    if DataSet == 'CB513':
        return CB513(isEightClass)
    elif DataSet == 'CASP12':
        return CASP12(isEightClass)
    elif DataSet == 'CASP13':
        return CASP13(isEightClass)
    elif DataSet == 'CASP14':
        return CASP14(isEightClass)
    elif DataSet == 'Test2016':
        return Test2016(isEightClass)
    elif DataSet == 'Test2018':
        return Test2018(isEightClass)
    else:
        return CB513(isEightClass)
