import numpy as np
import torch
from sklearn.metrics import f1_score
import subprocess
import os
from torch.utils.data import Dataset
from tqdm import tqdm
import time  

class H5Dataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        sample = {'ProteinID' : item.ProteinID, 'hybrid_input' : item.Profile, 'embed_input': item.Repr, 'target' : item.SecondarySeq, 'ProteinLen' : item.ProteinLen} 
        return sample
    
def H5collate_fn(batch_data):
    maxLenth = max([data['ProteinLen'] for data in batch_data])
    hybrid_feats = batch_data[0]['hybrid_input'].shape[1]
    embed_feats = batch_data[0]['embed_input'].shape[1]
    batch_size = len(batch_data) 
    
    batch_id =[]
    masks = torch.zeros(size=(batch_size, maxLenth), dtype=torch.bool)
    hybrid_inputs = torch.zeros(size=(batch_size, maxLenth, hybrid_feats), dtype=torch.float32)
    embed_inputs = torch.zeros(size=(batch_size, maxLenth, embed_feats), dtype=torch.float32)
    targets = torch.zeros(size=(batch_size, maxLenth), dtype=torch.long)
    
    for idx in range(batch_size):
        data = batch_data[idx]
        # pidx = batch_data[idx].ProteinIdx
        lenth = data['ProteinLen']
        masks[idx, :lenth] = True
        hybrid_inputs[idx, :lenth, :] = data['hybrid_input']
        embed_inputs[idx, :lenth, :] = data['embed_input']
        targets[idx, :lenth] = data['target']
        batch_id.append(data['ProteinID'])
    batch_dic = {}
    batch_dic['hybrid_input'] = hybrid_inputs.transpose(1, 2)
    batch_dic['embed_input'] = embed_inputs.transpose(1, 2)
    batch_dic['target'] = targets
    batch_dic['ProteinID'] = batch_id
    batch_dic['mask'] = masks
    return batch_dic

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0.0
    for batch in tqdm(train_loader, total=len(train_loader), desc='[{}th Epoch]'.format(epoch)):
        # input [b, C, L] targets [b, L] masks [b, L]
        hybrid_inputs, embed_inputs, targets, masks = batch['hybrid_input'], batch['embed_input'], batch['target'], batch['mask']
        hybrid_inputs, embed_inputs, targets, masks = hybrid_inputs.to(device), embed_inputs.to(device), targets.to(device), masks.to(device)
        optimizer.zero_grad()
        # forward
        outputs = model(hybrid_inputs, embed_inputs)
        # view 和 reshape 改变张量形状的区别
        outputs = outputs.reshape(-1, outputs.size(2))
        # output (b*L, C)
        loss = criterion(outputs, targets, masks)
        total_loss += loss.item()
        # backward
        loss.backward()
        optimizer.step()
    return total_loss / len(train_loader)



def eval(num_class, model, device, valid_loader):
    model.eval()
    labels_true = np.array([])
    labels_pred = np.array([])
    with torch.no_grad():
        for batch in tqdm(valid_loader, total=len(valid_loader)):
            hybrid_inputs, embed_inputs, targets, masks = batch['hybrid_input'], batch['embed_input'], batch['target'], batch['mask']
            hybrid_inputs, embed_inputs, targets, masks = hybrid_inputs.to(device), embed_inputs.to(device), targets.to(device), masks.to(device)
            outputs = model(hybrid_inputs, embed_inputs)
            pred_labels = torch.argmax(outputs, dim=2)
            
            Lengths = masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L = Lengths[i]
                if L > 0:
                    labels_true = np.hstack((labels_true, targets[i, :L].cpu().numpy()))
                    labels_pred = np.hstack((labels_pred, pred_labels[i, :L].cpu().numpy()))
    class_correct = list(0.0 for _ in range(num_class))
    class_total = list(0.0 for _ in range(num_class))
    for i in range(len(labels_true)):
        label = int(labels_true[i])
        class_total[label] += 1
        if label == labels_pred[i]:
            class_correct[label] += 1
    #    classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy = []
    for i in range(num_class):
        accuracy.append(100.0 * class_correct[i] / (class_total[i] + 1e-12))
    accuracy.append(100.0 * sum(class_correct) / sum(class_total))
    return accuracy


def eval_sov(params, dataset, model, device, test_loader):
    model.eval()
    labels_true = np.array([])
    labels_pred = np.array([])
    Eval_FileName = params['results'] + "/" + dataset + '_' + time.strftime('_%m_%d_%H_%M',time.localtime(time.time()))
    f = open(Eval_FileName + "_Q%d.txt" % (params['num_class']) , "w")
    count = 0
    if params['num_class'] == 8:
        SS_dict = {
            0: "L",
            1: "B",
            2: "E",
            3: "G",
            4: "I",
            5: "H",
            6: "S",
            7: "T",
            8: "X",
        }
    else:
        SS_dict = {0: "C", 1: "E", 2: "H", 3: "X"}
    with torch.no_grad():
        for batch in tqdm(test_loader, desc= dataset, total=len(test_loader)):
            count = 0
            hybrid_inputs, embed_inputs, targets, masks = batch['hybrid_input'], batch['embed_input'], batch['target'], batch['mask']
            hybrid_inputs, embed_inputs, targets, masks = hybrid_inputs.to(device), embed_inputs.to(device), targets.to(device), masks.to(device)
            outputs = model(hybrid_inputs, embed_inputs)
            pred_labels = torch.argmax(outputs, dim=2)
            Lengths = masks.sum(dim=-1).cpu().numpy()
            for i in range(len(Lengths)):
                L = Lengths[i]
                if L > 0:
                    label_t = targets[i, :L].cpu().numpy()
                    label_p = pred_labels[i, :L].cpu().numpy()
                    labels_true = np.hstack((labels_true, targets[i, :L].cpu().numpy()))
                    labels_pred = np.hstack(
                        (labels_pred, pred_labels[i, :L].cpu().numpy())
                    )
                    label_t_ = ""
                    for i in label_t:
                        label_t_ += SS_dict[i]
                    label_p_ = ""
                    for i in label_p:
                        label_p_ += SS_dict[i]
                    f.write(">%s %d\n"% (batch['ProteinID'][count], L))
                    f.write("%s\n" % (label_t_))
                    f.write("%s\n" % (label_p_))
                    count += 1

    f.close()
    commands = "perl SOV.pl " + Eval_FileName + "_Q%d.txt" % (params['num_class'])
    subprocess.call(
        commands, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True
    )
    f = open(Eval_FileName + "_Q%d_Eval.txt" % (params['num_class']), "rt")
    line = f.readline()
    sov_results = line.strip()
    f.close()
    if os.path.exists(Eval_FileName + "_Q%d_Eval.txt" % (params['num_class'])):
        os.remove(Eval_FileName + "_Q%d_Eval.txt" % (params['num_class']))
    idxs = np.where(labels_true != params['num_class'])
    labels_true = labels_true[idxs]
    labels_pred = labels_pred[idxs]
    F1 = f1_score(
        labels_true, labels_pred, average="macro", labels=np.unique(labels_pred)
    )
    class_correct = list(0.0 for i in range(params['num_class']))
    class_total = list(0.0 for i in range(params['num_class']))
    for i in range(len(labels_true)):
        label = int(labels_true[i])
        class_total[label] += 1
        if label == labels_pred[i]:
            class_correct[label] += 1
    #  classes=['L','B','E','G','I','H','S','T'] or ['C','E','H']
    accuracy = []
    for i in range(params['num_class']):
        accuracy.append(100.0 * class_correct[i] / (class_total[i] + 1e-12))
    accuracy.append(100.0 * sum(class_correct) / sum(class_total))
    return F1, accuracy, sov_results


def Compute_Mean_Std(train_list):
    N = 0.0
    d = train_list[0].Profile.shape[1]
    mean = torch.zeros(size=(d,), dtype=torch.float64)
    std = torch.zeros(size=(d,), dtype=torch.float64)
    for idx in range(len(train_list)):
        N += train_list[idx].ProteinLen
        mean += train_list[idx].Profile.sum(dim=0)
        std += train_list[idx].Profile.pow(2).sum(dim=0)
    mean = mean / N
    std = torch.sqrt((std - N * mean.pow(2)) / (N - 1))
    return mean.type(torch.float32).view(1, -1), std.type(torch.float32).view(1, -1)


def Normalized(data_list, mean, std):
    for idx in range(len(data_list)):
        data_list[idx].Profile = (data_list[idx].Profile - mean) / std