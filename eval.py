import torch
from net.pssp_predict import SS_predict
from dataset import *
import os
import utils 
import time  
import numpy as np
from loss import LDAMLoss
from torch.utils.data import DataLoader
 
maxEpochs = 100
m = 1.0
models_dir = 'best_models'
model_params = {'batch_size': 32, 'num_class': 3, 'results' : 'best_results', 'depth_4' :4} 
datasets = ['CASP13']                                                                                  

torch.manual_seed(0)
np.random.seed(0)
use_cuda = torch.cuda.is_available() 
device = torch.device("cuda:0" if use_cuda else "cpu")

if (os.path.isdir(models_dir) == False): 
    os.mkdir(models_dir)
if (os.path.isdir(model_params['results']) == False):
    os.mkdir(model_params['results'])

print("Starting eval...")

if model_params['num_class'] == 8:
    isEightClass = True
else:
    isEightClass = False
cls_num_list = np.zeros(shape=model_params['num_class'], dtype=np.compat.long)

for idx, dataset in enumerate(datasets):
    train_list, valid_list, test_list = Load_embedding_DataSet(dataset, isEightClass)
    mean, std = utils.Compute_Mean_Std(train_list)
    utils.Normalized(train_list, mean, std)
    utils.Normalized(valid_list, mean, std)
    utils.Normalized(test_list, mean, std)
    
    hybrid_features = train_list[0].Profile.shape[1]
    embed_features = train_list[0].Repr.shape[1]
    model = SS_predict(hybrid_features, embed_features, model_params['num_class'], model_params['depth_4']).to(device)

    for i in range(len(train_list)):
        for j in range(train_list[i].ProteinLen):
            cls_num_list[train_list[i].SecondarySeq[j]] += 1

    criterion = LDAMLoss(cls_num_list, max_m=m, weight=None).to(device)
    
    test_set = utils.H5Dataset(test_list)
    test_loader = DataLoader(test_set, batch_size=model_params['batch_size'], collate_fn=utils.H5collate_fn)

    checkpoint = torch.load('%s/%s_%d_%d_c.pth' % (models_dir, dataset, model_params['num_class'], model_params['depth_4']))
    model.load_state_dict(checkpoint['model'])
    criterion.load_state_dict(checkpoint['criterion'])

    F1, accuracy, sov_results = utils.eval_sov(model_params, dataset, model, device, test_loader)

    LogFileName = model_params['results'] + '/' + 'best' + '_' + dataset + '_' + str(model_params['num_class']) + '_' + time.strftime('_%m_%d_%H_%M',time.localtime(time.time()))
    f = open(LogFileName + '.txt', 'w')
    f.write('Dataset:%s \n' % (dataset))
    if model_params['num_class'] == 8:
        f.write('   Q8:%.2f %%, F1:%.2f %% on the dataset %s:\n' % (accuracy[-1], 100 * F1, dataset))      
        f.write('   Q8: L: %.2f, B: %.2f, E: %.2f, G: %.2f, I: %.2f, H: %.2f, S: %.2f, T: %.2f'%(accuracy[0],accuracy[1],accuracy[2],accuracy[3],accuracy[4],accuracy[5],accuracy[6],accuracy[7]))
        
        f.write('   \n  8-state SOV results:\n')
        f.write('   %s\n\n' % (sov_results))
    else:
        f.write('   Q3:%.2f %%, F1:%.2f %% on the dataset %s:\n' % (accuracy[-1], 100 * F1, dataset))   
        f.write('   Q3: C: %.2f, E: %.2f, H: %.2f' % (accuracy[0], accuracy[1], accuracy[2]))
        
        f.write('   \n  3-state SOV results:\n')
        f.write('   %s\n\n' % (sov_results))
f.close()

