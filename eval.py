import os
import sys
import torch
import json
import numpy as np
import torch.nn as nn
from dataset.dataset import VideoDataset
from model.network import C3DVQANet
from scipy.stats import spearmanr, pearsonr
from opts import parse_opts
from tool.draw import mos_scatter 

def test_model(model, device, criterion, dataloaders):

    phase = 'test'
    model.eval()

    epoch_labels = []
    epoch_preds = []

    for ref, dis, labels in dataloaders[phase]:

        ref = ref.to(device)
        dis = dis.to(device)
        labels = labels.to(device).float()

        # dim: [batch=1, P, C, D, H, W]
        ref = ref.reshape(-1, ref.shape[2], ref.shape[3], ref.shape[4], ref.shape[5])
        dis = dis.reshape(-1, dis.shape[2], dis.shape[3], dis.shape[4], dis.shape[5])

        with torch.no_grad():
            preds = model(ref, dis)
            preds = torch.mean(preds, 0, keepdim=True)

        epoch_labels.append(labels.flatten())
        epoch_preds.append(preds.flatten())

    epoch_labels = torch.cat(epoch_labels).flatten().data.cpu().numpy()
    epoch_preds = torch.cat(epoch_preds).flatten().data.cpu().numpy()
    
    ret = {}
    ret['MOS'] = epoch_labels.tolist()
    ret['PRED'] = epoch_preds.tolist()
 
    # print(json.dumps(ret))

    epoch_rmse = np.sqrt(np.mean((epoch_labels - epoch_preds)**2))
    print("{phase} RMSE: {rmse:.4f}".format(phase=phase, rmse=epoch_rmse))

    if len(epoch_labels) > 5:
        epoch_plcc = pearsonr(epoch_labels, epoch_preds)[0]
        epoch_srocc = spearmanr(epoch_labels, epoch_preds)[0]

        print("{phase}:\t PLCC: {plcc:.4f}\t SROCC: {srocc:.4f}".format(phase=phase, plcc=epoch_plcc, srocc=epoch_srocc))


if __name__=='__main__':

    opt = parse_opts()

    video_path = opt.video_dir
    subj_dataset = opt.score_file_path
    load_checkpoint = opt.load_model
    MULTI_GPU_MODE = opt.multi_gpu
    channel = opt.channel
    size_x = opt.size_x
    size_y = opt.size_y
    stride_x = opt.stride_x
    stride_y = opt.stride_y

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, channel, size_x, size_y, stride_x, stride_y) for x in ['test']}
    dataloaders = {x: torch.utils.data.DataLoader(video_dataset[x], batch_size=1, shuffle=False, num_workers=4, drop_last=False) for x in ['test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(load_checkpoint)

    model = C3DVQANet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(model, device_ids=device_ids)
        print("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))
    else:
        print('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))

    criterion = nn.MSELoss()

    test_model(model, device, criterion, dataloaders)
