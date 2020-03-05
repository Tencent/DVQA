import os
import sys
import copy
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr
from opts import parse_opts
from model.network import C3DVQANet
from dataset.dataset import VideoDataset


def train_model(model, device, criterion, optimizer, scheduler, dataloaders, save_checkpoint, epoch_resume=1, num_epochs=25):

    for epoch in tqdm(range(epoch_resume, num_epochs+epoch_resume), unit='epoch', initial=epoch_resume, total=num_epochs+epoch_resume):
        for phase in ['train', 'test']:
            epoch_labels = []
            epoch_preds = []
            epoch_loss = 0.0
            epoch_size = 0

            if phase == 'train':
                model.train()
            else:
                model.eval()

            for ref, dis, labels in dataloaders[phase]:
                ref = ref.to(device)
                dis = dis.to(device)
                labels = labels.to(device).float()

                ref = ref.reshape(-1, ref.shape[2], ref.shape[3], ref.shape[4], ref.shape[5])
                dis = dis.reshape(-1, dis.shape[2], dis.shape[3], dis.shape[4], dis.shape[5])

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    preds = model(ref, dis)
                    preds = torch.mean(preds, 0, keepdim=True)
                    loss = criterion(preds, labels)

                    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
                        loss = torch.mean(loss)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * labels.size(0)
                epoch_size += labels.size(0)
                epoch_labels.append(labels.flatten())
                epoch_preds.append(preds.flatten())

            epoch_loss = epoch_loss / epoch_size

            if phase == 'train':
                scheduler.step(epoch_loss)

            epoch_labels = torch.cat(epoch_labels).flatten().data.cpu().numpy()
            epoch_preds = torch.cat(epoch_preds).flatten().data.cpu().numpy()

            epoch_plcc = pearsonr(epoch_labels, epoch_preds)[0]
            epoch_srocc = spearmanr(epoch_labels, epoch_preds)[0]
            epoch_rmse = np.sqrt(np.mean((epoch_labels - epoch_preds)**2))

            print("{phase}-Loss: {loss:.4f}\t RMSE: {rmse:.4f}\t PLCC: {plcc:.4f}\t SROCC: {srocc:.4f}".format(phase=phase, loss=epoch_loss, rmse=epoch_rmse, plcc=epoch_plcc, srocc=epoch_srocc))

            if phase == 'test' and save_checkpoint and epoch % 10 == 0:
                _checkpoint = '{pt}_{epoch}'.format(pt=save_checkpoint, epoch=epoch)
                torch.save({'epoch': epoch, 'model_state_dict': model.module.state_dict(), 'optimizer_state_dict': optimizer.state_dict()}, _checkpoint)


if __name__=='__main__':

    opt = parse_opts()

    video_path = opt.video_dir
    subj_dataset = opt.score_file_path
    save_checkpoint = opt.save_model
    load_checkpoint = opt.load_model
    LEARNING_RATE = opt.learning_rate
    L2_REGULARIZATION = opt.weight_decay
    NUM_EPOCHS = opt.epochs
    MULTI_GPU_MODE = opt.multi_gpu
    size_x = opt.size_x
    size_y = opt.size_y
    stride_x = opt.stride_x
    stride_y = opt.stride_y

    video_dataset = {x: VideoDataset(subj_dataset, video_path, x, size_x, size_y, stride_x, stride_y) for x in ['train', 'test']}
    dataloaders = {x: torch.utils.data.DataLoader(video_dataset[x], batch_size=1, shuffle=True, num_workers=8, drop_last=True) for x in ['train', 'test']}

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
        device_ids = range(0, torch.cuda.device_count())
        model = torch.nn.DataParallel(C3DVQANet().to(device), device_ids=device_ids)
        print("muti-gpu mode enabled, use {0:d} gpus".format(torch.cuda.device_count()))
    else:
        model = C3DVQANet().to(device)
        print('use {0}'.format('cuda' if torch.cuda.is_available() else 'cpu'))

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=L2_REGULARIZATION)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.9, patience=5)
    epoch_resume = 1

    if os.path.exists(load_checkpoint):
        checkpoint = torch.load(load_checkpoint)
        print("loading checkpoint")

        if torch.cuda.device_count() > 1 and MULTI_GPU_MODE == True:
            model.module.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'])

        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_resume = checkpoint['epoch']

    train_model(model, device, criterion, optimizer, scheduler, dataloaders, save_checkpoint, epoch_resume, num_epochs=NUM_EPOCHS)
