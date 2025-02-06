#!/usr/bin/python
# -*- coding: UTF-8 -*-

import argparse
import numpy as np
from sklearn.metrics import f1_score
import torch
from torch_geometric.data import DataLoader

from deepmol.models.atmol.utils_gat_pretrain import TestbedDataset
from .encoder_gat import GATNet
from deepmol.models.atmol.model_clr_downstream import Model
import torch.nn as nn
import math
import os

# train for one epoch to learn unique features
def train(model, device, data_loader, optimizer, epoch, loss_fn):
    LOG_INTERVAL = 100
    print('Training on {} samples...'.format(len(data_loader.dataset)))
    model.train()
    feature_x = torch.Tensor()
    feature_org = torch.Tensor()
    feature_weight = torch.Tensor()
    edge_weight = torch.Tensor()
    for batch_idx, data in enumerate(data_loader):
        data = data.to(device)
        output, y, ew, xw = model(data)
        feature_x = torch.cat((feature_x, torch.Tensor(output.cpu().data.numpy())), 0)
        # feature_org = torch.cat((feature_org, torch.Tensor(xc16.cpu().data.numpy())), 0)
        feature_weight = torch.cat((feature_weight, torch.Tensor(xw.cpu().data.numpy())), 0)
        edge_weight = torch.cat((edge_weight, torch.Tensor(ew.cpu().data.numpy())), 0)
        # pred = nn.Sigmoid()(output)
        loss = loss_fn(output,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch_idx % LOG_INTERVAL == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(data.x),
                                                                           len(data_loader.dataset),
                                                                           100. * batch_idx / len(data_loader),
                                                                           loss.item()))
    return feature_x, feature_org, edge_weight.numpy(), feature_weight.numpy()

def predicting(model, device, data_loader):
    model.eval()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    feature_weight = torch.Tensor()
    edge_weight = torch.Tensor()
    print('Make prediction for {} samples...'.format(len(data_loader.dataset)))
    with torch.no_grad():
        for data in data_loader:
            data = data.to(device)
            output, y, e_weight, weight = model(data)
            # pred = nn.Sigmoid()(output)
            pred = output.to('cpu')
            y_ = y.to('cpu')
            e_weight = e_weight.to('cpu')
            weight = weight.to('cpu')
            total_preds = torch.cat((total_preds, pred), 0)
            total_labels = torch.cat((total_labels, y_), 0)
            edge_weight = torch.cat((edge_weight, e_weight), 0)
            feature_weight = torch.cat((feature_weight, weight),0)


    return total_preds.numpy().flatten(), total_labels.numpy().flatten(), edge_weight.numpy(), feature_weight.numpy()

def fine_tune_model(root, batch_size, model, n_tasks, epochs, device):

    train_data = TestbedDataset(root, dataset='processed/data')
    train_data.load(os.path.join(root, "processed/data.pt"))

    train_data_loader = DataLoader(train_data, batch_size=batch_size, shuffle=None)

    model_encoder = GATNet().cuda()
    model_encoder.load_state_dict(torch.load(model, map_location='cuda:0'))
    model = Model(n_output=n_tasks, encoder=model_encoder).cuda()

    loss_fn = nn.BCELoss()
    optimizer = torch.optim.Adam(model.pre.parameters(), lr=0.00001, weight_decay=1e-7)

    for epoch in range(epochs+1):
        feature_x, feature_org, ew, xw = train(model, device, train_data_loader, optimizer, epoch + 1, loss_fn=loss_fn)
        
        if (epoch + 1) % 10 == 0:
            S, T,__,__ = predicting(model, device, train_data_loader)
            S = (S > 0.5).astype(int)
            print(f1_score(T, S, average="macro"))


    # model.load_state_dict(torch.load(model_file_name))
    # S, T,__,__ = predicting(model, device,test_data_loader)

