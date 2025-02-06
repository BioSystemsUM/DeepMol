


import os
import time
import torch
import torch_geometric
from tqdm import tqdm

from deepmol.models.atmol.encoder_gat import GATNet
from deepmol.models.atmol.model_gat_pre import GATCon
from deepmol.models.atmol.nt_xent import NT_Xent

from torch import optim

from torch_geometric.data import DataListLoader

from deepmol.models.atmol.utils_gat_pretrain import TestbedDataset


def _train(net, data_loader, train_optimizer, temperature, epoch, epochs):
    net.train()
    total_loss, total_num, train_bar = 0.0, 0, tqdm(data_loader)
    feature_graph = torch.Tensor()
    feature_org = torch.Tensor()
    edge_weight = torch.Tensor()
    feature_weight = torch.Tensor()
    for tem in train_bar:
        graph1, out_1, org2, out_2,ew,xw = net(tem)
        feature_graph = torch.cat((feature_graph, torch.Tensor(graph1.cpu().data.numpy())), 0)
        feature_org = torch.cat((feature_org, torch.Tensor(org2.cpu().data.numpy())), 0)
        edge_weight = torch.cat((edge_weight, torch.Tensor(ew.cpu().data.numpy())))
        feature_weight = torch.cat((feature_weight, torch.Tensor(xw.cpu().data.numpy())))
        criterion = NT_Xent(out_1.shape[0], temperature, 1)
        loss = criterion(out_1, out_2)
        total_num += len(tem)
        total_loss += loss.item() * len(tem)
        train_bar.set_description('Train Epoch: [{}/{}] Loss: {:.8f}'.format(epoch, epochs, total_loss / total_num))

        train_optimizer.zero_grad()
        loss.backward()
        train_optimizer.step()

    return total_loss / total_num, feature_graph, feature_org, edge_weight.numpy(), feature_weight.numpy()


def train(temperature, batch_size, epochs, root, save_name_pre, n_output=128, device_ids=[0]):
    train_data = TestbedDataset(root=root, dataset='processed/data', patt='')
    train_data.load(os.path.join(root, "processed/data.pt"))

    print('use GAT encoder')
    model_encoder1 = GATNet()
    model_encoder2 = GATNet()
    model = GATCon(n_output, encoder1=model_encoder1, encoder2=model_encoder2)
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = torch_geometric.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-7)

    # training loop
    results = {'train_loss': [], 'test_acc@1': [], 'test_acc@5': []}
    if not os.path.exists('results_atmol/'):
        os.makedirs('results_atmol/')

    for epoch in range(0, epochs + 1):
        train_loader = DataListLoader(train_data, batch_size=batch_size, shuffle=True)
        train_loss, features, org, ew,xw = _train(model, train_loader, optimizer, temperature, epoch, epochs)
        results["train_loss"].append(train_loss)

        if epoch in list(range(0, epochs + 1, 5)):
            torch.save(model_encoder1.state_dict(), 'results_atmol/' + str(epoch) +'_model_encoder_gat_' + save_name_pre +'.pt')
    
    import pickle
    # Save the list to a pickle file
    with open('train_loss_atmol.pkl', 'wb') as file:
        pickle.dump(results["train_loss"], file)