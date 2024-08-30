from torch.nn.modules.module import Module
from torch import nn, optim
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch
import math
import copy
import numpy as np
from sklearn.metrics import mean_absolute_error, roc_auc_score
from scipy.special import expit

import seaborn as sns

class MLP(Module):
    def __init__(self, layer_list, num_clf,dropout):
        super(MLP, self).__init__()
        self.model_layer=nn.ModuleList()
        self.classifiers= nn.Linear(layer_list[-2],num_clf)
        for layer in range(len(layer_list)-1):
            
            self.model_layer.append(nn.Linear(layer_list[layer], layer_list[layer+1]))
            if layer == len(layer_list)-2:
                break
            if layer == len(layer_list)-3:
                self.model_layer.append(nn.Tanh())
                self.model_layer.append(nn.Dropout(dropout))
            else: 
                self.model_layer.append(nn.ReLU())     
                self.model_layer.append(nn.Dropout(dropout))
            self.model_layer.append(nn.BatchNorm1d(layer_list[layer+1]))    
           
    def forward(self, x):
            
       for layer in range(len(self.model_layer)):
           x = self.model_layer[layer](x)
           if layer == len(self.model_layer)-2: #save fingerprint
               fingerprint = x
       
       out_class = self.classifiers(fingerprint)
       return(x, out_class,fingerprint)

class FP_AE(Module):
    def __init__(self, layer_list, additional_outputs ,dropout):
        super(FP_AE, self).__init__()
        self.model_layer = nn.ModuleList()
        mid_layer = int((len(layer_list)/2)-0.5 )
        self.ll_pred = nn.Linear(layer_list[mid_layer], additional_outputs)
        for layer in range(len(layer_list)-1):
                        
            self.model_layer.append(nn.Linear(layer_list[layer], layer_list[layer+1]))
            if layer == len(layer_list)-2: 
                break 
            self.model_layer.append(nn.Dropout(dropout))
            if layer == mid_layer-1:
                self.model_layer.append(nn.Tanh())
            else: 
                self.model_layer.append(nn.ReLU())
            self.model_layer.append(nn.BatchNorm1d(layer_list[layer+1]))
   
    
    def forward(self, x):
        for layer in range(len(self.model_layer)):
            x = self.model_layer[layer](x)
            if layer == int((len(self.model_layer)/2)+0.5): #save fingerprint
                fingerprint = x
        np = self.ll_pred(fingerprint)
        return(x,np, fingerprint)


class EarlyStopping():
    def __init__(self, patience = 10, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_metric = 1e16
        self.stop = False
    def __call__(self,metric):
        if self.best_metric - metric > self.min_delta:
            self.best_metric = metric
        else:
            self.counter += 1 
            if self.counter >= self.patience:
                self.stop = True
                print("Max Patience Reached")


class train_model():
    
    def __init__(self,model, seed, with_npl, norm = 0):
        torch.manual_seed(seed)
        self.model = model
        self.seed = seed
        self.reg_criterion = nn.MSELoss()
        self.clf_criterion = nn.BCEWithLogitsLoss()
        self.measures= {"loss":{"train":[], "val":[]}, "mae_overall":[], "mae_npl":[], "auc":[]}
        self.val_loss =[]
        self.with_npl = with_npl
        self.norm = norm
    def validate(self,data, scaler_std):
        self.model.eval()
    
        epoch_val_loss = 0
        pred_reg = []
        true_reg = []
        pred_clf = []
        true_clf = []
    
        for k, batch in enumerate(data):
            if self.model.__class__.__name__=="GCN":
                    reg_pred, clf_pred, fingerprint = self.model([batch[0][0].cuda(), batch[0][1].cuda(), batch[0][2]])
                    
            else:
                reg_pred, clf_pred, fingerprint = self.model(batch[0].cuda())
        
            loss_reg=self.reg_criterion(reg_pred, batch[1].cuda())
            loss_clf=self.clf_criterion(clf_pred, batch[2].cuda())
            if self.baseline == True:
                loss = loss_clf #
            else:
                loss = loss_clf + loss_reg
            if self.norm>0:
                loss +=  0.1*(torch.linalg.norm(fingerprint,ord= self.norm, dim=1).sum()/fingerprint.shape[0])

                
            epoch_val_loss += loss.cpu().item()
            pred_reg.append(reg_pred.cpu().detach().numpy())
            pred_clf.append(clf_pred.cpu().detach().numpy())
            true_reg.append(batch[1])
            true_clf.append(batch[2])
        epoch_val_loss /= len(data)
        pred_reg = scaler_std.inverse_transform(np.vstack(pred_reg))
        true_reg = scaler_std.inverse_transform(np.vstack(true_reg))
        pred_clf = expit(np.vstack(pred_clf))
        true_clf = np.vstack(true_clf)
        mae_overall = mean_absolute_error(true_reg,pred_reg)
        mae_npl = mean_absolute_error(true_reg[:,-1],pred_reg[:,-1])
        if self.with_npl == False:
            mae_npl = 99999
        auc = roc_auc_score( true_clf, pred_clf)
        
        return epoch_val_loss, mae_overall, mae_npl, auc
    
    def train(self,data, lr, epochs, scaler_std ,baseline = False, patience=10):
        self.model.cuda()
        self.lr = lr 
        self.epochs  =epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(data["train"]))  
        self.baseline = baseline
        earlystopping =EarlyStopping(patience=patience)
        
        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0
            for k,batch in enumerate(data["train"]):
                self.optimizer.zero_grad() 
                if self.model.__class__.__name__=="GCN":
                    reg_pred, clf_pred, fingerprint = self.model([batch[0][0].cuda(), batch[0][1].cuda(), batch[0][2]])

                else:
                    reg_pred, clf_pred, fingerprint = self.model(batch[0].cuda())
                loss_reg=self.reg_criterion(reg_pred, batch[1].cuda())
                loss_clf=self.clf_criterion(clf_pred, batch[2].cuda())
                if self.baseline == True:
                    loss = loss_clf #
                else:
                    loss = loss_clf + loss_reg
                if self.norm>0:
                    loss +=  0.1*(torch.linalg.norm(fingerprint,ord= self.norm, dim=1).sum()/fingerprint.shape[0])

                epoch_loss += loss.cpu().item()
                loss.backward()
                self.optimizer.step()

                self.scheduler.step()
                
        
            epoch_loss /= len(data["train"])
            epoch_val_loss, mae_overall, mae_npl, auc = self.validate(data["val"], scaler_std=scaler_std)
            self.measures["loss"]["train"].append(epoch_loss)
            self.measures["loss"]["val"].append(epoch_val_loss)
            self.measures["mae_overall"].append(mae_overall)
            self.measures["mae_npl"].append(mae_npl)
            self.measures["auc"].append(auc)
           
            print('Epoch {0}: Trainings Loss: {1:.{digits}f}, Val Loss: {2:.{digits}f}, Overall MAE: {3:.{digits}f}, NPL MAE: {4:.{digits}f}, AUC: {5:.{digits}f}'.format(epoch,epoch_loss,epoch_val_loss,
                                                                                                                      mae_overall,mae_npl,auc,digits=4 ))
            
            earlystopping(epoch_val_loss)                                                                                                                                                                            
            if earlystopping.stop == False:
                self.best_model = copy.deepcopy(self.model)
            if (earlystopping.stop == True) | (epoch == (self.epochs-1)):
                return self.best_model    

    def plot_loss(self):
        sns.lineplot(np.arange(len(self.measures["loss"]["train"])), self.measures["loss"]["train"])
        sns.lineplot(np.arange(len(self.measures["loss"]["train"])), self.measures["loss"]["val"])

    def save(self, path):
        torch.save(self.best_model.state_dict(), path)        

class train_ae():
    def __init__(self,model, seed, with_npl, norm ):    
        torch.manual_seed(seed)
        self.model = model
        self.seed = seed
        self.loss_function1 = nn.BCEWithLogitsLoss()
        self.loss_function2= nn.BCEWithLogitsLoss()
        self.reg_criterion = nn.MSELoss()
        self.val_loss =[]
        self.model.cuda()
        self.model.train()
        self.best_model = None
        self.with_npl = with_npl
        self.norm = norm
    
    def pretrain(self,data, lr = 0.0001,epochs = 200, patience = 10): 
        self.train(data, lr, epochs, pretrain = True, patience = 10)
        
    def train(self,data, lr, epochs, pretrain = False, patience=10, weighting = [0.5,0.5]):
        self.model.cuda()
        self.lr = lr 
        self.epochs  =epochs
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, max_lr=self.lr, epochs=self.epochs, steps_per_epoch=len(data["train"]))  
        earlystopping =EarlyStopping(patience=patience)
        
        
        for i in range(epochs):
            epoch_loss =0
            for _,batch in enumerate(data["train"]):
                self.optimizer.zero_grad()
                prediction,nps ,fingerprint = self.model(batch[0].cuda())
                loss_reconstruct = self.loss_function1(prediction, batch[0].cuda())
                loss = 0
                if pretrain != True:
                    loss += weighting[0]*loss_reconstruct 
                    loss +=weighting[1]*self.loss_function2(nps[:,0], batch[1].cuda())
                    if self.with_npl:
                        loss += 1/50*self.reg_criterion(nps[:,1], batch[2].cuda())
                        
                else:
                    loss = loss_reconstruct
                if self.norm>0:
                    loss +=  0.1*(torch.linalg.norm(fingerprint,ord= self.norm, dim=1).sum()/fingerprint.shape[0])
                
                
                ## 3. backward propagation
                loss.backward()
                    
                ## 4. weight optimization
                self.optimizer.step()
                self.scheduler.step() 
                #save epoch_loss
                epoch_loss+= loss.detach().item()       
        
            train_loss=epoch_loss / len(data["train"])
           
            # VALIDATION
            self.model.eval()  
        
            pred_fp = []
            true_fp = []
            pred_np = []
            true_np = []
        
            val_loss=0
            for _,batch in enumerate(data["val"]):
                prediction, nps,_ = self.model(batch[0].cuda())
                if pretrain != True:
                    val_loss +=99/100*self.loss_function1(prediction, batch[0].cuda()).detach().clone().item()  
                    val_loss +=1/100*self.loss_function2(nps[:,0], batch[1].cuda()).detach().clone().item()
                    if self.with_npl:
                        val_loss += 1/50*self.reg_criterion(nps[:,1], batch[2].cuda())
                    pred_np.append(nps[:,0].cpu().detach().numpy())
                    true_np.append(batch[1].detach().numpy())
            
                else:
                    val_loss +=self.loss_function1(prediction, batch[0].cuda()).detach().clone().item()
                if self.norm>0:
                    val_loss +=  0.1*(torch.linalg.norm(fingerprint,ord= self.norm, dim=1).sum()/fingerprint.shape[0])
                pred_fp.append(prediction.cpu().detach().numpy())
                true_fp.append(batch[0].detach().numpy())
                
            val_loss = val_loss/(len(data["val"]))
            pred_prop=expit(np.vstack(pred_fp))
            pred_binary = np.round(pred_prop)
            true = np.vstack(true_fp)
            
            if pretrain != True:
                pred_np=expit(np.hstack(pred_np))
                true_np=np.hstack(true_np)
                print(roc_auc_score(true_np, pred_np))
            
            # evalue number of correct fps
            eval_data = true-pred_binary
            num_correct_bits = np.mean(np.sum(eval_data==0, axis=1))
            num_correct_fps = np.sum(np.sum(eval_data==0, axis=1)==2048)/eval_data.shape[0]
            num_correct_on_bits = np.sum((true==pred_binary)*true)/np.sum(true)

            
            print('Epoch {0}: Trainings Loss: {1:.{digits}f}, Val Loss: {4:.{digits}f}, Correct Bit: {2:.{digits}f}, %Correct Bits: {3:.{digits}f}, %Correct On Bits: {5:.{digits}f}'.format(i,
                                                                                                                                                                                train_loss,num_correct_bits,num_correct_fps,val_loss, num_correct_on_bits ,digits=4 ))
                                 
            self.model.train()  
            earlystopping(val_loss)                                                                                                                                                                            
            if earlystopping.stop == False:
                self.best_model = copy.deepcopy(self.model)
            if (earlystopping.stop == True) | (i == (self.epochs-1)):
                return self.best_model    
    
    def save(self, path):
        torch.save(self.best_model.state_dict(), path)        


class GraphConvolutionSkip(Module):
    
    """
    Graph Convolution layer as in Kipf & Welling 2016
    Skip Connection proposeed by Cangea et. al. 
    
    (D+I)^(-1/2)(A+I)(D+I)^(-1/2)XW¹ +XW² 
     
    
    """
    
    def __init__(self, in_features, out_features, bias=True):
        """
        Parameters
        --------------
        
        in_features: int 
           size of input (number of features)
        out_features: int
            size of output (after convolution)
        bias: bool (optional)
            inculde bias in weight (default is True) 
        """
        
        super(GraphConvolutionSkip, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weightSkip=Parameter(torch.FloatTensor(in_features, out_features))
        
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(0, stdv)
        
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

        
        stdv = 1. / math.sqrt(self.weightSkip.size(1))
        self.weightSkip.data.uniform_(0, stdv)
        
        

    def forward(self, input, adj):
        
        """
        Parameters
        ------------
        input: 2D tensor 
            matrix containing the features for each node (size=[number_of_nodes, number_of_features])
        adj: 2D tensor
            normalized adjacency matrix  (D+I)^(-1/2)(A+I)(D+I)^(-1/2) (size=[number_of_nodes, number_of_nodes])
            
        Return
        --------
        
        2D tensor
            activations of Convolution Layer
        """
        support = torch.mm(input, self.weight)
        skip = torch.mm(input, self.weightSkip)
        output = torch.spmm(adj, support)
        if self.bias is not None:
            return output + self.bias +skip
        else:
            return output + skip

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'




class GCN(nn.Module):
    """Graph Convolution Network with fully connected layers"""
    
    def __init__(self, gcn_size,linear_size, num_cat, dropout:float,attention:bool = False):

        """
        Parameters
        ----------
        gcn_size: List[int]
            list containing the size of each Graph Convolution
        linear_size: List[int]
            list containing the size of each Graph Convolution
        dropout: float
            percentage of dropout
        attention: bool 
            should attention be applied before pooling (default is False)
        """
        super(GCN,self).__init__()
        self.do_conv = len(gcn_size)>1
        self.embeddings= gcn_size[0]
        
        if self.do_conv:
            self.gcn =nn.ModuleList([GraphConvolutionSkip(gcn_size[k], gcn_size[k+1]) for k in range(len(gcn_size)-1) ])
            self.bng =nn.ModuleList([nn.BatchNorm1d(gcn_size[k]) for k in range(1,len(gcn_size)) ])
            self.embeddings =sum(gcn_size[1:])
       
        self.linear = nn.ModuleList([nn.BatchNorm1d(self.embeddings*2), nn.Linear(self.embeddings*2, linear_size[0]) ])
        
        for k in range(len(linear_size)-1):
            self.linear.append(nn.BatchNorm1d(linear_size[k]))
            self.linear.append(nn.Linear(linear_size[k], linear_size[k+1]))

        self.dropout= nn.Dropout(dropout)
        
        self.cat_layer = nn.Linear(linear_size[-2], num_cat)
    def forward(self,x):
        x, adj, slice_list=x

        """
        Parameters
        ----------
        x: 2d tensor
            feature matrix [number_of_nodes, number_of features], 
            for molecules in batches the feature matrices are stacked row-wise
           
        adj: 2d tensor 
            adjacency matrix [number_of_nodes, number_of_nodex]
            for molecules in batches the adjacency matrices are combined
            on the diagonal with zeros on the off diagonals e.g.:
           
            |a 0 0 0| 
            |0 a 0 0|
            |0 0 a 0|
            |0 0 0 a|
           
            where a are adjacency matices
    
        slice_list: List[int]
            contains the number of nodes of each molecule in the batch
            use for splitting the feature matirx for pooling
            
        Return
        -------
        x: 2D tensor
            activations of output layer. 
            IMPORTANT: No Activation Function has been applied
        fingerprint: 2D tensor
            activations of the last hidden layer, which make up the fingerprint of the moelcule
        attention_weights: List[float]
            each element of the list is a 2d tensor with the attention_weights for a molecule
        """
        
        
        store_embeddings = torch.rand([adj.shape[0],0]).cuda()
        attention_weights=[1]
        if self.do_conv:
            for layer in range(len(self.gcn)):
                x=F.relu(self.gcn[layer](x,adj))
                x=self.bng[layer](x)
                store_embeddings=torch.cat((store_embeddings,x), dim=1)
            x= store_embeddings
            # use the mean_mat to split teh bacthed graphs in a list of graphs
        x= torch.split(x,slice_list, dim=0)
    
        # caculate mean and max of each feature for each graph and concat. them
        max_x=torch.stack([graph.max(0)[0] for graph in x],0)
        mean_x=torch.stack([graph.mean(0) for graph in x],0)
        x=torch.cat((max_x,mean_x),1)
       
        x=self.dropout(x)
       
        for k in range(0,len(self.linear)-2,2):
            x=self.linear[k](x)
            x=F.relu(self.linear[k+1](x))
            x=self.dropout(x)
       
        x=fingerprint=self.linear[-2](x)
        
        x=self.linear[-1](x)
        out_class = self.cat_layer(fingerprint)
       
        return(x, out_class,fingerprint)
    