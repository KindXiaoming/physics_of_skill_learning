import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import CCA

seed = 0
torch.manual_seed(seed)


class MLP(nn.Module):
    
    def __init__(self, width, act='relu', save_act=True, seed=0, device='cpu'):
        super(MLP, self).__init__()
        
        torch.manual_seed(seed)
        
        linears = []
        self.width = width
        self.depth = depth = len(width) - 1
        for i in range(depth):
            layer = nn.Linear(width[i], width[i+1])
            '''sm = sparse_mask(width[i], width[i+1]).T
            layer.weight.data *= sm * torch.sqrt(torch.tensor(width[i],))'''
            linears.append(layer)
        self.linears = nn.ModuleList(linears)
        
        if act == 'silu':
            self.act_fun = torch.nn.SiLU()
        elif act == 'relu':
            self.act_fun = torch.nn.ReLU()
        elif act == 'identity':
            self.act_fun = torch.nn.Identity()
        self.save_act = save_act
        self.device = device

    @property
    def w(self):
        return [self.linears[l].weight for l in range(self.depth)]
        
    def forward(self, x):
        
        
        for i in range(self.depth):
            
            x = self.linears[i](x)
            if i < self.depth - 1:
                x = self.act_fun(x)
                
        return x
    
        
    def fit(self, dataset, opt="LBFGS", steps=100, log=1, mask=None, lamb=0., lamb_l1=1., lamb_entropy=2., loss_fn=None, lr=1., batch=-1, metrics=None, in_vars=None, out_vars=None, beta=3, device='cpu', reg_metric='w', display_metrics=None, save_ckpt=False, save_freq=1, save_folder='ckpt'):
       
        pbar = tqdm(range(steps), desc='description', ncols=100)

        if loss_fn == None:
            if mask == None:
                loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2)
            else:
                loss_fn = loss_fn_eval = lambda x, y: torch.mean((x - y) ** 2 * mask)
        else:
            loss_fn = loss_fn_eval = loss_fn

        if opt == "Adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=lr, betas=(0.9,0.999))
        elif opt == "LBFGS":
            optimizer = LBFGS(self.parameters(), lr=lr, history_size=10, line_search_fn="strong_wolfe", tolerance_grad=1e-32, tolerance_change=1e-32, tolerance_ys=1e-32)
            

        results = {}
        results['train_loss'] = []
        results['test_loss'] = []
        if metrics != None:
            for i in range(len(metrics)):
                results[metrics[i].__name__] = []

        if batch == -1 or batch > dataset['train_input'].shape[0] or batch > dataset['test_input'].shape[0]:
            print('using full batch')
            batch_size = dataset['train_input'].shape[0]
            batch_size_test = dataset['test_input'].shape[0]
        else:
            batch_size = batch
            batch_size_test = batch

        global train_loss, reg_

        def closure():
            global train_loss, reg_
            optimizer.zero_grad()
            pred = self.forward(dataset['train_input'][train_id].to(self.device))
            train_loss = loss_fn(pred, dataset['train_label'][train_id].to(self.device))
            reg_ = torch.tensor(0.)
            objective = train_loss + lamb * reg_
            if opt == 'LBFGS':
                objective.backward()
            return objective

        for _ in pbar:
            
            if save_ckpt and _ % save_freq == 0:
                torch.save(self.state_dict(), f'./{save_folder}/{_}')
            
            train_id = np.random.choice(dataset['train_input'].shape[0], batch_size, replace=False)
            test_id = np.random.choice(dataset['test_input'].shape[0], batch_size_test, replace=False)

            if opt == "LBFGS":
                optimizer.step(closure)

            elif opt == "Adam":
                pred = self.forward(dataset['train_input'][train_id].to(self.device))
                train_loss = loss_fn(pred, dataset['train_label'][train_id].to(self.device))
                reg_ = torch.tensor(0.)
                loss = train_loss + lamb * reg_
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            test_loss = loss_fn_eval(self.forward(dataset['test_input'][test_id].to(self.device)), dataset['test_label'][test_id].to(self.device))
            
            
            if metrics != None:
                for i in range(len(metrics)):
                    results[metrics[i].__name__].append(metrics[i]().item())

            results['train_loss'].append(torch.sqrt(train_loss).cpu().detach().numpy())
            results['test_loss'].append(torch.sqrt(test_loss).cpu().detach().numpy())

            if _ % log == 0:
                if display_metrics == None:
                    pbar.set_description("| train_loss: %.2e | test_loss: %.2e | reg: %.2e | " % (torch.sqrt(train_loss).cpu().detach().numpy(), torch.sqrt(test_loss).cpu().detach().numpy(), reg_.cpu().detach().numpy()))
                else:
                    string = ''
                    data = ()
                    for metric in display_metrics:
                        string += f' {metric}: %.2e |'
                        try:
                            results[metric]
                        except:
                            raise Exception(f'{metric} not recognized')
                        data += (results[metric][-1],)
                    pbar.set_description(string % data)
                    
        return results

            
