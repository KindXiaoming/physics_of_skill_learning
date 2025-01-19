import time
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import numpy as np
import matplotlib.pyplot as plt


my_task_id = int(sys.argv[1])
num_tasks = int(sys.argv[2])

seeds = np.arange(4)
n_tasks = [1000]
#alphas = [0.5,1.0,2.0,4.0]
alphas = [0.5,1.0,1.2,1.4,1.6,1.8,2.0]
dims = (1 + np.arange(50))*20

xx, yy, zz, ww = np.meshgrid(seeds, n_tasks, alphas, dims)
params_ = np.transpose(np.array([xx.reshape(-1,), yy.reshape(-1,), zz.reshape(-1,), ww.reshape(-1,)]))

indices = np.arange(params_.shape[0])

my_indices = indices[my_task_id:indices.shape[0]:num_tasks]

for i in my_indices:

    seed = params_[i][0].astype('int') 
    n_task = params_[i][1].astype('int')
    alpha = params_[i][2].astype('float')
    dim = params_[i][3].astype('int')
    
    torch.manual_seed(seed)
    np.random.seed(seed)

    prob = 1/(np.arange(n_task) + 1)**(alpha)
    prob /= np.sum(prob)

    noise_scale = 0.0 # maybe due to non-linearity

    u = torch.randn(n_task, dim)
    u = u/torch.norm(u, dim=1, keepdim=True)
    task_vectors = u[:n_task]
    u_, s, v = torch.svd(task_vectors[:dim,:dim])
    task_vectors[:dim] = torch.eye(dim, dim)
    task_vectors[dim:] *= 0.

    batch_size = 1024

    # initial and target
    x = torch.nn.Parameter(torch.zeros(dim,))

    learning_rate = 0.01
    optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(0.0,0.0))
    #optimizer = torch.optim.Adam([x], lr=learning_rate, betas=(0.9,0.999))


    n_steps = 100000
    log = 1000
    task_abilities = []
    losses = []
    losses_skill = []
    n_uses = []


    for i in range(n_steps):

        optimizer.zero_grad()

        # project and get progress
        task_overlap = task_vectors @ x
        task_ability = torch.sigmoid(task_overlap)

        loss = np.sum(-np.log(task_ability.detach().numpy()) * prob)
        task_abilities.append(task_ability.detach().numpy())
        losses.append(loss)
        losses_skill.append(1 - task_ability.detach().numpy())

        # mimic gradient
        id = np.random.choice(n_task, batch_size, p=prob)
        task_vectors_batch = task_vectors[id]
        task_ability_batch = task_ability[id]

        loss_per_sample = -torch.log(task_ability_batch)

        loss = torch.mean(loss_per_sample)

        loss.backward()
        x.grad.data += noise_scale * torch.randn(dim,)

        # update 
        #x.grad = - neg_grad + torch.randn(dim) * noise_scale
        optimizer.step()

        neg_grad = - x.grad.detach()

        # compute #(used dimension)
        alignment = task_vectors * neg_grad[None, :] > 0
        n_used = torch.sum(alignment, dim=1)
        n_uses.append(n_used.detach().numpy())

        if i % log == 0:
            print(i)
            
            results = {}

            results['skill'] = np.array(task_abilities)
            results['losses'] = np.array(losses)

            np.savez(f'./results/nsl_ortho/alpha_{alpha}_dim_{dim}_ntask_{n_task}_seed_{seed}', **results)

    


