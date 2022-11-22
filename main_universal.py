#!/usr/bin/python
# -*- coding: UTF-8 -*-
from skimage.segmentation import slic
from PIL import Image
import numpy as np
import time
import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision.models as models

from utils import *
from flags import parse_handle
from model import CifarNet

# set random seed for reproduce
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True

#parsing input parameters
parser = parse_handle()
args = parser.parse_args()

#settings
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# mean and std, used for normalization
img_mean = np.array([0.5, 0.5, 0.5]).reshape((1, 3, 1, 1)).astype('float32')
img_std = np.array([1, 1, 1]).reshape((1, 3, 1, 1)).astype('float32')
img_mean_cuda = torch.from_numpy(img_mean).cuda()
img_std_cuda = torch.from_numpy(img_std).cuda()
img_normalized_ops = (img_mean_cuda, img_std_cuda)

def main(input_images, labels):
    # define model and move it cuda
    model = CifarNet().eval().cuda()
    model.load_state_dict(torch.load(args.attacked_model))

    # freeze model parameters
    for param in model.parameters():
        param.requires_grad = False

    # training
    num_success_list, success_init_list, counter_acc_list = batch_train(model, input_images, labels)
    return num_success_list, success_init_list, counter_acc_list

def image_split(input_image, kernel_size):
    img_height, img_width, channels = input_image.shape
    tile_height, tile_width = kernel_size
    tiled_array = input_image.reshape(
        img_height // tile_height,
        tile_height,
        img_width // tile_width,
        tile_width,
        channels
    )   
    tiled_array = tiled_array.swapaxes(1, 2)
    i = 0
    for j in range(len(tiled_array)):
        for k in range(len(tiled_array[0])):
            tiled_array[j, k, :, :, :] = i
            i += 1
    return tiled_array.reshape(32, 32, 3)

def batch_train(model, input_images, labels):  
    num_success_list = np.array([])
    success_init_list = np.array([])
    counter_list = np.array([])
    counter_acc_list = np.array([])
    L0 = 0.0
    L1 = 0.0
    L2 = 0.0
    Li = 0.0
    WL1 = 0.0
    WL2 = 0.0
    WLi = 0.0
    loss = np.array([])
    noise_weight_list = []
    scaled_image_list = []

    for i in range(len(input_images)):

        cur_start_time = time.time()
        # load image and preprocessing

        input_image = np.resize(input_images[i], (args.img_resized_width, args.img_resized_height, 3))
        # print(input_image.shape)
    
        # #calculate mask for group sparsity
        image_4_mask = np.array(input_image, dtype=np.uint8)
        segments = image_split(image_4_mask, (args.kernel_width, args.kernel_height))[:, :, 0]
    
        #axis transpose, rescaled to [0,1] and normalized
        input_image = np.array(input_image, dtype=np.float32)  
        if input_image.ndim <3:                               
            input_image = input_image[:,:,np.newaxis]
        
    
        input_image = np.transpose(input_image, (2, 0, 1))     
        input_image = input_image[np.newaxis,...]              
        input_image = input_image / 255              
        scaled_image = torch.Tensor(input_image).cuda()
    
        #process block mask
        block_num = max(segments.flatten())+1 - min(segments.flatten())     
        B = np.zeros((block_num, input_image.shape[1], args.img_resized_width, args.img_resized_height)) 

        label_gt = int(torch.argmax(model(scaled_image-0.5)).data)
        label_target = args.target
        if labels[i] == label_target:
            label_target += 1

        for index in range(min(segments.flatten()), max(segments.flatten())+1):
            mask = (segments == index)
            B[index - 1, :, mask] = 1
        B = torch.from_numpy(B).cuda().float()        
        noise_Weight = compute_sensitive(scaled_image, args.weight_type)
        noise_weight_list.append(noise_Weight.cpu().tolist())
        scaled_image_list.append(scaled_image.cpu().tolist())
    noise_weight_list = np.asarray(noise_weight_list)
    scaled_image_list = np.asarray(scaled_image_list)
    noise_Weight = np.mean(noise_weight_list, axis=0)

    # train
    for i in tqdm(range(len(scaled_image_list))):
        results = train_adptive(int(0), model, torch.Tensor(scaled_image_list[i]).cuda(), label_target, label_gt, B, noise_Weight)
        results['args'] = vars(args)
        results['running_time'] = time.time() - cur_start_time
        results['ground_truth'] = label_gt
        results['label_target'] = label_target
        results['segments'] = segments.tolist()
        results['noise_weight'] = torch.Tensor(noise_Weight).cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist()  
 
        # logging brief summary
        if results['status'] == True:
            num_success = 1.0
        else:
            num_success = 0.0
        num_success_list = np.append(num_success_list, num_success)

        if label_gt == labels[i]:
            success_init = 1.0
        else:
            success_init = 0.0
        success_init_list = np.append(success_init_list, success_init)

        if results['noise_label'][0] == labels[i]:
            counter_acc = 1.0
        else:
            counter_acc = 0.0
        counter_acc_list = np.append(counter_acc_list, counter_acc)

        # statistic for norm
        L0 += results['L0']
        L1 += results['L1']
        L2 += results['L2']
        Li += results['Li']
        WL1 += results['WL1']
        WL2 += results['WL2']
        WLi += results['WLi']
   
        # save metaInformation and results to logfile
        save_results(results, args)
        loss = np.append(loss, results['loss'])

    return num_success_list, success_init_list, counter_acc_list


def train_adptive(i, model, images, target, label_gt, B, noise_Weight):
    args.lambda1 = args.init_lambda1
    lambda1_upper_bound = args.lambda1_upper_bound
    lambda1_lower_bound = args.lambda1_lower_bound
    results_success_list=[]
    for search_time in range(1, args.lambda1_search_times+1):
        results = train_sgd_atom(model, images, target, label_gt, B, noise_Weight)
        results['lambda1'] = args.lambda1

        if results['status'] == True:
            results_success_list.append(results)
            
        if search_time < args.lambda1_search_times:
            if results['status'] == True:
                if args.lambda1 < 0.01*args.init_lambda1:  
                    break
                # success, divide lambda1 by two
                lambda1_upper_bound = min(lambda1_upper_bound,args.lambda1)
                if lambda1_upper_bound < args.lambda1_upper_bound:
                    args.lambda1 = (lambda1_upper_bound+ lambda1_lower_bound)/2
            else:
                # failure, either multiply by 10 if no solution found yet
                # or do binary search with the known upper bound
                lambda1_lower_bound = max(lambda1_lower_bound, args.lambda1)
                if lambda1_upper_bound < args.lambda1_upper_bound:
                    args.lambda1 = (lambda1_upper_bound+ lambda1_lower_bound)/2
                else:
                    args.lambda1 *= 10
    
    # if succeed, return the last successful results  
    if results_success_list:       
        return results_success_list[-1]
    # if fail, return the current results 
    else:
        return results

        
def train_sgd_atom(model, images, target_label, label_gt, B, noise_Weight):
    target_label_tensor=torch.tensor([target_label]).cuda()

    G = torch.ones(images.shape, dtype=torch.float32).cuda()
    epsilon = torch.zeros(images.shape, dtype=torch.float32).cuda()
    noise_Weight = torch.FloatTensor(noise_Weight).cuda()
    
    cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B, noise_Weight)
    ori_prediction, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)  
    
    cur_lr_e = args.lr_e
    cur_lr_g = {'cur_step_g': args.lr_g, 'cur_rho1': args.rho1, 'cur_rho2': args.rho2, 'cur_rho3': args.rho3,'cur_rho4': args.rho4}
    for mm in range(1,args.maxIter_mm+1):   
        epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, G, cur_lr_e, B, noise_Weight, mm, False)

        G, cur_lr_g = update_G(model, images, target_label_tensor, epsilon, G, cur_lr_g, B, noise_Weight, mm)

    G = (G > 0.5).float()
    epsilon, cur_lr_e = update_epsilon(model, images, target_label_tensor, epsilon, G, cur_lr_e, B, noise_Weight, mm, True)   
    
    G = G.cuda()
    epsilon = epsilon.cuda()
    noise_Weight = noise_Weight.cuda()
    cur_meta = compute_loss_statistic(model, images, target_label_tensor, epsilon, G, args, img_normalized_ops, B, noise_Weight)
    noise_label, adv_image = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)
    
    # recording results per iteration
    if label_gt != noise_label[0]:
        results_status=True
    else:
        results_status=False

    results = {
        'status': results_status,
        'noise_label': noise_label.tolist(),
        'ori_prediction': ori_prediction.tolist(),
        'loss': cur_meta['loss']['loss'],
        'l2_loss': cur_meta['loss']['l2_loss'],
        'cnn_loss': cur_meta['loss']['cnn_loss'],
        'group_loss':cur_meta['loss']['group_loss'],
        'G_sum': cur_meta['statistics']['G_sum'],
        'L0': cur_meta['statistics']['L0'],
        'L1': cur_meta['statistics']['L1'],
        'L2': cur_meta['statistics']['L2'],
        'Li': cur_meta['statistics']['Li'],
        'WL1': cur_meta['statistics']['WL1'],
        'WL2': cur_meta['statistics']['WL2'],
        'WLi': cur_meta['statistics']['WLi'],
        'G' : G.detach().cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist(),
        'epsilon' : epsilon.detach().cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist(),
        'adv_image' : adv_image.detach().cpu().numpy().squeeze(axis=0).transpose((1,2,0)).tolist()
    }
    return results

def update_epsilon(model, images, target_label, epsilon, G, init_lr, B, noise_Weight, out_iter, finetune):

    epsilon_list = []
    for i in range(len(images)):
        cur_step = init_lr
        train_epochs = int(args.maxIter_e/2.0) if finetune else args.maxIter_e
 
        epsilon_iter = []
        for cur_iter in range(1, train_epochs + 1):
            epsilon = epsilon.cuda()
            epsilon.requires_grad = True  
            G.requires_grad = False
            if G.is_cuda and epsilon.is_cuda:
                pass
            else:
                if not G.is_cuda:
                    G = G.cuda()
                else:
                    epsilon = epsilon.cuda()
            image_s = torch.Tensor(images[i])
            image_s.cuda()
            images_s = image_s + torch.mul(epsilon, G) 
            images_s = torch.clamp(images_s, args.min_pix_value, args.max_pix_value)  
            images_s = Normalization(images_s, img_normalized_ops) 
            prediction = model(images_s)
        
            #loss
            if args.loss == 'ce':
                ce = nn.CrossEntropyLoss()
                loss = ce(prediction, target_label)  
            elif args.loss == 'cw':     
                label_to_one_hot = torch.tensor([[target_label.item()]])
                label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
            
                real = torch.sum(prediction*label_one_hot)
                other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction - (label_one_hot*10000))
                loss = torch.clamp(other_max - real + args.confidence, min=0)

            if epsilon.grad is not None:
                epsilon.grad.data.zero_()
            loss.backward(retain_graph=True)
            epsilon_cnn_grad = epsilon.grad

            epsilon_grad = 2*epsilon*G*G*noise_Weight*noise_Weight + args.lambda1 * epsilon_cnn_grad
            epsilon = epsilon - cur_step * epsilon_grad
            epsilon = epsilon.detach()
        
            # updating learning rate
            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
                   
            # tick print
            if cur_iter % args.tick_loss_e == 0:
                cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)
                noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)

        epsilon_list.append(epsilon.cpu().tolist())   
    epsilon_list = np.asarray(epsilon_list)

    epsilon = torch.Tensor(np.mean(epsilon_list, axis=0))
    return epsilon, cur_step

def update_G(model, images, target_label, epsilon, G, init_params, B, noise_Weight, out_iter):

    G_list = []
    for i in range(len(images)):
        # initialize learning rate
        cur_step = init_params['cur_step_g']
        cur_rho1 = init_params['cur_rho1']
        cur_rho2 = init_params['cur_rho2']
        cur_rho3 = init_params['cur_rho3']
        cur_rho4 = init_params['cur_rho4']

        # initialize y1, y2 as all 1 matrix, and z1, z2, z4 as all zeros matrix
        y1 = torch.ones_like(G)
        y2 = torch.ones_like(G)
        y3 = torch.ones_like(G)
        z1 = torch.zeros_like(G)
        z2 = torch.zeros_like(G)
        z3 = torch.zeros_like(G)
        z4 = torch.zeros(1).cuda()
        ones = torch.ones_like(G)

        for cur_iter in range(1,args.maxIter_g+1):
            G = G.cuda()
            G.requires_grad = True
            epsilon.requires_grad = False
        
            # 1.update y1 & y2
            y1 = torch.clamp((G.detach() + z1/cur_rho1), 0.0, 1.0)
            y2 = project_shifted_lp_ball(G.detach() + z2/cur_rho2, 0.5*torch.ones_like(G))      
        
            # 2.update y3
            C=G.detach()+z3/cur_rho3                       
            BC = C*B                                       
            n,c,w,h = BC.shape
            Norm = torch.norm(BC.reshape(n, c*w*h), p=2, dim=1).reshape((n,1,1,1))   
            coefficient = 1-args.lambda2/(cur_rho3*Norm)    
            coefficient = torch.clamp(coefficient, min=0)   
            BC = coefficient*BC                           
        
            y3 = torch.sum(BC, dim=0, keepdim=True)      

            # 3.update G
            #cnn_grad_G
            if G.is_cuda and epsilon.is_cuda:
                pass
            else:
                if not G.is_cuda:
                    G = G.cuda()
                else:
                    epsilon = epsilon.cuda()
            image_s = images[i] + torch.mul(G, epsilon)
            image_s = torch.clamp(image_s, args.min_pix_value, args.max_pix_value)
            image_s = Normalization(image_s, img_normalized_ops).cuda()

            prediction = model(image_s)
        
            if args.loss == 'ce':
                ce = nn.CrossEntropyLoss()
                loss = ce(prediction, target_label)   

            elif args.loss == 'cw':
                label_to_one_hot = torch.tensor([[target_label.item()]])
                label_one_hot = torch.zeros(1, args.categories).scatter_(1, label_to_one_hot, 1).cuda()
            
                real = torch.sum(prediction*label_one_hot)
                other_max = torch.max((torch.ones_like(label_one_hot).cuda()-label_one_hot)*prediction - (label_one_hot*10000))
                loss = torch.clamp(other_max - real + args.confidence, min=0)
   
            if G.grad is not None:  #the first time there is no grad
                G.grad.data.zero_()
            loss.backward()
            cnn_grad_G = G.grad

            grad_G = 2 * G * epsilon * epsilon * noise_Weight * noise_Weight + args.lambda1*cnn_grad_G \
                    + z1 + z2 + z3+ z4 * ones + cur_rho1 * (G-y1) \
                    + cur_rho2 * (G-y2)+ cur_rho3 * (G-y3) \
                    + cur_rho4 * (G.sum().item() - args.k) * ones
                 
            G = G - cur_step*grad_G
            G = G.detach()

            # 4.update z1,z2,z3,z4
            z1 = z1 + cur_rho1 * (G.detach() - y1)
            z2 = z2 + cur_rho2 * (G.detach() - y2)
            z3 = z3 + cur_rho3 * (G.detach() - y3)
            z4 = z4 + cur_rho4 * (G.sum().item()-args.k)

            # 5.updating rho1, rho2, rho3, rho4
            if cur_iter % args.rho_increase_step == 0:
                cur_rho1 = min(args.rho_increase_factor * cur_rho1, args.rho1_max)
                cur_rho2 = min(args.rho_increase_factor * cur_rho2, args.rho2_max)
                cur_rho3 = min(args.rho_increase_factor * cur_rho3, args.rho3_max)
                cur_rho4 = min(args.rho_increase_factor * cur_rho4, args.rho4_max)
            
            # updating learning rate
            if cur_iter % args.lr_decay_step == 0:
                cur_step = max(cur_step*args.lr_decay_factor, args.lr_min)
        
            if cur_iter % args.tick_loss_g == 0:
                cur_meta = compute_loss_statistic(model, images, target_label, epsilon, G, args, img_normalized_ops, B, noise_Weight)
                noise_label, _ = compute_predictions_labels(model, images, epsilon, G, args, img_normalized_ops)  
            
            cur_iter = cur_iter + 1

            res_param = {'cur_step_g': cur_step, 'cur_rho1': cur_rho1,'cur_rho2': cur_rho2, 'cur_rho3': cur_rho3,'cur_rho4': cur_rho4}

        G_list.append(G.cpu().tolist())   
    G_list = np.asarray(G_list)

    G = torch.Tensor(np.mean(G_list, axis=0))
    return G, res_param

if __name__ == '__main__':
    main()

