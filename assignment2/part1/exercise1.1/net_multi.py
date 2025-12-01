#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Experiment with different convolution types
Multi-device support: CUDA, MPS (Apple Silicon), CPU
"""

import torch
import numpy as np
np.random.seed(42)  # for reproducibility
torch.backends.cudnn.deterministic = True
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import argparse
import json
import os
from datetime import datetime
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from utils import gen_box_data, gen_box_data_test


def get_device():
    """
    Automatically detect and return the best available device.
    Priority: CUDA > MPS > CPU
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    return device


class Net(nn.Module):
    '''
    Create network with 4 Conv layers
    '''
    def __init__(self, conv_type, net_type):
        super(Net, self).__init__()
        self.net_type = net_type
        if conv_type == 'fconv':
            pad_type = 'zeros'
            padding_size = 2
            fc2_inp_dim = 64 * 6 * 6

        elif conv_type == 'sconv':
            pad_type = 'zeros'
            padding_size = 1
            fc2_inp_dim = 64 * 4 * 4

        elif conv_type == 'circular':
            pad_type = 'circular'
            padding_size = 2
            fc2_inp_dim = 64 * 6 * 6

        elif conv_type == 'reflect':
            pad_type = 'reflect'
            padding_size = 2
            fc2_inp_dim = 64 * 6 * 6

        elif conv_type == 'replicate':
            pad_type = 'replicate'
            padding_size = 2
            fc2_inp_dim = 64 * 6 * 6

        elif conv_type == 'valid':
            pad_type = 'zeros'
            padding_size = 0
            fc2_inp_dim = 64 * 2 * 2

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=2,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=2,
                               padding=padding_size, bias=False,
                               padding_mode=pad_type)
        self.fc1 = nn.Linear(64 * 1 * 1, 2)
        if self.net_type == 'Net1':
            self.adap_max = nn.AdaptiveMaxPool2d(1)
        elif self.net_type == 'Net2':
            self.fc2 = nn.Linear(fc2_inp_dim, 64 * 1 * 1)
        

    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        
        if self.net_type == 'Net1':
            x = self.adap_max(x)
        elif self.net_type == 'Net2':
            x = torch.flatten(x, start_dim=1)
            x = self.fc2(x)
        else:
            raise NotImplementedError
        x = x.view(-1, 64 * 1 * 1)
        x = self.fc1(x)
        return x

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_size", type=int, default=2000, help="training set size")
    parser.add_argument("--val_size", type=int, default=1000, help="validation set size")
    parser.add_argument("--test_size", type=int, default=1000, help="test set size")
    parser.add_argument("--conv_type", type=str, default='fconv',
                        help="please choose convolution type from 'valid', 'sconv','fconv','circular', 'reflect', 'replicate'")
    parser.add_argument("--net_type", type=str, default='Net1', help="please choose network type from 'Net1' and 'Net2'")
    parser.add_argument("--image_size", type=int, default=32, help="spatial size of sample image")
    parser.add_argument("--offset1", type=int, default=7, help="offset for class-1")
    parser.add_argument("--offset2", type=int, default=23, help="offset for class-2")
    parser.add_argument("--fluctuation", type=int, default=6, 
                        help="paramater for location fluctuation on the y-axis")
    parser.add_argument("--n_repeat", type=int, default=10,
                        help="training the network with n different initializations")
    parser.add_argument("--epochs", type=int, default=50, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=200, help="size of each image batch")
    parser.add_argument("--log", action="store_true", default=False,
                        help="log and store metrics and visualized results")
    parser.add_argument("--device", type=str, default="auto",
                        help="device to use: 'auto', 'cuda', 'mps', 'cpu'")

    opt = parser.parse_args()
    print(opt)

    # Device selection
    if opt.device == "auto":
        device = get_device()
    else:
        device = torch.device(opt.device)
        print(f"Using specified device: {device}")
    
    net_model = 'net_model_wts.pth'
    conv_type = opt.conv_type
    net_type = opt.net_type
    batch_size = opt.batch_size

    # Setup logging
    if opt.log:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_dir = f"logs/{conv_type}_{net_type}_{timestamp}"
        os.makedirs(log_dir, exist_ok=True)
        print(f"Logging enabled. Results will be saved to {log_dir}")
        
        # Save configuration
        config_dict = vars(opt)
        config_dict['device'] = str(device)  # Add device info
        with open(os.path.join(log_dir, "config.json"), "w") as f:
            json.dump(config_dict, f, indent=4)

    #******************************************************************#
    ###******************* DATASET GENERATION ***********************###
    #******************************************************************#

    trainset = np.zeros([opt.train_size, 3, opt.image_size, opt.image_size], dtype=float)
    y_train = np.zeros(opt.train_size)
    valset = np.zeros([opt.val_size, 3, opt.image_size, opt.image_size], dtype=float)
    y_val = np.zeros(opt.val_size)
    testset = np.zeros([opt.test_size, 3, opt.image_size, opt.image_size], dtype=float)
    y_test = np.zeros(opt.test_size)

    trainset, y_train = gen_box_data(trainset, y_train,
                                    length=opt.train_size, image_size=opt.image_size,
                                    offset1=opt.offset1, offset2=opt.offset2,
                                    shiftdiv=opt.fluctuation)
    valset, y_val = gen_box_data(valset, y_val,
                                length=opt.val_size, image_size=opt.image_size,
                                offset1=opt.offset1, offset2=opt.offset2,
                                shiftdiv=opt.fluctuation)

    testset1, y_test1, testset2, y_test2 = gen_box_data_test(testset, y_test,
                                                            testset.copy(), y_test.copy(),
                                                            length=opt.test_size,
                                                            image_size=opt.image_size, 
                                                            offset1=opt.offset1, offset2=opt.offset2, 
                                                            shiftdiv=opt.fluctuation)

    # from numpy to torch tensor

    train_set = torch.from_numpy(trainset)
    ytrain = torch.from_numpy(y_train)

    val_set = torch.from_numpy(valset)
    yval = torch.from_numpy(y_val)

    test_set1 = torch.from_numpy(testset1)
    ytest1 = torch.from_numpy(y_test1)

    test_set2 = torch.from_numpy(testset2)
    ytest2 = torch.from_numpy(y_test2)

    # dataloaders

    dataset = TensorDataset(train_set, ytrain)
    trainloader = DataLoader(dataset, batch_size=batch_size)

    datasetval = TensorDataset(val_set, yval)
    valloader = DataLoader(datasetval, batch_size=batch_size)

    datasettest_sim = TensorDataset(test_set1, ytest1)
    testloader_sim = DataLoader(datasettest_sim, batch_size=batch_size)

    datasettest_diss = TensorDataset(test_set2, ytest2)
    testloader_diss = DataLoader(datasettest_diss, batch_size=batch_size)

    results_val = {}  # dict for validation set
    results_test = {}  # dict for  testset
    
    # Track metrics for logging
    if opt.log:
        all_metrics = []

    for m in range(opt.n_repeat):

        print('Round  :{:4d}'.format(m + 1))

        #******************************************************************#
        ###****************** MODEL INITIALISATION **********************###
        #******************************************************************#

        torch.manual_seed(m)
        net = Net(conv_type=conv_type, net_type=net_type)
        net = net.to(device)  # Move model to device
        print(net)
        param_size = 0
        for param in net.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in net.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_kb = (param_size + buffer_size) / 1024
        print('Model size: {:.3f}KB'.format(size_all_kb))

        #******************************************************************#
        ###******************** TRAINING/TESTING ************************###
        #******************************************************************#

        ''' Loss functions and optimizer '''
        criterion_class = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta([{'params': net.parameters()}], lr=1.0, rho=0.9,
                                    eps=1e-06, weight_decay=0.00005)

        # Training

        print('Training is starting...')
        net.train()
        best = 50.0
        patience = 0
        flag = True
        
        # Track metrics per run
        if opt.log:
            run_metrics = {
                'run': m,
                'train_loss': [],
                'train_acc': [],
                'val_loss': [],
                'val_acc': []
            }
        
        for epoch in range(opt.epochs):
            train_loss = 0.0
            counter = 0.0
            total = 0.0
            train_corrects = 0.0

            if flag:

                for i, data in enumerate(trainloader, 0):
                    images = data[0]
                    label_class = data[1]
                    images = images.type(torch.FloatTensor)
                    label_class = label_class.type(torch.LongTensor)

                    # Move data to device
                    images = images.to(device)
                    label_class = label_class.to(device)

                    optimizer.zero_grad()
                    outputs_class = net(images)

                    _, predicted = torch.max(outputs_class, 1)
                    loss = criterion_class(outputs_class, label_class)

                    loss.backward()
                    optimizer.step()
                    # statistics
                    train_loss += loss.item()
                    train_corrects += torch.sum(predicted == label_class)
                    counter += 1
                    total += label_class.shape[0]

                epoch_loss = train_loss / (counter)
                epoch_acc = train_corrects.item() / (total)
                print('Epoch:{:4d}'.format(epoch + 1))
                print('Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                
                # Log training metrics
                if opt.log:
                    run_metrics['train_loss'].append(epoch_loss)
                    run_metrics['train_acc'].append(epoch_acc)

                # Validation

                val_loss = 0.0
                correct = 0.0
                total = 0.0

                for data in valloader:
                    images = data[0]
                    images = images.type(torch.FloatTensor)
                    label_class = data[1]
                    label_class = label_class.type(torch.LongTensor)

                    # Move data to device
                    images = images.to(device)
                    label_class = label_class.to(device)

                    outputs_class = net(images)

                    loss_class = criterion_class(outputs_class, label_class)
                    val_loss += loss_class.item()
                    _, predicted = torch.max(outputs_class.data, 1)

                    total += label_class.size(0)
                    correct += torch.sum(predicted == label_class)

                val_loss /= total
                print('val loss: ', val_loss)
                val_acc = 100 * correct.item() / (total)
                print('Accuracy of the network on the val images: %.4f %%' % (val_acc))
                
                # Log validation metrics
                if opt.log:
                    run_metrics['val_loss'].append(val_loss)
                    run_metrics['val_acc'].append(val_acc)
                
                results_val[m] = val_acc

                if best > val_loss:
                    best = val_loss
                    net_model_wts = net.state_dict()
                    torch.save(net.state_dict(net_model_wts), net_model)
                    patience = 0

                elif patience < 8:
                    patience += 1
                    print('patience', patience)

                elif patience >= 8:
                    flag = False

        print('Training finished.')

        #  -------------------------------------------------
        # Testing 
        net.load_state_dict(torch.load(net_model))  # using best model

        net.eval()
        testing_loss = 0.0
        correct = 0.0
        total = 0.0

        for data in testloader_diss:
            images = data[0]
            images = images.type(torch.FloatTensor)
            label_class = data[1]
            label_class = label_class.type(torch.LongTensor)

            # Move data to device
            images = images.to(device)
            label_class = label_class.to(device)

            outputs_class = net(images)

            loss_class = criterion_class(outputs_class, label_class)
            testing_loss += loss_class.item()
            _, predicted = torch.max(outputs_class.data, 1)

            total += label_class.size(0)
            correct += torch.sum(predicted == label_class)

        testing_loss /= total
        acc = 100 * correct.item() / (total)
        print('testing loss: ', testing_loss)
        print('Accuracy of the network on the testset images: %.4f %%' % (acc))

        results_test[m] = acc
        
        # Log test metrics
        if opt.log:
            run_metrics['test_loss'] = testing_loss
            run_metrics['test_acc'] = acc
            all_metrics.append(run_metrics)
        
        print('Testing with testset finished')

    acc_val = np.zeros(m+1)
    acc_test = np.zeros(m+1)

    for i in range(m+1):

        acc_val[i] = results_val[i]
        acc_test[i] = results_test[i]

    mean_val = np.mean(acc_val)
    std_val = np.std(acc_val)
    mean_test = np.mean(acc_test)
    std_test = np.std(acc_test)

    print("*******************************************")
    print(" Type of convolution : ", conv_type)
    print(" Type of network : ", net_type)
    print(" Device : ", device)
    print("*******************************************")
    print('Results for validation dataset', results_val)
    print('mean: {:.4f} std: {:.4f} for validation'.format(mean_val, std_val))
    print('Results for test dataset', results_test)
    print('mean: {:.4f} std: {:.4f} for test'.format(mean_test, std_test))
    
    # Save metrics and create visualizations
    if opt.log:
        # Save all metrics to JSON
        metrics_summary = {
            'conv_type': conv_type,
            'net_type': net_type,
            'device': str(device),
            'validation': {
                'all_runs': results_val,
                'mean': float(mean_val),
                'std': float(std_val)
            },
            'test': {
                'all_runs': results_test,
                'mean': float(mean_test),
                'std': float(std_test)
            },
            'detailed_metrics': all_metrics
        }
        
        with open(os.path.join(log_dir, "metrics.json"), "w") as f:
            json.dump(metrics_summary, f, indent=4)
        print(f"Metrics saved to {os.path.join(log_dir, 'metrics.json')}")
        
        # Create visualizations
        # 1. Training and validation curves for each run
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        for run_data in all_metrics:
            run_id = run_data['run']
            epochs_range = range(1, len(run_data['train_loss']) + 1)
            
            # Training loss
            axes[0, 0].plot(epochs_range, run_data['train_loss'], alpha=0.5, label=f"Run {run_id}")
            # Training accuracy
            axes[0, 1].plot(epochs_range, run_data['train_acc'], alpha=0.5, label=f"Run {run_id}")
            # Validation loss
            axes[1, 0].plot(epochs_range, run_data['val_loss'], alpha=0.5, label=f"Run {run_id}")
            # Validation accuracy
            axes[1, 1].plot(epochs_range, run_data['val_acc'], alpha=0.5, label=f"Run {run_id}")
        
        axes[0, 0].set_title('Training Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        axes[0, 0].grid(True)
        
        axes[0, 1].set_title('Training Accuracy')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        axes[0, 1].grid(True)
        
        axes[1, 0].set_title('Validation Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        axes[1, 0].grid(True)
        
        axes[1, 1].set_title('Validation Accuracy (%)')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Accuracy (%)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        axes[1, 1].grid(True)
        
        plt.suptitle(f'Training Progress: {conv_type} - {net_type} ({device})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'training_curves.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Training curves saved to {os.path.join(log_dir, 'training_curves.png')}")
        
        # 2. Final accuracies comparison
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        runs = list(results_val.keys())
        val_accs = [results_val[r] for r in runs]
        test_accs = [results_test[r] for r in runs]
        
        axes[0].bar(runs, val_accs, alpha=0.7, color='blue')
        axes[0].axhline(y=mean_val, color='r', linestyle='--', label=f'Mean: {mean_val:.2f}%')
        axes[0].fill_between(runs, mean_val - std_val, mean_val + std_val, alpha=0.2, color='red', label=f'Std: {std_val:.2f}%')
        axes[0].set_title('Validation Accuracy per Run')
        axes[0].set_xlabel('Run')
        axes[0].set_ylabel('Accuracy (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].bar(runs, test_accs, alpha=0.7, color='green')
        axes[1].axhline(y=mean_test, color='r', linestyle='--', label=f'Mean: {mean_test:.2f}%')
        axes[1].fill_between(runs, mean_test - std_test, mean_test + std_test, alpha=0.2, color='red', label=f'Std: {std_test:.2f}%')
        axes[1].set_title('Test Accuracy per Run')
        axes[1].set_xlabel('Run')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Final Accuracies: {conv_type} - {net_type} ({device})', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'final_accuracies.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Final accuracies plot saved to {os.path.join(log_dir, 'final_accuracies.png')}")
        
        # 3. Summary statistics
        fig, ax = plt.subplots(figsize=(10, 6))
        
        categories = ['Validation', 'Test']
        means = [mean_val, mean_test]
        stds = [std_val, std_test]
        
        x_pos = np.arange(len(categories))
        ax.bar(x_pos, means, yerr=stds, alpha=0.7, capsize=10, color=['blue', 'green'])
        ax.set_xticks(x_pos)
        ax.set_xticklabels(categories)
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(f'Summary Statistics: {conv_type} - {net_type} ({device})\n({opt.n_repeat} runs)')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for i, (mean, std) in enumerate(zip(means, stds)):
            ax.text(i, mean + std + 1, f'{mean:.2f}±{std:.2f}%', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(os.path.join(log_dir, 'summary_statistics.png'), dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Summary statistics plot saved to {os.path.join(log_dir, 'summary_statistics.png')}")
        
        print(f"\nAll results saved to {log_dir}/")
