###################################################
#
#   Script to:
#   - Load the images and extract the patches
#   - Define the neural network
#   - define the training
#
##################################################


import numpy as np
from six.moves import configparser
import torch.backends.cudnn as cudnn
from data import ImageFolder
import sys
sys.path.insert(0, '../')
#from lib.help_functions import *
from tensorboardX import SummaryWriter
writer = SummaryWriter()
#function to obtain data for traininsg/testing (validation)
#from lib.extract_patches import get_data_training
import random
import os

from losses import *

import torch.optim as optim

from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

import random


print("Random Seed: ", 13)
random.seed(13)
torch.manual_seed(13)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

#=========  Load settings from Config file
config = configparser.RawConfigParser()
config.read('../configuration.txt')

#patch to the datasets
path_data = config.get('data paths', 'path_local')
#Experiment name
name_experiment = config.get('experiment name', 'name')
#training settings
N_epochs = int(config.get('training settings', 'N_epochs'))
batch_size = int(config.get('training settings', 'batch_size'))

#========== Define parameters here =============================
# log file
if not os.path.exists('./logs'):
    os.mkdir('logs')

device = 'cuda' if torch.cuda.is_available() else 'cpu'

start_epoch = 0  # start from epoch 0 or last checkpoint epoch
total_epoch = 250

val_portion = 0.1

#Define epoch and lr 
lr_epoch = np.array([20, 150, total_epoch])
lr_value= np.array([0.01, 0.001, 0.0001])
#Number of layers and filters
layers = 4
filters = 10
input_channel = 1

from LadderNetv65 import LadderNetv6

net = LadderNetv6(num_classes=2,layers=layers,filters=filters,inplanes=input_channel)
print("Total number of parameters: "+str(count_parameters(net)))

check_path = 'LadderNetv65_layer_%d_filter_%d.pt7'% (layers,filters) #'UNet16.pt7'#'UNet_Resnet101.pt7'

resume = False

criterion = LossMulti(jaccard_weight=0)
#criterion = CrossEntropy2d()

#optimizer = optim.SGD(net.parameters(),
#                     lr=lr_schedule[0], momentum=0.9, weight_decay=5e-4, nesterov=True)

optimizer = optim.Adam(net.parameters(),lr=lr_value[0])



##################################################NEW################################
dataset = ImageFolder(root_path="../../FDRIVE", datasets='Brain',mode ='train')
data_loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=70,
    shuffle=True,
    num_workers=4)


valid = ImageFolder(root_path="../../FDRIVE", datasets='Brain', mode = 'valid')
data_loader_v = torch.utils.data.DataLoader(
    valid,
    batch_size=70,
    shuffle=True,
    num_workers=4)                          


best_loss = np.Inf

# create a list of learning rate with epochs
lr_schedule = np.zeros(total_epoch)
for l in range(len(lr_epoch)):
    if l ==0:
        lr_schedule[0:lr_epoch[l]] = lr_value[l]
    else:
        lr_schedule[lr_epoch[l-1]:lr_epoch[l]] = lr_value[l]

if device == 'cuda':
    net.cuda()
    #net = torch.nn.DataParallel(net, device_ids=range(torch.cuda.device_count()))
    cudnn.benchmark = True
if resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/'+check_path)
    net.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    train_accuracy = 0
    train_dice_loss = 0
    JS= 0
    
    data_loader_iter = iter(data_loader)
    
    # get learning rate from learing schedule
    lr = lr_schedule[epoch]
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    print("Learning rate = %4f\n" % lr)
    i = 0
    for inputs, targets in data_loader_iter:
        inputs, targets = inputs.to(device), targets.to(device).long()
        optimizer.zero_grad()
        outputs_logits = net(inputs)

        loss, dice_loss , jacard = criterion(outputs_logits, targets)
        loss.backward()
        optimizer.step()
        if loss.item()<=1000:
            train_loss += loss.item()
        train_dice_loss += dice_loss.item()
        JS += jacard.item()

        outputs = F.softmax(outputs_logits, dim =1)
        _, predicted = torch.max(outputs.data, 1)
        total_train = targets.size(0)*targets.size(1)*targets.size(2)
        correct_train = predicted.eq(targets.data).sum().item()

        accuracy = correct_train / total_train
        train_accuracy += accuracy
        #Metrics
        #print(i, accuracy, jacard.item(), 1-dice_loss.item() )
        i+=1

    JS = JS/len(data_loader_iter)
    DC = 1 -train_dice_loss/len(data_loader_iter)
    total_loss = train_loss/len(data_loader_iter)
    TA = train_accuracy/len(data_loader_iter)
    writer.add_scalar('Train Loss', total_loss, epoch)
    writer.add_scalar('Train Dice Coefficient', DC, epoch)
    writer.add_scalar('Train Jacard Similarity', JS, epoch)
    writer.add_scalar('Train Accuracy', TA, epoch)
    print("Epoch %d: Train loss %4f Train dice_coeff %4f Train Accuracy %4f  Train Jacard_sim %4f \n" % (epoch, total_loss, DC, TA, JS ))

def test(epoch, display=False):
    data_loader_iter_v = iter(data_loader_v)
    global best_loss
    net.eval()
    test_loss = 0
    test_dice_loss = 0
    valid_accuracy = 0
    JS = 0
    
    with torch.no_grad():

        for inputs, targets in data_loader_iter_v:
            inputs, targets = inputs.to(device), targets.to(device).long()

            outputs_logits = net(inputs)

            loss, dice_loss , jacard = criterion(outputs_logits, targets)


            test_loss += loss.item()
            test_dice_loss += dice_loss.item()
            JS += jacard.item()

            outputs = F.softmax(outputs_logits, dim =1)
            _, predicted = torch.max(outputs.data, 1)
            total_train = targets.size(0)*targets.size(1)*targets.size(2)
            correct_train = predicted.eq(targets.data).sum().item()

            accuracy = correct_train / total_train
            valid_accuracy += accuracy
        JS = JS/len(data_loader_iter_v)
        DC = 1 -test_dice_loss/len(data_loader_iter_v)
        test_loss = test_loss/len(data_loader_iter_v)
        TA = valid_accuracy/len(data_loader_iter_v)
        writer.add_scalar('Valid Loss', test_loss, epoch)
        writer.add_scalar('Valid Dice Coefficient', DC, epoch)
        writer.add_scalar('Valid Jacard Similarity', JS, epoch)
        writer.add_scalar('Valid Accuracy', TA, epoch)
        print("Epoch %d: Valid loss %4f Valid dice_coeff %4f Valid Accuracy %4f  Valid Jacard_sim %4f \n" % (epoch, test_loss, DC, TA, JS ))
    # Save checkpoint.
    if test_loss < best_loss:
        print('Saving..')
        state = {
            'net': net.state_dict(),
            'best_loss': best_loss,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/' + str(epoch) + check_path)
        best_loss = test_loss

for epoch in range(start_epoch,total_epoch):
    train(epoch)
    test(epoch,False)
writer.close()
