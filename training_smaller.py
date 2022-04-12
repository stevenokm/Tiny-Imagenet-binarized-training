import glob
import os
import copy
from shutil import move
from os import rmdir

# NOTE: import onnx before torch
# reference: https://github.com/onnx/onnx/issues/2394#issuecomment-581638840
import onnx
import onnx.numpy_helper as nph

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d
from torch.nn import Identity
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

from torchinfo import summary

import os
import argparse
import numpy as np
import random

from thop import profile as thop_profile
from thop.vision import basic_hooks as thop_basic_hooks

import brevitas.onnx as bo
import brevitas.nn as qnn
from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.core.restrict_val import RestrictValueType
from brevitas_examples.bnn_pynq.models.tensor_norm import TensorNorm
from brevitas_examples.bnn_pynq.models.common import CommonWeightQuant, CommonActQuant

from utils import progress_bar

from losses import SqrHingeLoss

parser = argparse.ArgumentParser(
    description='PyTorch Complement Objective Training (COT)')
parser.add_argument('--resume',
                    '-r',
                    action='store_true',
                    help='resume from checkpoint')
parser.add_argument('--sess',
                    default='tiny_imagenet_smaller',
                    type=str,
                    help='session id')
parser.add_argument('--optimizer', default='SGD', type=str, help='optimizer')
parser.add_argument('--mem_fault',
                    default='baseline',
                    type=str,
                    help='mem fault pattern')
parser.add_argument('--seed', default=11111, type=int, help='rng seed')
parser.add_argument('--decay',
                    default=0,
                    type=float,
                    help='weight decay (default=1e-4)')
parser.add_argument('--lr',
                    default=0.02,
                    type=float,
                    help='initial learning rate')
parser.add_argument('--batch-size',
                    '-b',
                    default=128,
                    type=int,
                    help='mini-batch size (default: 128)')
parser.add_argument('--epochs',
                    default=50,
                    type=int,
                    help='number of total epochs to run')
parser.add_argument('-j',
                    '--workers',
                    default=8,
                    type=int,
                    metavar='N',
                    help='number of data loading workers (default: 8)')
parser.add_argument('--duplicate',
                    default=1,
                    type=int,
                    help='number of duplication of dataset')
parser.add_argument('--export_finn',
                    action='store_true',
                    help='export saved model to Xilinx FINN-used onnx')
parser.add_argument('--dataset_cache',
                    action='store_true',
                    help='use dataset npy cache for lower CPU utilization')
parser.add_argument('--train',
                    action='store_true',
                    help='perform model training')
parser.add_argument('--noise',
                    default=0.0,
                    type=float,
                    help='scale of injected noises')
parser.add_argument('--p_factor',
                    default=0.1,
                    type=float,
                    help='factor of p params regularization')
parser.add_argument('--pynq',
                    action='store_true',
                    help='perform model inference on pynq')
parser.add_argument('--hls_test',
                    action='store_true',
                    help='perform model inference on pynq only hls_test')
parser.add_argument('--cppsim',
                    action='store_true',
                    help='perform model inference on pynq only cppsim')
parser.add_argument('--rtlsim',
                    action='store_true',
                    help='perform model inference on pynq only rtlsim')
parser.add_argument('--fpga',
                    action='store_true',
                    help='perform model inference on pynq only deploy to fpga')

args = parser.parse_args()

print("args: ", args)

np.random.seed(args.seed)
random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
cudnn.deterministic = True

use_cuda = torch.cuda.is_available()
device = 'cuda' if use_cuda else 'cpu'
best_acc = 0  # best test accuracy
batch_size = args.batch_size
test_batch_size = 100
base_learning_rate = args.lr

if use_cuda:
    n_gpu = torch.cuda.device_count()
    batch_size *= n_gpu
    base_learning_rate *= n_gpu

# #### dataset import ####
# data_dir = './tiny-imagenet-200'
# num_label = 200
# normalize = transforms.Normalize((0.4802, 0.4481, 0.3975),
#                                  (0.2770, 0.2691, 0.2821))
# #transform_train = transforms.Compose([transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),        normalize, ])
# #transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
# transform_train = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     normalize,
# ])
# transform_test = transforms.Compose([
#     transforms.ToTensor(),
#     normalize,
# ])
# trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'),
#                                 transform=transform_train)
# testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'),
#                                transform=transform_test)
# #trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'))
# #testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'))
# train_loader = torch.utils.data.DataLoader(trainset,
#                                            batch_size=batch_size,
#                                            shuffle=True,
#                                            pin_memory=True)
# test_loader = torch.utils.data.DataLoader(testset,
#                                           batch_size=test_batch_size,
#                                           shuffle=False,
#                                           pin_memory=True)

#### dataset import (CIFAR-10 ####
data_dir = './cifar'
num_label = 10
# normalize = transforms.Normalize((0.4914, 0.4822, 0.4465),
#                                  (0.247, 0.243, 0.261))
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    # normalize,
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
    # normalize,
])
trainset = datasets.CIFAR10(root=os.path.join(data_dir),
                            train=True,
                            transform=transform_train,
                            download=True)
testset = datasets.CIFAR10(root=os.path.join(data_dir),
                           train=False,
                           transform=transform_test,
                           download=True)
train_loader = torch.utils.data.DataLoader(trainset,
                                           batch_size=batch_size,
                                           shuffle=True,
                                           pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset,
                                          batch_size=test_batch_size,
                                          shuffle=False,
                                          pin_memory=True)

#### CNV declaration ####
# VGG-13 like 9 layers CNN
CNV_OUT_CH_POOL = [(64, False), (64, True), (128, False), (128, True),
                   (256, False), (256, False)]
INTERMEDIATE_FC_FEATURES = [(16384, 4096), (4096, 1024)]
LAST_FC_IN_FEATURES = INTERMEDIATE_FC_FEATURES[1][1]
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3


class CNV(Module):

    def __init__(self,
                 input_channels=3,
                 num_classes=200,
                 weight_bit_width=1,
                 act_bit_width=1,
                 in_bit_width=8):
        super(CNV, self).__init__()

        in_ch = input_channels
        self.conv_features = ModuleList()
        self.linear_features = ModuleList()

        self.conv_features.append(
            QuantIdentity(  # for Q1.7 input format
                act_quant=CommonActQuant,
                bit_width=in_bit_width,
                min_val=-1.0,
                max_val=1.0 - 2.0**(-7),
                narrow_range=False,
                restrict_scaling_type=RestrictValueType.POWER_OF_TWO))

        for out_ch, is_pool_enabled in CNV_OUT_CH_POOL:
            self.conv_features.append(
                QuantConv2d(kernel_size=KERNEL_SIZE,
                            in_channels=in_ch,
                            out_channels=out_ch,
                            bias=False,
                            padding=1,
                            weight_quant=CommonWeightQuant,
                            weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(
                QuantIdentity(act_quant=CommonActQuant,
                              bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(
                QuantLinear(in_features=in_features,
                            out_features=out_features,
                            bias=False,
                            weight_quant=CommonWeightQuant,
                            weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(
                QuantIdentity(act_quant=CommonActQuant,
                              bit_width=act_bit_width))

        self.linear_features.append(
            QuantLinear(in_features=LAST_FC_IN_FEATURES,
                        out_features=num_classes,
                        bias=False,
                        weight_quant=CommonWeightQuant,
                        weight_bit_width=weight_bit_width))
        self.linear_features.append(TensorNorm())

        for m in self.modules():
            if isinstance(m, QuantConv2d) or isinstance(m, QuantLinear):
                torch.nn.init.uniform_(m.weight.data, -1, 1)

    def clip_weights(self, min_val, max_val):
        for mod in self.conv_features:
            if isinstance(mod, QuantConv2d):
                mod.weight.data.clamp_(min_val, max_val)
        for mod in self.linear_features:
            if isinstance(mod, QuantLinear):
                mod.weight.data.clamp_(min_val, max_val)

    def forward(self, x):
        x = 2.0 * x - torch.tensor([1.0], device=x.device)
        for mod in self.conv_features:
            x = mod(x)
            if __debug__:
                print(x.shape)
        x = x.view(x.shape[0], -1)
        if __debug__:
            print(x.shape)
        for mod in self.linear_features:
            x = mod(x)
        return x


def cnv():
    net = CNV(num_classes=num_label)
    return net


# Model
start_epoch = 0

net = cnv()
net.to(device)

brevitas_op_count_hooks = {
    QuantConv2d: thop_basic_hooks.count_convNd,
    QuantIdentity: thop_basic_hooks.zero_ops,
    QuantLinear: thop_basic_hooks.count_linear
}
input_size = (1, 3, 32, 32)
inputs = torch.rand(size=input_size, device=device)
thop_model = copy.deepcopy(net)
summary(thop_model, input_size=input_size)
macs, params = thop_profile(thop_model,
                            inputs=(inputs, ),
                            custom_ops=brevitas_op_count_hooks)

print('')
print("#MACs/batch: {macs:d}, #Params: {params:d}".format(
    macs=(int(macs / inputs.shape[0])), params=(int(params))))
print('')

del thop_model
torch.cuda.empty_cache()

if use_cuda:
    net = torch.nn.DataParallel(net)
    print('Using', torch.cuda.device_count(), 'GPUs.')
    cudnn.benchmark = True
    print('Using CUDA..')

if args.resume:
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
    checkpoint = torch.load('./checkpoint/ckpt.t7.' + args.sess + '_' +
                            str(args.seed) + '.pth')
    net.load_state_dict(checkpoint['net'])
    net.to(device)
    best_acc = checkpoint['acc']
    start_epoch = checkpoint['epoch'] + 1
    torch.set_rng_state(checkpoint['rng_state'])

criterion = nn.CrossEntropyLoss()
# criterion = SqrHingeLoss()
# optimizer = optim.SGD(net.parameters(),
#                       lr=base_learning_rate,
#                       momentum=0.9,
#                       weight_decay=args.decay)
optimizer = optim.Adam(net.parameters(),
                       lr=base_learning_rate,
                       betas=(0.9, 0.999),
                       weight_decay=args.decay)


def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(train_loader):
        (inputs, targets) = data

        if use_cuda:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

        # for hingeloss only
        if isinstance(criterion, SqrHingeLoss):
            targets = targets.unsqueeze(1)
            target_onehot = torch.Tensor(targets.size(0),
                                         num_label).to(device,
                                                       non_blocking=True)
            target_onehot.fill_(-1)
            target_onehot.scatter_(1, targets, 1)
            targets = targets.squeeze()
            target_var = target_onehot
        else:
            target_var = targets

        # Baseline Implementation
        inputs, target_var = Variable(inputs), Variable(target_var)
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, target_var)
        p_loss = 0.0
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        progress_bar(
            batch_idx, len(train_loader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (train_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / batch_idx, 100. * correct / total)


def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0

    test_model = copy.deepcopy(net)

    with torch.no_grad():

        print("#")
        for layer in test_model.modules():
            if isinstance(layer, qnn.QuantConv2d) or isinstance(
                    layer, qnn.QuantLinear):
                layer_mean = torch.abs(torch.mean(layer.weight))
                layer_quant_mean = torch.abs(torch.mean(layer.quant_weight()))
                print(layer.__module__, layer_mean, layer_quant_mean,
                      layer_mean - layer_quant_mean)

        for batch_idx, data in enumerate(test_loader):
            (inputs, targets) = data

            if use_cuda:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

            inputs, targets = Variable(inputs), Variable(targets)

            outputs = test_model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            progress_bar(
                batch_idx, len(test_loader),
                'Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (test_loss /
                 (batch_idx + 1), 100. * correct / total, correct, total))

    del test_model
    torch.cuda.empty_cache()

    # Save checkpoint.
    acc = 100. * correct / total
    if acc > best_acc:
        best_acc = acc
        checkpoint(acc, epoch)
    return (test_loss / batch_idx, 100. * correct / total)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    checkpoint_name = 'ckpt.t7.' + args.sess + '_' + str(args.seed) + '.pth'
    torch.save(state, './checkpoint/' + checkpoint_name + '.pth')

    if not os.path.isdir('export_finn'):
        os.mkdir('export_finn')
    export_file_temp_name = checkpoint_name + '.onnx'
    export_file_temp_path = './export_finn/' + export_file_temp_name

    torch_model = copy.deepcopy(net)
    if use_cuda:
        torch_model = torch_model.to('cpu').module
    bo.export_finn_onnx(torch_model, input_size, export_file_temp_path)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch >= 40:
        lr = 1e-3
    if epoch >= 80:
        lr = 5e-4
    if epoch >= 100:
        lr = 1e-4
    if epoch >= 120:
        lr = 5e-5
    if epoch >= 140:
        lr = 1e-5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


for epoch in range(start_epoch, args.epochs):
    adjust_learning_rate(optimizer, epoch)
    train_loss, train_acc = train(epoch)
    test_loss, test_acc = test(epoch)
