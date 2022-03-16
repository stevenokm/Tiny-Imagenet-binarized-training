import glob
import os
import copy
from shutil import move
from os import rmdir
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import brevitas.nn as qnn
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

data_dir = './tiny-imagenet-200'
num_label = 200
normalize = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821))
#transform_train = transforms.Compose([transforms.RandomResizedCrop(32), transforms.RandomHorizontalFlip(), transforms.ToTensor(),        normalize, ])
#transform_test = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), normalize, ])
transform_train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(),        normalize, ])
transform_test = transforms.Compose([transforms.ToTensor(), normalize, ])
trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'), transform=transform_train)
testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'), transform=transform_test)
#trainset = datasets.ImageFolder(root=os.path.join(data_dir, 'train'))
#testset = datasets.ImageFolder(root=os.path.join(data_dir, 'val'))
train_loader = torch.utils.data.DataLoader(trainset, batch_size=10, shuffle=True, pin_memory=True)
test_loader = torch.utils.data.DataLoader(testset, batch_size=10, shuffle=False, pin_memory=True)


import torch
from torch.nn import Module, ModuleList, BatchNorm2d, MaxPool2d, BatchNorm1d
from torch.nn import Identity
import brevitas.onnx as bo

from brevitas.nn import QuantConv2d, QuantIdentity, QuantLinear
from brevitas.core.restrict_val import RestrictValueType
from tensor_norm import TensorNorm
from common import CommonWeightQuant, CommonActQuant

CNV_OUT_CH_POOL = [(4, False), (4, True), (8, False), (8, True), (16, False), (16, False)]
INTERMEDIATE_FC_FEATURES = [(1296, 750), (750, 500)]
LAST_FC_IN_FEATURES = 500
LAST_FC_PER_OUT_CH_SCALING = False
POOL_SIZE = 2
KERNEL_SIZE = 3

class CNV(Module):

    def __init__(self,
                 input_channels = 3,
                 num_classes = 200,
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
            self.conv_features.append(QuantConv2d(
                kernel_size=KERNEL_SIZE,
                in_channels=in_ch,
                out_channels=out_ch,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            in_ch = out_ch
            self.conv_features.append(BatchNorm2d(in_ch, eps=1e-4))
            self.conv_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))
            if is_pool_enabled:
                self.conv_features.append(MaxPool2d(kernel_size=2))

        for in_features, out_features in INTERMEDIATE_FC_FEATURES:
            self.linear_features.append(QuantLinear(
                in_features=in_features,
                out_features=out_features,
                bias=False,
                weight_quant=CommonWeightQuant,
                weight_bit_width=weight_bit_width))
            self.linear_features.append(BatchNorm1d(out_features, eps=1e-4))
            self.linear_features.append(QuantIdentity(
                act_quant=CommonActQuant,
                bit_width=act_bit_width))

        self.linear_features.append(QuantLinear(
            in_features=LAST_FC_IN_FEATURES,
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
        x = x.view(x.shape[0], -1)
        for mod in self.linear_features:
            x = mod(x)
        return x
    
def cnv():
    net = CNV()
    return net


net = cnv()
net.to(device)

from torch.autograd import Variable
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(train_loader):
        (inputs, targets) = data
        '''if args.sess == 'cnv_1w1a' or args.sess == 'cnv_1w1a_wsconv':
            (inputs, targets) = data
        else:
            (inputs, _, targets, _, _) = data'''

        # Baseline Implementation
        inputs, targets = Variable(inputs), Variable(targets)
        outputs = net(inputs)

        optimizer.zero_grad()
        loss = criterion(outputs, targets)

        '''if __debug__:
            print("#")
            print("# before")
            with torch.no_grad():
                for layer in net.modules():
                    if isinstance(layer, qnn.QuantConv2d) or isinstance(
                            layer, qnn.QuantLinear):
                        layer_mean = torch.mean(layer.weight)
                        layer_quant_mean = torch.mean(layer.quant_weight())
                        print(
                            layer.__module__, layer_mean, layer_quant_mean,
                            torch.abs(layer_mean) -
                            torch.abs(layer_quant_mean))
'''
        p_loss = 0.0
        # if wsconv:
        #     all_p_params = torch.zeros(1, device=device)
        #     with torch.no_grad():
        #         for layer in net.modules():
        #             if isinstance(layer, qnn.QuantConv2d) or isinstance(
        #                     layer, qnn.QuantLinear):
        #                 layer_std, layer_mean = torch.std_mean(layer.weight)
        #                 layer.weight -= layer_mean
        #                 layer.weight /= layer_std
        #                 layer.weight *= torch.numel(layer.weight)**-.5
        #             elif isinstance(layer, NegBiasLayer):
        #                 all_p_params = torch.cat(
        #                     (all_p_params, layer.bias.data))
        #     p_loss = args.p_factor * torch.norm(all_p_params, 1)
        #     loss += p_loss

        '''if __debug__:
            print("# after")
            with torch.no_grad():
                for layer in net.modules():
                    if isinstance(layer, qnn.QuantConv2d) or isinstance(
                            layer, qnn.QuantLinear):
                        layer_mean = torch.mean(layer.weight)
                        layer_quant_mean = torch.mean(layer.quant_weight())
                        print(
                            layer.__module__, layer_mean, layer_quant_mean,
                            torch.abs(layer_mean) -
                            torch.abs(layer_quant_mean))'''

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += targets.size(0)
        correct += predicted.eq(targets.data).cpu().sum()
        correct = correct.item()

        # progress_bar(
        #     batch_idx, len(trainloader),
        #     'Loss: %.3f | P loss: %.3f | Acc: %.3f%% (%d/%d)' %
        #     (train_loss /
        #      (batch_idx + 1), p_loss, 100. * correct / total, correct, total))
        print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %(train_loss /
             (batch_idx + 1), 100. * correct / total, correct, total))

    return (train_loss / batch_idx, 100. * correct / total)

    best_acc = 0  # best test accuracy

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

            inputs, targets = Variable(inputs), Variable(targets)

            outputs = test_model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum()
            correct = correct.item()

            print('Loss: %.3f | Acc: %.3f%% (%d/%d)' %
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
input_size = (1, 3, 64, 64)
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
    checkpoint_name = 'tiny_imagenet_smaller'
    torch.save(state, './checkpoint/' + checkpoint_name + '.pth')


    if not os.path.isdir('export_finn'):
        os.mkdir('export_finn')
    export_file_temp_name = checkpoint_name + '.onnx'
    export_file_temp_path = './export_finn/' + export_file_temp_name

    torch_model = copy.deepcopy(net)
    bo.export_finn_onnx(torch_model, input_size, export_file_temp_path)


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""

    lr = base_learning_rate
    if epoch >= 10:
        lr /= 10
    if epoch >= 30:
        lr /= 10
    if epoch >= 50:
        lr /= 10
    if epoch >= 60:
        lr /= 10
    if epoch >= 80:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

optimizer = optim.SGD(net.parameters(),
                      lr=0.1,
                      momentum=0.9,
                      weight_decay=1e-4)

criterion = nn.CrossEntropyLoss()
train_loss, train_acc = train(1)

best_acc = 0
test_loss, test_acc = test(1)
