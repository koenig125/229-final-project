import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--out', help='output path')
parser.add_argument('--res', help='ResNet model, choose from [r50, r101, r152]')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')


# to replace the last fully connected layer
class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

def main():
    global args
    args = parser.parse_args()
    model = None

    if args.res == 'r50':
        model = models.resnet50(pretrained=True)
    elif args.res == 'r101':
        model = models.resnet101(pretrained=True)
    elif args.res == 'r152':
        model = models.resnet152(pretrained=True)
    else:
        print('Misuse, please choose from r50, r101, r152')
    
    print('Loading dataset...')
    directory = args.data
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(directory, transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    print("Finished loading dataset.")

    model.fc = Identity()
    model = torch.nn.DataParallel(model).cuda()
    model.eval() 

    outputs = []
    labels = []

    for i, (input, label) in enumerate(loader):
        input_var = torch.autograd.Variable(input)

        with torch.set_grad_enabled(False):
            output = model(input_var)
            outputs.append(output)
            labels.append(label)

        if i % 50 == 0:
            print("Progress: current batch number ", i)

    print('Finished generating embeddings.')
        
    torch.save(torch.cat(outputs), args.out + '/embeddings_' + args.res + '.pt')
    torch.save(torch.cat(labels), args.out + '/labels_' + args.res + '.pt')
    
    print('Successfully saved embeddings and labels.')
        


if __name__ == '__main__':
    main()