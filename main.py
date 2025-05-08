import copy
import math
import numpy as np
import torchvision.models as models
import torch.utils.data as data
from torchvision import transforms
import cv2
import torch.nn.functional as F
from torch.autograd import Variable
import pandas as pd
import os, torch
import torch.nn as nn
from utils import image_utils
import argparse, random


from PIL import Image
from models.NLFER import NLFER, load_pretrained_weights

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raf_path', type=str, default='./RAF-DB_basic', help='Raf-DB dataset path.')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Pytorch checkpoint file path')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Pretrained weights')
    parser.add_argument('--beta', type=float, default=0.7, help='Ratio of high importance group in one mini-batch.')
    parser.add_argument('--margin_1', type=float, default=0.07,
                        help='Rank regularization margin. Details described in the paper.')
    parser.add_argument('--margin_2', type=float, default=0.2,
                        help='Relabeling margin. Details described in the paper.')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
    parser.add_argument('--optimizer', type=str, default="adam", help='Optimizer, adam or sgd.')
    parser.add_argument('--lr', type=float, default=0.00004, help='Initial learning rate for sgd.')
    parser.add_argument('--momentum', default=0.9, type=float, help='Momentum for sgd')
    parser.add_argument('--workers', default=4, type=int, help='Number of data loading workers (default: 4)')
    parser.add_argument('--epochs', type=int, default=40, help='Total training epochs.')
    return parser.parse_args()


def run_training():

    args = parse_args()
    model = NLFER(img_size=224, num_classes=7)


    if args.checkpoint:
        print("Loading pretrained weights...", args.checkpoint)
        checkpoint = torch.load(args.checkpoint)
        checkpoint = checkpoint["state_dict"]
        model = load_pretrained_weights(model, checkpoint)
        for param in model.parameters():
            param.requires_grad = True

    data_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(scale=(0.02, 0.25)),
    ])

    train_dataset = Dataset(args.raf_path, train=True, transform=data_transforms, basic_aug=True)

    print('Train set size:', train_dataset.__len__())
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=args.batch_size,
                                               num_workers=args.workers,
                                               shuffle=True,
                                               pin_memory=True)

    data_transforms_val = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])])
    val_dataset = Dataset(args.raf_path, train=False, transform=data_transforms_val)
    print('Validation set size:', val_dataset.__len__())

    val_loader = torch.utils.data.DataLoader(val_dataset,
                                             batch_size=args.batch_size,
                                             num_workers=args.workers,
                                             shuffle=False,
                                             pin_memory=True)

    params = model.parameters()
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(params, weight_decay=1e-4)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(params, args.lr,
                                    momentum=args.momentum,
                                    weight_decay=1e-4)
    else:
        raise ValueError("Optimizer not supported.")

    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    model  = model.cuda()
    criterion = torch.nn.CrossEntropyLoss()

    margin_1 = args.margin_1
    margin_2 = args.margin_2
    beta = args.beta

    train_loss = []
    train_acc = []
    best_acc = 0
    for i in range(1, args.epochs + 1):
        running_loss = 0.0
        correct_sum = 0
        iter_cnt = 0
        model.train()
        for batch_i, (imgs, targets, indexes) in enumerate(train_loader):
            batch_sz = imgs.size(0)
            iter_cnt += 1
            tops = int(batch_sz * beta)
            optimizer.zero_grad()
            imgs = imgs.cuda()

            #
            attention_weights, outputs = model(imgs)
            if args.epochs  == 30:
                print(attention_weights)

            # Deterministic Evaluation
            _, top_idx = torch.topk(attention_weights.squeeze(), tops)
            _, down_idx = torch.topk(attention_weights.squeeze(), batch_sz - tops, largest=False)

            high_group = attention_weights[top_idx]
            low_group = attention_weights[down_idx]


            high_mean = torch.mean(high_group)
            low_mean = torch.mean(low_group)
            diff = low_mean - high_mean + margin_1

            if diff > 0:
                DC_loss = diff
            else:
                DC_loss = 0.0

            targets = targets.cuda()

            #
            targets = targets.long()
            loss = criterion(outputs, targets) + DC_loss
            loss.backward()
            optimizer.step()

            running_loss += loss
            _, predicts = torch.max(outputs, 1)
            correct_num = torch.eq(predicts, targets).sum()
            correct_sum += correct_num

        scheduler.step()
        acc = correct_sum.float() / float(train_dataset.__len__())
        running_loss = running_loss / iter_cnt
        train_loss.append(running_loss.item())
        train_acc.append(acc.item())

        print('[Epoch %d] Training accuracy: %.4f. Loss: %.3f' % (i, acc, running_loss))

        with torch.no_grad():
            running_loss = 0.0
            iter_cnt = 0
            bingo_cnt = 0
            sample_cnt = 0
            model.eval()
            for batch_i, (imgs, targets, _) in enumerate(val_loader):
                _, outputs = model(imgs.cuda())
                targets = targets.cuda()
                targets = targets.long()
                loss = criterion(outputs, targets)
                running_loss += loss
                iter_cnt += 1
                _, predicts = torch.max(outputs, 1)
                correct_num = torch.eq(predicts, targets)
                bingo_cnt += correct_num.sum().cpu()
                sample_cnt += outputs.size(0)

            running_loss = running_loss / iter_cnt
            acc = bingo_cnt.float() / float(sample_cnt)
            acc = np.around(acc.numpy(), 4)
            print("[Epoch %d] Validation accuracy:%.4f. Loss:%.3f" % (i, acc, running_loss))

        if acc > best_acc:
            torch.save({'iter': i,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(), },
                       os.path.join('checkpoint', "RAF-DB"+str(acc)+".pth"))
            print('Model saved.')
        if acc > best_acc:
            best_acc = acc
            print("best_acc:" + str(best_acc))


if __name__ == "__main__":
    run_training()
