from model import EventDetector
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from dataloader import StsqDB, ToTensor, Normalize
import torch.nn.functional as F
import numpy as np
from util import correct_preds
import collections
import matplotlib.pyplot as plt

import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def eval(model, split, seq_length, bs, n_cpu, disp):
    
    if use_no_element == False:
        dataset = StsqDB(data_file='data/no_ele/seq_length_{}/val_split_{}.pkl'.format(int(seq_length), split),
                        vid_dir='/home/akiho/projects/golfdb/data/videos_40/',
                        seq_length=int(seq_length),
                        transform=transforms.Compose([ToTensor(),
                                                    Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                        train=False)
    else:
        dataset = StsqDB(data_file='data/seq_length_{}/train_split_{}.pkl'.format(args.seq_length, args.split),
                    vid_dir='/home/akiho/projects/golfdb/data/videos_40/',
                    seq_length=int(seq_length),
                    transform=transforms.Compose([ToTensor(),
                                                Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),
                    train=True)

    data_loader = DataLoader(dataset,
                             batch_size=int(bs),
                             shuffle=False,
                             num_workers=n_cpu,
                             drop_last=True)

    correct = []

    if use_no_element == False:
        element_correct = [ [] for i in range(12) ]
        element_sum = [ [] for i in range(12)]
        confusion_matrix = np.zeros([12,12], int)
    else:
        element_correct = [ [] for i in range(13) ]
        element_sum = [ [] for i in range(13)]
        confusion_matrix = np.zeros([13,13], int)

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'].to(device), sample['labels'].to(device)
        logits = model(images) 
        probs = F.softmax(logits.data, dim=1)  ##確率
        labels = labels.view(int(bs)*int(seq_length))
        _, c, element_c, element_s, conf = correct_preds(probs, labels.squeeze())
        if disp:
            print(i, c)
        correct.append(c)
        for j in range(len(element_c)):
            element_correct[j].append(element_c[j])
        for j in range(len(element_s)):
            element_sum[j].append(element_s[j])
        confusion_matrix = confusion_matrix + conf


    PCE = np.mean(correct)
    all_element_correct = np.sum(element_correct, axis=1)
    all_element_sum = np.sum(element_sum, axis=1)
    element_PCE = all_element_correct / all_element_sum
    return PCE, element_PCE, all_element_correct, all_element_sum, confusion_matrix



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', default=1)
    parser.add_argument('--batch_size', default=4)
    parser.add_argument('--seq_length', default=300) 
    parser.add_argument('--model_num', default=900)
    parser.add_argument('--use_no_element', action='store_true') 
    args = parser.parse_args() 


    split = args.split
    seq_length = args.seq_length
    n_cpu = 6
    bs = args.batch_size

    use_no_element = args.use_no_element

    model = EventDetector(pretrain=True,
                          width_mult=1.,
                          lstm_layers=1,
                          lstm_hidden=256,
                          device=device,
                          bidirectional=True,
                          dropout=False,
                          use_no_element=use_no_element)

    save_dict = torch.load('models/swingnet_{}.pth.tar'.format(args.model_num))
    model.load_state_dict(save_dict['model_state_dict'])
    model.to(device)
    model.eval()
    PCE, element_PCE, all_element_correct, all_element_sum, confusion_matrix = eval(model, split, seq_length, bs, n_cpu, True)
    print('Average PCE: {}'.format(PCE))

    if use_no_element == False:
        element_names = ['Bracket', 'Change_edge', 'Chasse','Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle']
    else:
        element_names = ['Bracket', 'Change_edge', 'Chasse','Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','No_element']



    for j in range(len(element_PCE)):
        element_name = element_names[j]
        print('{}: {}  ({} / {})'.format(element_name, element_PCE[j], all_element_correct[j], all_element_sum[j]))

    
    ####################################################################
    print(confusion_matrix)
    fig, ax = plt.subplots(1,1,figsize=(8,6))
    ax.matshow(confusion_matrix, aspect='auto', vmin=0, vmax=10000, cmap=plt.get_cmap('Blues'))
    if args.use_no_element == False:
        plt.ylabel('Actual Category')
        plt.yticks(range(12), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(12), element_names)

        save_dir = '/home/akiho/projects/StSqDB/'
        plt.savefig(save_dir + 'op_figure_12.png')

    else:
        plt.ylabel('Actual Category')
        plt.yticks(range(13), element_names)
        plt.xlabel('Predicted Category')
        plt.xticks(range(13), element_names)      

        save_dir = '/home/akiho/projects/StSqDB/'
        plt.savefig(save_dir + 'op_figure_13.png')