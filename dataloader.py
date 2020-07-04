import os.path as osp
import cv2
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

#TODO:  StsqDBを作る
class StsqDB(Dataset):
    def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
        # self.df = pd.read_pickle(data_file)
        self.data_file = data_file
        self.images, self.labels = self.load_data()
        self.vid_dir = vid_dir
        self.seq_length = seq_length
        self.transform = transform
        self.train = train

    def __len__(self):
        with open(self.data_file, "rb") as f:  ##ここで持ってくるファイルは全体のやつ？？or train?? >> これはtrainだね。
            data = pickle.load(f)
        return len(data)
        # return len(self.df)
     
    @property
    def element(self):
        ['Bracket', 'Change_edge', 'Chasse','Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','No_element']

    def get_element_name(self, id):
        return self.element[id]

    def __getitem__(self, idx):
        images = self.images[idx][:self.seq_length]
        labels = self.labels[idx][:self.seq_length]

        sample = { 'images':np.asarray(images), 'labels':np.asarray(labels) }
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_data(self):
        # pklファイルからimg, labelを読み込む
        with open(self.data_file, "rb") as f:  ##このファイルはtrain_split1をload中のはず。。あれ、ここは全部のやつか？？
            data = pickle.load(f)
        images = [ pair[0] for pair in data ]  ##len(images)=300
        labels = [ pair[1] for pair in data ] 
        return images, labels 


        # images = []
        # labels = []
        # for image,label in data:
        #     images.append(image)
        #     labels.append(label)

        # data = [ [[[]], [[]]] , [1,2, 4,0, ],   [[[]], [[]]] , [1,2, 4,0, ],   [[[]], [[]]] , [1,2, 4,0, ] ....  ] 

  
    # def __old_getitem__(self, idx):
    #     a = self.df.loc[idx, :]  # annotation info
    #     events = a['events'] 
    #     events -= events[0]  # now frame #s correspond to frames in preprocessed video clips
    #     images, labels = [], []
    #     cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

    #     if self.train:
    #         # random starting position, sample 'seq_length' frames
    #         start_frame = np.random.randint(events[-1] + 1)  ##終了点フレーム＋１までの中でrandomな整数
    #         cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  ##Ture or False (start_frameから再生)
    #         pos = start_frame
            
    #         while len(images) < self.seq_length:
    #             ret, img = cap.read()
    #             if ret:  ##True 
    #                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #                 images.append(img)
    #                 if pos in events[1:-1]:  ##start_frameがAddless~Finishのいずれかのフレームと一致する場合
    #                     labels.append(np.where(events[1:-1] == pos)[0][0])  ##(0~7)
    #                 else:  
    #                     labels.append(8)  ##No_event=8
    #                 pos += 1
    #             else:  ##False
    #                 cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  ##0から再生
    #                 pos = 0
    #         cap.release()
    #     else:  ##eval
    #         # full clip
    #         for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):  ##総フレーム数
    #             _, img = cap.read()
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             images.append(img)
    #             if pos in events[1:-1]:  ##posがAddless~Finishのいずれかのフレームと一致する場合
    #                 labels.append(np.where(events[1:-1] == pos)[0][0])
    #             else:
    #                 labels.append(8)
    #         cap.release()

    #     sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
    #     if self.transform:
    #         sample = self.transform(sample)
    #     return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images = images.transpose((0, 3, 1, 2))
        return {'images': torch.from_numpy(images).float().div(255.),
                'labels': torch.from_numpy(labels).long()}


class Normalize(object):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

    def __call__(self, sample):
        images, labels = sample['images'], sample['labels']
        images.sub_(self.mean[None, :, None, None]).div_(self.std[None, :, None, None])
        return {'images': images, 'labels': labels}


if __name__ == '__main__':

    norm = Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # ImageNet 平均 and 標準偏差 (RGB)で正規化

    dataset = StsqDB(data_file='train_split_1.pkl',
                     vid_dir='data/videos_40/',
                     seq_length=300,
                     transform=transforms.Compose([ToTensor(), norm]),
                     train=False)
   

    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=6, drop_last=False) ##num_workers:６つ並行処理を行う 

    for i, sample in enumerate(data_loader):
        images, labels = sample['images'], sample['labels']
        events = np.where(labels.squeeze() < 12)[0]  ##np.where:labels.squeeze()<8のインデックスを取得
        print('{} events: {}'.format(len(events), events)) ##8つのフレーム番号
    



    
#########################################################
# class GolfDB(Dataset):
#     def __init__(self, data_file, vid_dir, seq_length, transform=None, train=True):
#         self.df = pd.read_pickle(data_file)
#         self.vid_dir = vid_dir
#         self.seq_length = seq_length
#         self.transform = transform
#         self.train = train

#     def __len__(self):
#         return len(self.df)

#     def __getitem__(self, idx):
#         a = self.df.loc[idx, :]  # annotation info
#         events = a['events'] 
#         events -= events[0]  # now frame #s correspond to frames in preprocessed video clips
#         images, labels = [], []
#         cap = cv2.VideoCapture(osp.join(self.vid_dir, '{}.mp4'.format(a['id'])))

#         if self.train:
#             # random starting position, sample 'seq_length' frames
#             start_frame = np.random.randint(events[-1] + 1)  ##終了点フレーム＋１までの中でrandomな整数
#             cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  ##Ture or False (start_frameから再生)
#             pos = start_frame
            
#             while len(images) < self.seq_length:
#                 ret, img = cap.read()
#                 if ret:  ##True 
#                     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                     images.append(img)
#                     if pos in events[1:-1]:  ##start_frameがAddless~Finishのいずれかのフレームと一致する場合
#                         labels.append(np.where(events[1:-1] == pos)[0][0])  ##(0~7)
#                     else:  
#                         labels.append(8)  ##No_event=8
#                     pos += 1
#                 else:  ##False
#                     cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  ##0から再生
#                     pos = 0
#             cap.release()
#         else:  ##eval
#             # full clip
#             for pos in range(int(cap.get(cv2.CAP_PROP_FRAME_COUNT))):  ##総フレーム数
#                 _, img = cap.read()
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 images.append(img)
#                 if pos in events[1:-1]:  ##posがAddless~Finishのいずれかのフレームと一致する場合
#                     labels.append(np.where(events[1:-1] == pos)[0][0])
#                 else:
#                     labels.append(8)
#             cap.release()

#         sample = {'images':np.asarray(images), 'labels':np.asarray(labels)}
#         if self.transform:
#             sample = self.transform(sample)
#         return sample
# ###################################################################################




       

