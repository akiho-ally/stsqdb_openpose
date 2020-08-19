import pickle
import matplotlib.pyplot as plt
# from sklearn.feature_extraction.text import TfidfVectorizer
# from dataloader import StsqDB, Normalize, ToTensor
import cv2
import numpy as np
import pandas as pd

import argparse




parser = argparse.ArgumentParser()
parser.add_argument('--split', default=1)
parser.add_argument('--iteration', default=8000)
parser.add_argument('--it_save', default=100)
parser.add_argument('--batch_size', default=8)
parser.add_argument('--seq_length', default=300) 
parser.add_argument('--use_no_element', action='store_true') 
args = parser.parse_args() 


# with open('data/no_ele/seq_length_{}/train_split_{}.pkl'.format(args.seq_length, args.split), "rb") as f:  
#     data = pickle.load(f)
# images = [ pair[0] for pair in data ]  ##len(images)=300
# labels = [ pair[1] for pair in data ] 
# images = cv2.cvtColor(images, cv2.COLOR_BGR2GRAY)
# import pdb; pdb.set_trace()
# images = np.array(images)
# labels = np.array(labels)


# images = images.astype(np.float32) / 255


# from sklearn.manifold import TSNE
# images2d = TSNE(n_components=2).fit_transform(images)


with open("annotationed_movie.pkl", "rb") as annotationed_movie:
    movie_dic = pickle.load(annotationed_movie)



parser = argparse.ArgumentParser()
parser.add_argument('--seq_length', default=300)
parser.add_argument('--img_size', default=224) 
args = parser.parse_args() 

# joint_lists = {}
images = []
labels = []
for mid, frames in movie_dic.items(): 
    split_id = np.random.randint(1, 5)
    for frame in frames:
        filename = frame[0]
        label_id = frame[1]
        filepath = "/home/akiho/projects/golfdb/data/videos_40/img" +str( mid )+ '/' + filename
        oriImg = cv2.imread(filepath)  # B,G,Rの順番
        im_gray = cv2.cvtColor(oriImg, cv2.COLOR_BGR2GRAY)

        # 画像の前処理
        img = im_gray.astype(np.float32) / 255

        images.append(img)
        labels.append(label_id)



    import pdb; pdb.set_trace()