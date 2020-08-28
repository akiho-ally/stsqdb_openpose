import pickle
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from matplotlib import pylab as plt
import argparse




# parser = argparse.ArgumentParser()
# parser.add_argument('--split', default=1)
# parser.add_argument('--iteration', default=8000)
# parser.add_argument('--it_save', default=100)
# parser.add_argument('--batch_size', default=8)
# parser.add_argument('--seq_length', default=300) 
# parser.add_argument('--use_no_element', action='store_true') 
# args = parser.parse_args() 


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




images = []
labels = []
for mid, frames in movie_dic.items(): 
    split_id = np.random.randint(1, 5)
    for frame in frames:
        filename = frame[0]
        label_id = frame[1]
        filepath = "/home/akiho/projects/golfdb/data/videos_40/img" +str( mid )+ '/' + filename
        oriImg = Image.open(filepath)
        img_resize = np.array(oriImg.resize((224,224)))
        # oriImg = cv2.imread(filepath)  # B,G,Rの順番
        im_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
        

        # 画像の前処理
        img = im_gray.astype(np.float32) / 255

        images.append(img)
        labels.append(label_id)
    print(mid)
    

images = np.array(images)
labels = np.array(labels)



images = images.reshape(len(images), len(images[0])*len(images[0][1]))
print(images.shape)

print('reshape完了')

images2d = TSNE(n_components=2).fit_transform(images)

elements = [
'Bracket', 
'Change_edge', 
'Chasse', 
'Choctaw', 
'Counter_turn', 
'Cross_roll', 
'Loop', 
'Mohawk', 
'Rocker_turn', 
'Three_turn', 
'Toe_step', 
'Twizzle',
'No_element'
]

f, ax = plt.subplots(1, 1, figsize=(20, 20))
for i in range(13):
    target = images2d[labels == i]
    ax.scatter(x=target[:1000, 0], y=target[:1000, 1], label=elements[i], alpha=0.5)
    print('label' + str(i))
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')


plt.show()

save_dir = '/home/akiho/projects/stsqdb_op/'
plt.savefig(save_dir + 't_sne_figure_13_224.png')







# fashion3d = TSNE(n_components=3).fit_transform(images)

# fig = plt.figure(figsize=(20, 20)).gca(projection='3d')
# for i in range(10):
#     target = fashion3d[labels == i]
#     fig.scatter(target[:500, 0], target[:500, 1], target[:500, 2], label=elements[i], alpha=0.5)
# fig.legend(bbox_to_anchor=(1.02, 0.7), loc='upper left')

# plt.show()

# save_dir = '/home/akiho/projects/stsqdb_op/'
# plt.savefig(save_dir + 't_sne_figure_13_224_3d.png')