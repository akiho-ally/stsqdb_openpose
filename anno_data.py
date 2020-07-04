import os
import pandas as pd
import pickle

from PIL import Image
import numpy as np

element_names = ['Bracket', 'Change_edge', 'Chasse', 'Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','No_element']

file_dict = {}

for element_name in element_names:
    # 'Bracket', 'Change_edge'
    #dir_path = '/Users/akiho/projects/d-hacks/skate/dataset/train/' + element_name
    dir_path = '/home/akiho/projects/golfdb/data/dataset/train_all/' + element_name
    # '/Users/akiho/projects/d-hacks/skate/dataset/train/Bracket')
    files = sorted(os.listdir(dir_path))
    # [file, file, file, ..., .DS_store]
    for file in files:
        if not file.startswith('.'):
            file_dict[file] = element_name

# #保存
# pd.to_pickle(file_dict, "anno_data.pkl")
with open("anno_data.pkl", "wb") as anno_data:
    pickle.dump(file_dict, anno_data) 

# #読み出し
# hoge = pd.read_pickle("anno_data.pkl") 






## ここから下は別ファイルに切り取ってもいいかもしれない
# TODO: filepathを見て、fire_dictの中身を動画ごとに分割
movie_dic = {}
# 
# ↓
# img_tensor: Tensor[(xxxx.pngのtensor, xxxx.pngのtensor, xxxx.pngのtensor]), label_tensor([1, 2, 3, 3, 5, ...])
# この時点でエレメントはlabel_id化してしまっていいと思う
movie_id = 0
frame = ""
for filepath, element_label in file_dict.items():
    # filepath: "img22_576.jpg"とかになってるはず
    mid = filepath.split("_")[0].replace("img", "")
    frame_id = filepath.split("_")[1].strip()
    if mid == "" :
        mid = 1
    else:
         mid = int(mid)
    label_id = element_names.index(element_label)
    if mid in movie_dic.keys():
        movie_dic[mid].append((filepath, label_id, frame_id))
    else:
        movie_dic[mid] = [(filepath, label_id, frame_id)]

for mid, frames in movie_dic.items():
    movie_dic[mid] = sorted(frames, key=lambda x:x[2])

with open("annotationed_movie.pkl", "wb") as annotationed_movie:
    pickle.dump(movie_dic, annotationed_movie) 
# pd.to_pickle(movie_dic, "annotationed_movie.pkl")
# hoge2 = pd.read_pickle("annotationed_movie.pkl") 
#print(hoge2)
# TODO: bounding box, player等補助情報が増えた場合はdfを作成。それに伴いdataloader.pyも変更


# # TODO: dataloaderの形に合わせて保存
# # movie_dict = {0: [(filename, label_id, frame_id), (), (), ]}
# # ↓
# # data = [ [[[]], [[]]] , [1,2, 4,0, ],   [[[]], [[]]] , [1,2, 4,0, ],   [[[]], [[]]] , [1,2, 4,0, ] ....  ] 

# data = []
# for mid, frames in movie_dic.items():  ##frames:(filename, label_id, frame_id)
#     images = []
#     labels = []

#     for frame in frames:
#         filename = frame[0]
#         label_id = frame[1]

#         filepath = "data/videos_40/img" +str( mid )+ '/' + filename
#         img = np.array(Image.open(filepath)) # TODO: filenameから画像データの値を呼び出し
#         img_resize = img.resize((224, 224))
#         images.append(img)
#         labels.append(label_id)
#     data.append(images)
#     data.append(labels)

# # pd.to_pickle(data, "train.pkl")
# with open("train.pkl", "wb") as train:
#     pickle.dump(data, train) 
# import pdb; pdb.set_trace()
# print(data)





# bracket_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Bracket'))
# change_edge_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Change_edge'))
# chasse_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Chasse'))
# choctaw_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Choctaw'))
# counter_turn_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Counter_turn'))
# cross_roll_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Cross_roll'))
# loop_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Loop'))
# mohawk_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Mohawk'))
# rocker_turn_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Rocker_turn'))
# three_turn_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Three_turn'))
# toe_step_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Toe_step'))
# twizzle_files = sorted(os.listdir('/Users/akiho/projects/d-hacks/skate/dataset/train/Twizzle'))


# bracket_dict = {file : 'Bracket' for file in bracket_files }
# change_edge_dict = {file : 'Change_edge' for file in change_edge_files }
# chasse_dict = {file : 'Chasse' for file in chasse_files }
# choctaw_dict = {file : 'Choctaw' for file in choctaw_files }
# counter_turn_dict = {file : 'Counter_turn' for file in counter_turn_files }
# cross_roll_dict = {file : 'Cross_roll' for file in cross_roll_files }
# loop_dict = {file : 'Loop' for file in loop_files }
# mohawk_dict = {file : 'Mohawk' for file in mohawk_files }
# rocker_turn_dict = {file : 'Rocker_turn' for file in rocker_turn_files }
# three_turn_dict = {file : 'Three_turn' for file in three_turn_files }
# toe_step_dict = {file : 'Toe_step' for file in toe_step_files }
# twizzle_dict = {file : 'Twizzle' for file in twizzle_files }


# file_dict.update(**file_dict, **choctaw_dict, **bracket_dict)


