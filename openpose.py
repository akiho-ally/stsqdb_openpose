import pickle
from PIL import Image
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
## % matplotlib inline

import os
import torch

from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

# 学習済みモデルと本章のモデルでネットワークの層の名前が違うので、対応させてロードする
# モデルの定義
net = OpenPoseNet()

# 学習済みパラメータをロードする
net_weights = torch.load(
    './weights/pose_model_scratch.pth', map_location={'cuda:0': 'cpu'})
keys = list(net_weights.keys())

weights_load = {}

# ロードした内容を、本書で構築したモデルの
# パラメータ名net.state_dict().keys()にコピーする
for i in range(len(keys)):
    weights_load[list(net.state_dict().keys())[i]
                 ] = net_weights[list(keys)[i]]

# コピーした内容をモデルに与える
state = net.state_dict()
state.update(weights_load)
net.load_state_dict(state)

print('ネットワーク設定完了：学習済みの重みをロードしました')

###########################################################
joint_lists = {}
path = './data/'
files = os.listdir(path)
for file in files:

    oriImg = cv2.imread(path + file)  # B,G,Rの順番

    # BGRをRGBにして表示
    oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)
    plt.imshow(oriImg)
    # plt.show()

    # 画像のリサイズ
    size = (368, 368)
    img = cv2.resize(oriImg, size, interpolation=cv2.INTER_CUBIC)

    # 画像の前処理
    img = img.astype(np.float32) / 255.

    # 色情報の標準化
    color_mean = [0.485, 0.456, 0.406]
    color_std = [0.229, 0.224, 0.225]

    preprocessed_img = img.copy()[:, :, ::-1]  # BGR→RGB

    for i in range(3):
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] - color_mean[i]
        preprocessed_img[:, :, i] = preprocessed_img[:, :, i] / color_std[i]

    # （高さ、幅、色）→（色、高さ、幅）
    img = preprocessed_img.transpose((2, 0, 1)).astype(np.float32)

    # 画像をTensorに
    img = torch.from_numpy(img)

    # ミニバッチ化：torch.Size([1, 3, 368, 368])
    x = img.unsqueeze(0)

    #######################################################################

    net.eval()
    predicted_outputs, _ = net(x)

    # 画像をテンソルからNumPyに変化し、サイズを戻します
    pafs = predicted_outputs[0][0].detach().numpy().transpose(1, 2, 0)
    heatmaps = predicted_outputs[1][0].detach().numpy().transpose(1, 2, 0)

    pafs = cv2.resize(pafs, size, interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(heatmaps, size, interpolation=cv2.INTER_CUBIC)


    pafs = cv2.resize(
        pafs, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)
    heatmaps = cv2.resize(
        heatmaps, (oriImg.shape[1], oriImg.shape[0]), interpolation=cv2.INTER_CUBIC)

    ########################################################################
    #joint_listがそれぞれの関節:joint_list.shape=(検出できた関節数,5)
    _, result_img, joint_list, person_to_joint_assoc = decode_pose(oriImg, heatmaps, pafs)
    # 結果を描画

    # result_img = cv2.resize(result_img, (224,224), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(result_img)
    # plt.imsave('./op_data/' + file, result_img)
    joint_lists[file] = joint_list

with open("joint_lists.pkl","wb") as f:
    pickle.dump(joint_lists, f)




