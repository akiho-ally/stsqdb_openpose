import pickle
from PIL import Image
from PIL import ImageEnhance
import cv2
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
## % matplotlib inline

import argparse
import os
import torch

from utils.openpose_net import OpenPoseNet
from utils.decode_pose import decode_pose

def main():
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

    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', default=300)
    parser.add_argument('--img_size', default=224) 
    args = parser.parse_args() 

    # 1. movie_dicの読み込み
    with open("annotationed_movie.pkl", "rb") as annotationed_movie:
        movie_dic = pickle.load(annotationed_movie)

    data = []
    for mid, frames in movie_dic.items(): 

###########################################################################リサイズ
        images = []
        labels = []
        split_id = np.random.randint(1, 5)
        for frame in frames:
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            oriImg = cv2.imread(filepath)  # B,G,Rの順番

            # BGRをRGBにして表示
            oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

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
            
            _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)
            # resize_img = np.array(result_img.resize((args.img_size, args.img_size)))
            resize_img = cv2.resize(result_img, (args.img_size,args.img_size), interpolation=cv2.INTER_CUBIC)
            images.append(resize_img)
            labels.append(label_id)

        index = 0
        for i in range(len(images)):
            if index+int(args.seq_length) <=len(images):
                split_image = images[index : index+int(args.seq_length)]
                split_label = labels[index : index+int(args.seq_length)]
            else:
                break
            data.append((split_image, split_label, split_id))  ##split_id = 3
            index += 10
# #######################################################################################################反転
        fliped_images = []
        fliped_labels = []
        for frame in frames:
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            oriImg = cv2.imread(filepath)  # B,G,Rの順番

            # BGRをRGBにして表示
            oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

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
            
            _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)
            resize_img = cv2.resize(result_img, (args.img_size,args.img_size), interpolation=cv2.INTER_CUBIC)

            img_fliped = np.array(cv2.flip(resize_img, 1)) 
            fliped_images.append(img_fliped)
            fliped_labels.append(label_id)

        index = 0
        for i in range(len(fliped_images)):
            if index+int(args.seq_length) <=len(images):
                split_image = fliped_images[index : index+int(args.seq_length)]
                split_label = fliped_labels[index : index+int(args.seq_length)]
            else:
                break
            data.append((split_image, split_label, split_id))  ##split_id = 3
            index += 10

 ############################################################################################色補正

        bgr_images = []
        bgr_labels = []
        for frame in frames:
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            oriImg = cv2.imread(filepath)  # B,G,Rの順番

            # BGRをRGBにして表示
            oriImg = cv2.cvtColor(oriImg, cv2.COLOR_BGR2RGB)

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
            
            _, result_img, _, _ = decode_pose(oriImg, heatmaps, pafs)
            resize_img = cv2.resize(result_img, (args.img_size,args.img_size), interpolation=cv2.INTER_CUBIC)
            img_hsv = cv2.cvtColor(resize_img,cv2.COLOR_BGR2HSV)
            img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
            img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
            bgr_images.append(img_bgr)
            bgr_labels.append(label_id)

        index = 0
        for i in range(len(bgr_images)):
            if index+int(args.seq_length) <=len(images):
                split_image = bgr_images[index : index+int(args.seq_length)]
                split_label = bgr_labels[index : index+int(args.seq_length)]
            else:
                break
            data.append((split_image, split_label, split_id))  ##split_id = 3
            index += 10
################################################################################################
        print('movie {}  len(data) = {} '.format(mid, len(data)) )



    if not os.path.exists('data/seq_length_{}'.format(args.seq_length)):
            os.mkdir('data/seq_length_{}'.format(args.seq_length))


    for i in range(1, 5):  
        ##評価
        images = []
        labels = []
        val_split = []
        train_split = []

        for movie_data in data:
            if movie_data[2] == i:
                val_split.append((movie_data[0], movie_data[1]))
            else:
                train_split.append((movie_data[0], movie_data[1]))


        with open("data/seq_length_{}/val_split_{:1d}.pkl".format(args.seq_length, i), "wb") as f:
            pickle.dump(val_split, f)
        with open("data/seq_length_{}/train_split_{:1d}.pkl".format(args.seq_length, i), "wb") as f:
            pickle.dump(train_split, f)
        print("finish {}".format(i))



if __name__ == "__main__":
    main()