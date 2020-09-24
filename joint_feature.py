import pickle
from PIL import Image
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

    with open("/home/akiho/projects/StSqDB/anno_data_train.pkl", "rb") as annotationed_train:
        anno_element_train = pickle.load(annotationed_train) 
    
    with open("/home/akiho/projects/StSqDB/anno_data_eval.pkl", "rb") as annotationed_eval:
        anno_element_eval = pickle.load(annotationed_eval) 


    element_names = ['Bracket', 'Change_edge', 'Chasse', 'Choctaw', 'Counter_turn', 'Cross_roll', 'Loop', 'Mohawk', 'Rocker_turn', 'Three_turn', 'Toe_step', 'Twizzle','No_element']


    train_data = []
    coordinates = []
    labels = []
    for file_name, element_name in anno_element_train.items():
        label_id = element_names.index(element_name)
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        oriImg = cv2.imread(filepath)  # B,G,Rの順番

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
        #joint_listがそれぞれの関節:joint_list [6.81000000e+02, 3.26000000e+02, 1.57175213e-01, 0.00000000e+00,0.00000000e+00] →[x座標,y座標,確率,index,関節番号]
        #person_to_joint_assocが各個人の関節:[0~17はjoint_listのindex番号, スコア, 検出できた関節の数]
        _, _, joint_list, person_to_joint_assoc = decode_pose(oriImg, heatmaps, pafs)


        if joint_list.ndim == 1 or person_to_joint_assoc.ndim == 1:
            continue
        else:
            one_person_to_joint_index = np.delete(person_to_joint_assoc[0], [18,19]) ##一人の関節（joint_listのindexが並んでる）
            joint_list = np.delete(joint_list, [2,3,4] , 1)  ##全ての関節の座標のみのリスト
            # TODO : 上半身と下半身で分ける
            one_person_to_joint_index = one_person_to_joint_index[8:14]  ##下半身
            # one_person_to_joint_index = np.delete(one_person_to_joint_index, [8,9,10,11,12,13]) ##上半身

            each_coordinates = []
            for i, v in enumerate(one_person_to_joint_index):

                if one_person_to_joint_index[i] == -1:
                    each_joint_coordinate = [0,0]  

                else:
                    each_joint_coordinate = joint_list[int(v)] ##[indexで座標を取得]
                    each_joint_coordinate = each_joint_coordinate.tolist()
                each_coordinates.append(each_joint_coordinate)
            coordinates.append(each_coordinates)
            labels.append(label_id)
            ##[[[716.0, 234.0], [723.0, 266.0], [685.0, 267.0], [677.0, 314.0], [679.0, 355.0], [762.0, 263.0], [793.0, 276.0], [0, 0], [707.0, 365.0], [684.0, 443.0], [696.0, 528.0], [759.0, 363.0], [764.0, 447.0], [764.0, 521.0], [706.0, 228.0], [722.0, 227.0], [696.0, 235.0], [737.0, 230.0], [682.0, 260.0], [665.0, 296.0], [629.0, 299.0], [607.0, 350.0], [604.0, 399.0], [703.0, 293.0], [734.0, 333.0], [779.0, 353.0], [652.0, 399.0], [662.0, 481.0], [606.0, 546.0], [698.0, 398.0], [705.0, 483.0], [640.0, 534.0], [674.0, 256.0], [687.0, 254.0], [652.0, 260.0], [0, 0]], [[716.0, 234.0], [723.0, 266.0], [685.0, 267.0], [677.0, 314.0], [679.0, 355.0], [762.0, 263.0], [793.0, 276.0], [0, 0], [707.0, 365.0], [684.0, 443.0], [696.0, 528.0], [759.0, 363.0], [764.0, 447.0], [764.0, 521.0], [706.0, 228.0], [722.0, 227.0], [696.0, 235.0], [737.0, 230.0], [682.0, 260.0], [665.0, 296.0], [629.0, 299.0], [607.0, 350.0], [604.0, 399.0], [703.0, 293.0], [734.0, 333.0], [779.0, 353.0], [652.0, 399.0], [662.0, 481.0], [606.0, 546.0], [698.0, 398.0], [705.0, 483.0], [640.0, 534.0], [674.0, 256.0], [687.0, 254.0], [652.0, 260.0], [0, 0]]]
            ##len()2

    sum_coordinates = 18 * len(coordinates)

    print(len(coordinates))
    index = 0
    for i in range(len(coordinates)):
        if index+int(args.seq_length) <=len(coordinates):
            split_coordinates = coordinates[index : index+int(args.seq_length)]
            split_label = labels[index : index+int(args.seq_length)]
        else:
            break
        train_data.append((split_coordinates, split_label))  ##split_id = 3
        index += 30
    
    print(len(train_data))




    


        # '''
        # array([[681., 326.,   0.,   0.],
        #     [628., 346.,   1.,   1.],
        #     [644., 349.,   2.,   2.],
        #     [669., 297.,   3.,   3.],
        #     [630., 410.,   4.,   3.],
        #     [671., 284.,   5.,   4.],
        #     [676., 445.,   6.,   4.],
        #     [608., 344.,   7.,   5.],
        #     [654., 305.,   8.,   6.],
        #     [592., 423.,   9.,   8.],
        #     [631., 476.,  10.,   9.],
        #     [631., 476.,  11.,   9.],
        #     [604., 535.,  12.,  10.],
        #     [566., 422.,  13.,  11.],
        #     [632., 472.,  14.,  12.],
        #     [625., 536.,  15.,  13.],
        #     [681., 320.,  16.,  14.],
        #     [655., 318.,  17.,  16.],
        #     [625., 314.,  18.,  17.]])
        # array([[ 0.        ,  1.        ,  2.        ,  4.        ,  6.        ,
        #  7.        , -1.        , -1.        ,  9.        , 10.        ,
        # 12.        , 13.        , 14.        , 15.        , 16.        ,
        # -1.        , 17.        , 18.        , 16.74321419, 15.        ]])
        # '''
        # joint_lists[mid] = [(filename, label_id, person_joint, split_id)]
    with open("data/same_frames/train_split_1.pkl", "wb") as f:
        pickle.dump(train_data, f)
    
##############################################################################


    eval_data = []
    coordinates = []
    labels = []
    for file_name, element_name in anno_element_eval.items():
        label_id = element_names.index(element_name)
        filepath = '/home/akiho/projects/StSqDB/data/dataset/train_all/' + element_name + '/' + file_name
        oriImg = cv2.imread(filepath)  # B,G,Rの順番

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
        #joint_listがそれぞれの関節:joint_list [6.81000000e+02, 3.26000000e+02, 1.57175213e-01, 0.00000000e+00,0.00000000e+00] →[x座標,y座標,確率,index,関節番号]
        #person_to_joint_assocが各個人の関節:[0~17はjoint_listのindex番号, スコア, 検出できた関節の数]
        _, _, joint_list, person_to_joint_assoc = decode_pose(oriImg, heatmaps, pafs)


        if joint_list.ndim == 1 or person_to_joint_assoc.ndim == 1:
            continue
        else:
            one_person_to_joint_index = np.delete(person_to_joint_assoc[0], [18,19]) ##一人の関節（joint_listのindexが並んでる）
            joint_list = np.delete(joint_list, [2,3,4] , 1)  ##全ての関節の座標のみのリスト
            # TODO : 上半身と下半身で分ける
            one_person_to_joint_index = one_person_to_joint_index[8:14]  ##下半身
            # one_person_to_joint_index = np.delete(one_person_to_joint_index, [8,9,10,11,12,13]) ##上半身

            each_coordinates = []
            for i, v in enumerate(one_person_to_joint_index):

                if one_person_to_joint_index[i] == -1:
                    each_joint_coordinate = [0,0]  

                else:
                    each_joint_coordinate = joint_list[int(v)] ##[indexで座標を取得]
                    each_joint_coordinate = each_joint_coordinate.tolist()
                each_coordinates.append(each_joint_coordinate)
            coordinates.append(each_coordinates)
            labels.append(label_id)
            ##[[[716.0, 234.0], [723.0, 266.0], [685.0, 267.0], [677.0, 314.0], [679.0, 355.0], [762.0, 263.0], [793.0, 276.0], [0, 0], [707.0, 365.0], [684.0, 443.0], [696.0, 528.0], [759.0, 363.0], [764.0, 447.0], [764.0, 521.0], [706.0, 228.0], [722.0, 227.0], [696.0, 235.0], [737.0, 230.0], [682.0, 260.0], [665.0, 296.0], [629.0, 299.0], [607.0, 350.0], [604.0, 399.0], [703.0, 293.0], [734.0, 333.0], [779.0, 353.0], [652.0, 399.0], [662.0, 481.0], [606.0, 546.0], [698.0, 398.0], [705.0, 483.0], [640.0, 534.0], [674.0, 256.0], [687.0, 254.0], [652.0, 260.0], [0, 0]], [[716.0, 234.0], [723.0, 266.0], [685.0, 267.0], [677.0, 314.0], [679.0, 355.0], [762.0, 263.0], [793.0, 276.0], [0, 0], [707.0, 365.0], [684.0, 443.0], [696.0, 528.0], [759.0, 363.0], [764.0, 447.0], [764.0, 521.0], [706.0, 228.0], [722.0, 227.0], [696.0, 235.0], [737.0, 230.0], [682.0, 260.0], [665.0, 296.0], [629.0, 299.0], [607.0, 350.0], [604.0, 399.0], [703.0, 293.0], [734.0, 333.0], [779.0, 353.0], [652.0, 399.0], [662.0, 481.0], [606.0, 546.0], [698.0, 398.0], [705.0, 483.0], [640.0, 534.0], [674.0, 256.0], [687.0, 254.0], [652.0, 260.0], [0, 0]]]
            ##len()2

    sum_coordinates = 18 * len(coordinates)
    print(len(coordinates))
    index = 0
    for i in range(len(coordinates)):
        if index+int(args.seq_length) <=len(coordinates):
            split_coordinates = coordinates[index : index+int(args.seq_length)]
            split_label = labels[index : index+int(args.seq_length)]
        else:
            break
        eval_data.append((split_coordinates, split_label))  ##split_id = 3
        index += 30


    print(len(eval_data))

    


        # '''
        # array([[681., 326.,   0.,   0.],
        #     [628., 346.,   1.,   1.],
        #     [644., 349.,   2.,   2.],
        #     [669., 297.,   3.,   3.],
        #     [630., 410.,   4.,   3.],
        #     [671., 284.,   5.,   4.],
        #     [676., 445.,   6.,   4.],
        #     [608., 344.,   7.,   5.],
        #     [654., 305.,   8.,   6.],
        #     [592., 423.,   9.,   8.],
        #     [631., 476.,  10.,   9.],
        #     [631., 476.,  11.,   9.],
        #     [604., 535.,  12.,  10.],
        #     [566., 422.,  13.,  11.],
        #     [632., 472.,  14.,  12.],
        #     [625., 536.,  15.,  13.],
        #     [681., 320.,  16.,  14.],
        #     [655., 318.,  17.,  16.],
        #     [625., 314.,  18.,  17.]])
        # array([[ 0.        ,  1.        ,  2.        ,  4.        ,  6.        ,
        #  7.        , -1.        , -1.        ,  9.        , 10.        ,
        # 12.        , 13.        , 14.        , 15.        , 16.        ,
        # -1.        , 17.        , 18.        , 16.74321419, 15.        ]])
        # '''
        # joint_lists[mid] = [(filename, label_id, person_joint, split_id)]
    with open("data/same_frames/val_split_1.pkl", "wb") as f:
        pickle.dump(eval_data, f)


if __name__ == "__main__":
    main()