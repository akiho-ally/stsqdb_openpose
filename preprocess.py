import pickle
import numpy as np
import cv2
from PIL import Image
from PIL import ImageEnhance
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq_length', default=300)
    parser.add_argument('--img_size', default=224) 
    args = parser.parse_args() 

    # 1. movie_dicの読み込み
    with open("annotationed_movie.pkl", "rb") as annotationed_movie:
        movie_dic = pickle.load(annotationed_movie)

        # 1つのmovie_data = (images, labels)
        # data = [(images, labels), (images, labels), ..., (images, labels)]

    # 2. 前処理
    data = []
    for mid, frames in movie_dic.items():  ##frames:(filename, label_id, frame_id)
        images = []
        labels = []
        split_id = np.random.randint(1, 5)
        ##############リサイズ
        for frame in frames:
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            img = Image.open(filepath)
            img_resize = np.array(img.resize((args.img_size, args.img_size)))
            images.append(img_resize)
            labels.append(label_id)
            
            ##len(images):1467

        index = 0
        for i in range(len(images)):
            if index+int(args.seq_length) <=len(images):
                split_image = images[index : index+int(args.seq_length)]
                split_label = labels[index : index+int(args.seq_length)]
            else:
                break
            data.append((split_image, split_label, split_id))  ##split_id = 3
            index += 10

        # import pdb; pdb.set_trace()  ##len(data)=117
        
        ################反転処理
        fliped_images = []
        fliped_labels = []
        for frame in frames:  
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            img = Image.open(filepath)
            img_resize = np.array(img.resize((args.img_size, args.img_size)))
            img_fliped = np.array(cv2.flip(img_resize, 1)) 
            fliped_images.append(img_fliped)
            fliped_labels.append(label_id)


        
        index = 0

        for i in range(len(fliped_images)):
            if index+int(args.seq_length) <=len(images):
                split_fliped_image = fliped_images[index : index+int(args.seq_length)]
                split_fliped_label = fliped_labels[index : index+int(args.seq_length)]
            else:
                break
            data.append((split_fliped_image, split_fliped_label, split_id))  
            index += 10

        # import pdb; pdb.set_trace() ##len(data) = 234

        ##############色補正
        bgr_images = []
        bgr_labels = []
        for frame in frames:  
            filename = frame[0]
            label_id = frame[1]
            filepath = "data/videos_40/img" +str( mid )+ '/' + filename
            img = Image.open(filepath)
            img_resize = np.array(img.resize((args.img_size, args.img_size)))
            img_hsv = cv2.cvtColor(img_resize,cv2.COLOR_BGR2HSV)
            img_hsv[:,:,(1)] = img_hsv[:,:,(1)]*0.5
            img_bgr = cv2.cvtColor(img_hsv,cv2.COLOR_HSV2BGR)
            bgr_images.append(img_bgr)
            bgr_labels.append(label_id)




        index = 0
        for i in range(len(bgr_images)):
            if index+int(args.seq_length) <=len(images):
                split_bgr_image = bgr_images[index : index+int(args.seq_length)]
                split_bgr_label = bgr_labels[index : index+int(args.seq_length)]
            else:
                break
            data.append((split_bgr_image, split_bgr_label, split_id))
            index += 10
        #import pdb; pdb.set_trace()  ##len(data)=351



        # TODO: 系列長について、padding or 長い動画の分割
            # data.append((split_images, split_labels, split_id))  
        print('movie {}  len(data) = {} '.format(mid, len(data)) )

    # import pdb; pdb.set_trace() ##len(data)=10197


    
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
        # TODO: movieをシャッフル
  
        with open("data/seq_length_{}/val_split_{:1d}.pkl".format(args.seq_length, i), "wb") as f:
            pickle.dump(val_split, f)
        with open("data/seq_length_{}/train_split_{:1d}.pkl".format(args.seq_length, i), "wb") as f:
            pickle.dump(train_split, f)
        print("finish {}".format(i))

    print(data[0])
    
    # TODO: dataset 情報を表示


if __name__ == "__main__":
    main()