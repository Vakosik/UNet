import argparse
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import os
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str, help="An image file to segment. If a directory is passed, it segements all images in the directory.")
parser.add_argument("net", type=int, choices=['unet_QUANTA_SE', 'unet_ST_BSE', 'unet_ST_SE'], help="The neural network (trained model) to use.")
parser.add_argument("-netpath", type=str, default='', help="Directory with saved neural networks. If not passed, it works with the current working directory.")
args = parser.parse_args()


def get_filenames(path):
    names = []
    for fname in os.listdir(path):
        if fname[-4:] == '.tif':
            names.append(fname)
    return names


def segment(path, net, netpath):
    if os.path.isfile(path) is True and path[-4:] == '.tif':
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        img = np.atleast_3d(img)  # we need dim (128,128,1) not just (128,128)
        img = np.expand_dims(img, axis=0)
        if net == 0:
            model_path = netpath+'unet_QUANTA_SE.h5'
        elif net == 1:
            model_path = netpath+'unet_ST_BSE.h5'
        else:
            model_path = netpath+'unet_ST_SE.h5'
        if os.path.isfile(model_path) == False:
            print('There is no saved neural net in the directory')
            return
        model = load_model(model_path)
        prediction = model.predict(img)
        prediction2 = prediction[0, :, :, 0]
        prediction3 = np.where(prediction2 < 0.5, 0, 1) * 255
        save_name = path[:-4] + 'pred.tif'
        cv2.imwrite(save_name, prediction3)
        print('Segmentation saved at', save_name)
    else:
        file_names = get_filenames(path)
        if len(file_names) == 0:
            print('No tif images in the input directory')
            return
        if net == 0:
            model_path = netpath + 'unet_QUANTA_SE.h5'
        elif net == 1:
            model_path = netpath + 'unet_ST_BSE.h5'
        else:
            model_path = netpath + 'unet_ST_SE.h5'
        if os.path.isfile(model_path) == False:
            print('There is no saved neural net in the directory')
            return
        model = load_model(model_path)
        for file_name in file_names:
            img = cv2.imread(path+file_name, flags=cv2.IMREAD_GRAYSCALE)
            img = np.atleast_3d(img)  # we need dim (128,128,1) not just (128,128)
            img = np.expand_dims(img, axis=0)
            prediction = model.predict(img)
            prediction2 = prediction[0, :, :, 0]
            prediction3 = np.where(prediction2 < 0.5, 0, 1) * 255
            save_name = path + file_name[:-4] + 'pred.tif'
            print(path + file_name[:-4] + 'pred.tif')
            cv2.imwrite(save_name, prediction3)
            print('Segmentation saved at', save_name)

if __name__ == "__main__":
    segment(args.input, args.net, args.netpath)
