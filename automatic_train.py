import argparse
import cv2
import os
import numpy as np
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping
import random

parser = argparse.ArgumentParser()
parser.add_argument("model", type=str, help="A path to a model that you want to fine-tune.")
parser.add_argument("data", type=str,
                    help="A directory with data. The directory must contain two folders train and test. Both these folders must contain two subfolders input and target.")
args = parser.parse_args()


def get_filenames(path):
    names = []
    for fname in os.listdir(path):
        names.append(fname)
    return names


def get_dataset(image_file, path, random_height, random_width):
    images = []
#    if path.endswith('input/'):
#        random_height = []
#        random_width = []
    for j, name in enumerate(image_file):
        file_name = path + name
        img = cv2.imread(file_name, flags=cv2.IMREAD_GRAYSCALE)
        img = np.atleast_3d(img)  # we need dim (128,128,1) not just (128,128)

        if path.endswith('train/input/'):
            random_height.append(random.randrange(img.shape[0] - 512))
            random_width.append(random.randrange(img.shape[1] - 512))

        if path.endswith('target/'):  # masks
            img = img / 255
        images.append(img)
        # print(random_height[j], random_width[j], name)
    if path.endswith('train/input/') or path.endswith('train/target/'):
        for j in range(len(images)):
            images[j] = images[j][random_height[j]:(random_height[j] + 512), random_width[j]:(random_width[j] + 512), :]
    return np.array(images), random_height, random_width


def generate(X, y, data_len, pathx, pathy, batchsize=1):
    """Generator"""
    while True:
        c = list(zip(X, y))
        random.shuffle(c)
        X, y = zip(*c)
        X = list(X)
        y = list(y)
        cuts = [(b, min(b + batchsize, data_len)) for b in range(0, data_len, batchsize)]
        for start, end in cuts:
            inputs = X[start:end].copy()
            inputs, h, w = get_dataset(inputs, pathx, [], [])
            targets = y[start:end].copy()
            targets, h, w = get_dataset(targets, pathy, h, w)
            yield (inputs, targets)


def main(model_path, data_path):
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    if model_path[-3:] != '.h5':
        print('Error: The path to a model does not end with .h5')
        return
    if os.path.isfile(model_path) is False:
        print('Error: The path to a model is not a file.')
        return
    X_trn_path = data_path + 'train' + '/input/'
    y_trn_path = data_path + 'train' + '/target/'
    X_val_path = data_path + 'test' + '/input/'
    y_val_path = data_path + 'test' + '/target/'
    if os.path.isdir(X_trn_path) is False or os.path.isdir(y_trn_path) is False or os.path.isdir(
            X_val_path) is False or os.path.isdir(y_val_path) is False:
        print('Error: Data directory does not contain required folders. Check help for the second parameter.')
        return
    X_trn = get_filenames(X_trn_path)
    y_trn = get_filenames(y_trn_path)
    X_val = get_filenames(X_val_path)
    y_val = get_filenames(y_val_path)
    X_trn.sort()
    y_trn.sort()
    X_val.sort()
    y_val.sort()
    model = load_model(model_path)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

    # Show architecture of the model:
    #model.summary()

    # Saving the model that behaves better in validation set:
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=50, verbose=1, restore_best_weights=True),
        # ModelCheckpoint('models/u_net.hdf5', monitor='val_loss', save_best_only=True, verbose=1)
    ]

    # Train model:
    size_train = len(X_trn)
    size_val = len(X_val)
    batch_size = 6

    history = model.fit(generate(X_trn, y_trn, size_train, X_trn_path, y_trn_path, batch_size),
                        steps_per_epoch=size_train / batch_size, epochs=600, verbose=1,
                        callbacks=callbacks,
                        validation_data=generate(X_val, y_val, size_val, X_val_path, y_val_path, 1),
                        validation_steps=size_val / 1)

    save_model_path = os.path.dirname(model_path)
    save_model_path = save_model_path + '/new_model.h5'
    model.save(save_model_path)
    print('Model saved at', save_model_path)
    print('Accuracy of the model on the validation set is', round(max(history.history['val_accuracy']), 3) * 100, '%' )


if __name__ == '__main__':
    main(args.model, args.data)
