import py7zr
import os
from data_utils.audio_util import AudioUtil
import numpy as np
from sklearn.model_selection import train_test_split
import data_utils.classes_labels as cl


def extract_data(directory):
    with py7zr.SevenZipFile(f'{directory}/train.7z', mode='r') as z:
        z.extractall(path=directory)

    os.remove(f'{directory}/train/testing_list.txt')
    os.remove(f'{directory}/train/audio/_background_noise_/README.md')


def extract_test_data(directory):
    with py7zr.SevenZipFile(f'{directory}/test.7z', mode='r') as z:
        z.extractall(path=directory)


def prepare_train_files_list(directory):
    path = f'{directory}/train/audio'
    
    with open(f'{directory}/train/validation_list.txt') as file:
        validation = file.read().splitlines()
    
    f = []
    f2 = []
    for (dirpath, dirnames, filenames) in os.walk(path):
        for name in filenames:
            s = str(os.path.join(os.path.relpath(dirpath, path), name)).replace('\\', '/')
            if s not in validation and not s.startswith('_background_noise_'):
                f.append(s)
            if s not in validation and os.path.relpath(dirpath, path) in cl.class_to_label_number2.keys():
                f2.append(s)

    f3 = []
    for file in validation:
        if file.split('/')[0] in cl.class_to_label_number2.keys():
            f3.append(file)
    
    with open(f'{directory}/train/train_list.txt', 'w') as txt_file:
        for file in f:
            txt_file.write(f'{file}\n')

    with open(f'{directory}/train/train_list2.txt', 'w') as txt_file:
        for file in f2:
            txt_file.write(f'{file}\n')

    with open(f'{directory}/train/validation_list2.txt', 'w') as txt_file:
        for file in f3:
            txt_file.write(f'{file}\n')


def prepare_test_files_list(directory):
    path = f'{directory}/test/audio'
    
    f = os.listdir(path)
    
    with open(f'{directory}/test/test_list.txt', 'w') as txt_file:
        for file in f:
            txt_file.write(f'{file}\n')


def prepare_silence_audio(directory):
    path_target = f'{directory}/train/audio/silence'
    path_source = f'{directory}/train/audio/_background_noise_'
    os.mkdir(path_target)
    for file in os.listdir(path_source):
        aud = AudioUtil.open(os.path.join(path_source, file))
        (split, sr) = AudioUtil.cut(aud, 1000)
        for i, sig in enumerate(split):
            AudioUtil.save((sig, sr), os.path.join(path_target, file[:-4] + str(i) + '.wav'))
    
    _, files = train_test_split(np.array(os.listdir(path_target)), test_size=0.1, random_state=7)

    with open(f'{directory}/train/validation_list.txt', "a") as validation_list:
        for file in files:
            validation_list.write(f'silence/{file}\n')