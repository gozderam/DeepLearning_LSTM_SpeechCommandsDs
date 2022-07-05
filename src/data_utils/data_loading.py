import os
import data_utils.data_preparation as data_prep
from data_utils.soundds import SoundDS, SoundTestDS
import torch
import data_utils.classes_labels as cl


def load_train_data(batch_size, data_dir, all=True, second=False):
    
    audio_dir = f'{data_dir}/train/audio'
    
    if not os.path.isdir(audio_dir):
        data_prep.extract_data(data_dir)
        data_prep.prepare_silence_audio(data_dir)
        data_prep.prepare_train_files_list(data_dir)

    if all:
        trainset = SoundDS(f'{data_dir}/train/train_list.txt', audio_dir, cl.class_to_label_number)
    elif not second:
        trainset = SoundDS(f'{data_dir}/train/train_list.txt', audio_dir, cl.class_to_label_number1)
    else:
        trainset = SoundDS(f'{data_dir}/train/train_list2.txt', audio_dir, cl.class_to_label_number2)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)

    if all:
        valset = SoundDS(f'{data_dir}/train/validation_list.txt', audio_dir, cl.class_to_label_number)
    elif not second:
        valset = SoundDS(f'{data_dir}/train/validation_list.txt', audio_dir, cl.class_to_label_number1)
    else:
        valset = SoundDS(f'{data_dir}/train/validation_list2.txt', audio_dir, cl.class_to_label_number2)

    valloader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=0)

    return trainloader, valloader


def load_test_data(batch_size, data_dir):

    audio_dir = f'{data_dir}/test/audio'
    
    if not os.path.isdir(audio_dir):
        data_prep.extract_test_data(data_dir)
        data_prep.prepare_test_files_list(data_dir)

    testset = SoundTestDS(f'{data_dir}/test/test_list.txt', audio_dir)

    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0), testset.number_of_audio


def load_validation_as_test_data(batch_size, data_dir):

    audio_dir = f'{data_dir}/train/audio'

    if not os.path.isdir(audio_dir):
        data_prep.extract_data(data_dir)
        data_prep.prepare_silence_audio(data_dir)
        data_prep.prepare_train_files_list(data_dir)
    
    testset = SoundTestDS(f'{data_dir}/train/validation_list.txt', audio_dir)

    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0), testset.number_of_audio