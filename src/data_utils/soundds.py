import pandas as pd
from torch.utils.data import Dataset
from data_utils.audio_util import AudioUtil


class SoundDS(Dataset):
  def __init__(self, filename, data_path, dictionary):
    self.df = pd.read_csv(filename, header=None, squeeze=True)
    self.data_path = str(data_path)
    self.dictionary = dictionary
    self.duration = 1000
    self.sr = 44100

  def __len__(self):
    return len(self.df)    
    

  def __getitem__(self, idx):
    audio_file = self.df.iloc[idx]
    audio_file_full = f'{self.data_path}/{audio_file}'

    class_id = self.dictionary[audio_file.rsplit('/', 1)[0]]

    aud = AudioUtil.open(audio_file_full)    
    aud = AudioUtil.resample(aud, self.sr) 
    aud = AudioUtil.pad_trunc(aud, self.duration)
    aud = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)

    return aud, class_id


class SoundTestDS(Dataset):
  def __init__(self, filename, data_path):
    self.df = pd.read_csv(filename, header=None, squeeze=True)
    self.number_of_audio = len(self.df)
    self.data_path = str(data_path)
    self.duration = 1000
    self.sr = 44100

  def __len__(self):
    return len(self.df)    


  def __getitem__(self, idx):
    audio_file = self.df.iloc[idx]
    audio_file_full = f'{self.data_path}/{audio_file}'

    aud = AudioUtil.open(audio_file_full)    
    aud = AudioUtil.resample(aud, self.sr) 
    aud = AudioUtil.pad_trunc(aud, self.duration)
    aud = AudioUtil.spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None)

    return [aud, audio_file]