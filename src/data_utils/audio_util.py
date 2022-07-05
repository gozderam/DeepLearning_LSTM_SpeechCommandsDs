# source: https://towardsdatascience.com/audio-deep-learning-made-simple-sound-classification-step-by-step-cebc936bbe5
import random
import numpy as np
import torch
import torchaudio
from torchaudio import transforms


class AudioUtil():
  @staticmethod
  def open(audio_file):
    sig, sr = torchaudio.load(audio_file)
    return (sig, sr)


  @staticmethod
  def cut(aud, ms):
    sig, sr = aud
    max_len = sr // 1000 * ms
    sig_len = sig.shape[1]
    count = sig_len // max_len
    split = np.hsplit(sig[:, :max_len * count], count)
    if sig_len > max_len * count:
      split.append(sig[:, max_len * count + 1:])
    return (split, sr)


  @staticmethod
  def save(aud, file_name):
    sig, sr = aud
    torchaudio.save(file_name, sig, sr, format='wav')


  @staticmethod
  def resample(aud, newsr):
    sig, sr = aud

    if (sr == newsr):
      return aud

    num_channels = sig.shape[0]

    resig = transforms.Resample(sr, newsr)(sig[:1,:])
    if (num_channels > 1):
      retwo = transforms.Resample(sr, newsr)(sig[1:,:])
      resig = torch.cat([resig, retwo])

    return ((resig, newsr))


  @staticmethod
  def pad_trunc(aud, max_ms):
    sig, sr = aud
    num_rows, sig_len = sig.shape
    max_len = sr // 1000 * max_ms

    if (sig_len > max_len):
      sig = sig[:,:max_len]

    elif (sig_len < max_len):
      pad_begin_len = random.randint(0, max_len - sig_len)
      pad_end_len = max_len - sig_len - pad_begin_len

      pad_begin = torch.zeros((num_rows, pad_begin_len))
      pad_end = torch.zeros((num_rows, pad_end_len))

      sig = torch.cat((pad_begin, sig, pad_end), 1)
      
    return (sig, sr)


  @staticmethod
  def spectro_gram(aud, n_mels=64, n_fft=1024, hop_len=None):
    sig,sr = aud
    top_db = 80

    spec = transforms.MelSpectrogram(sr, n_fft=n_fft, hop_length=hop_len, n_mels=n_mels)(sig)

    spec = transforms.AmplitudeToDB(top_db=top_db)(spec)
    return (spec)