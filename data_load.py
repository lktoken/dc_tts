# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata
import json

def load_vocab():
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def load_mandarin():
    # find all words
    transcript = os.path.join(hp.data, 'transcript/aishell_transcript_v0.8.txt')
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    charset = set()
    for line in lines:
        _, text = line.strip().split(' ', 1)
        for c in text:
            charset.add(c)
    charlist = ['E',] + list(charset)
    
    char2idx = {char: idx for idx, char in enumerate(charlist)}
    idx2char = {idx: char for idx, char in enumerate(charlist)}
    with open(hp.logdir + '/idx2char-json.txt', 'wt') as jsonfile:
        jsonfile.write(json.dumps(idx2char))
    return char2idx, idx2char

def mandarin_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents
    # '[^[\u4e00-\u9fa5]]'
    # text = re.sub("\W", " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def text_normalize(text):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def load_data(mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab()

    if mode=="train":
        if "LJ" in hp.data:
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text = line.strip().split("|")

                fpath = os.path.join(hp.data, "wavs", fname + ".wav")
                fpaths.append(fpath)

                text = text_normalize(text) + "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

            return fpaths, text_lengths, texts
        elif "aishell" in hp.data:
            fpaths, text_lengths, texts = [], [], []
            zhchar2idx, zhidx2char = load_mandarin()
            transcript = os.path.join(hp.data, 'transcript/aishell_transcript_v0.8.txt')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, text = line.strip().split(' ', 1)

                fpath = os.path.join(hp.data, "wav/train/", fname[6:11], fname + ".wav")
                fpaths.append(fpath)

                text = mandarin_normalize(text) + "E"  # E: EOS
                text = [zhchar2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())

            return fpaths, text_lengths, texts

        else: # nick or kate
            # Parse
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()
            for line in lines:
                fname, _, text, is_inside_quotes, duration = line.strip().split("|")
                duration = float(duration)
                if duration > 10. : continue

                fpath = os.path.join(hp.data, fname)
                fpaths.append(fpath)

                text += "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())


        return fpaths, text_lengths, texts

    else: # synthesize on unseen test text.
        if "aishell" in hp.data:
            # Parse
            zhidx2char = json.load(open(hp.logdir + '/idx2char-json.txt'))
            zhchar2idx = {char:idx for idx, char in zhidx2char.items()}

            lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
            sents = [mandarin_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [zhchar2idx[char] for char in sent]
            return texts
        else:
            # Parse
            lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
            sents = [text_normalize(line.split(" ", 1)[-1]).strip() + "E" for line in lines] # text normalization, E: EOS
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent]
            return texts

def get_batch():
    """Loads training data and put them in queues"""
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data() # list
        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = "mels/{}".format(fname.replace("wav", "npy"))
                mag = "mags/{}".format(fname.replace("wav", "npy"))
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.n_fft//2+1))

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.B,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=hp.B*4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch
