import os
import re
import numpy as np
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split


def consistent_split(sentences, emotions, ratio):
    dialogs_train, dialogs_test, emotions_train, emotions_test = [], [], [], []
    flat_sentences = flatten(sentences)
    flat_emotion = flatten(emotions)
    for i, (s, e) in enumerate(zip(flat_sentences, flat_emotion)):
        if i % 100 > ratio * 100:
            dialogs_train.append(s)
            emotions_train.append(e)
        else:
            dialogs_test.append(s)
            emotions_test.append(e)
    return dialogs_train, dialogs_test, emotions_train, emotions_test


def train_dev_test_split(data_choice, train_file, predefined=False, chunk_size=None, ratio=0.2, random=True,
                         categories=None):
    if not predefined:
        dialogs, emotions, emotion_labels, _ = data_choice(train_file, chunk_size=chunk_size, categories=categories)
        if random:
            if len(dialogs.shape) < 2:
                flat_dialog = dialogs
                flat_emotion = emotions
            else:
                flat_dialog = flatten(dialogs)
                flat_emotion = flatten(emotions)
            dialogs_train, dialogs_test, emotions_train, emotions_test = train_test_split(flat_dialog, flat_emotion,
                                                                                          test_size=ratio)
            dialogs_train, dialogs_dev, emotions_train, emotions_dev = train_test_split(dialogs_train, emotions_train,
                                                                                        test_size=ratio)
        else:
            dialogs_train, dialogs_test, emotions_train, emotions_test = consistent_split(dialogs, emotions, ratio)
            dialogs_train, dialogs_dev, emotions_train, emotions_dev = consistent_split(dialogs_train, emotions_train,
                                                                                        ratio)
    else:
        dialogs_train, emotions_train, emotion_labels, _ = data_choice(train_file, mode="train")
        dialogs_dev, emotions_dev, _, _ = data_choice(train_file, mode="val")
        dialogs_test, emotions_test, _, _ = data_choice(train_file, mode="test")
        dialogs_train, emotions_train, dialogs_dev, emotions_dev, dialogs_test, emotions_test = \
            flatten(dialogs_train), flatten(emotions_train), flatten(dialogs_dev), flatten(emotions_dev), \
            flatten(dialogs_test), flatten(emotions_test)

    return dialogs_train, dialogs_dev, dialogs_test, emotions_train, emotions_dev, emotions_test, emotion_labels


def flatten(array):
    if isinstance(array, list):
        if len(array) > 1 and isinstance(array[0], list):
            array = np.array([val for sublist in array for val in sublist])
        return array
    if len(array.shape) == 1:
        return array
    else:
        return np.array([val for sublist in array for val in sublist])


def unique_count_data(labels):
    return np.unique(flatten(labels), return_counts=True)


def compute_class_weights(labels, method):
    unique, counts = unique_count_data(labels)
    if method == 'log':
        return np.array([abs(np.log(x / sum(counts))) for x in counts])
    elif method == 'avg':
        return np.array([1 - (x / sum(counts)) for x in counts])
    elif method == 'scikit':
        return compute_class_weight('balanced', unique, flatten(labels))
    elif method == 'ratio':
        return np.array([sum(counts) / x for x in counts])
    elif method == 'effective':
        beta = (sum(counts) - 1) / sum(counts)
        return np.array([(1 - beta) / (1 - np.power(beta, c)) for c in counts])
    elif method == 'none':
        return np.array([1 for _ in range(len(unique))])
    else:
        raise ValueError(f'Unknown method for class weight calculation {method}')


def balance_data(inputs, labels):
    unique, counts = unique_count_data(labels)
    max_tenth_count = max([c for c in counts if c/sum(counts) < 0.1])
    lower_half = [u for u, c in zip(unique, counts) if c <= max_tenth_count]

    in_batches = []
    out_batches = []
    for inp, label in zip(inputs, labels):
        unique_labels = np.unique(label)
        if len([u for u in unique_labels if u in lower_half]) > 0:
            in_batches.append(inp)
            out_batches.append(label)
    return in_batches, out_batches
