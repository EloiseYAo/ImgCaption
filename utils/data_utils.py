import sys
sys.path.append('..')

import os, json
import numpy as np
import h5py
import urllib.request, urllib.parse, os, tempfile
from urllib.error import URLError, HTTPError
import cv2
import matplotlib.pyplot as plt
import traceback

from utils.config import CONFIG


# **** ALL FUNCTONS IN THIS FILE ARE COPIED FROM CS231n, assignment 3. ****
# http://cs231n.stanford.edu/syllabus.html

def load_coco():
    path = CONFIG.COCO_PATH
    data = {}
    caption_file = os.path.join(path, 'coco2014_captions.h5')

    # print(caption_file)

    with h5py.File(caption_file, 'r') as f:
        for k, v in f.items():
            data[k] = np.asarray(v)

    dict_file = os.path.join(path, 'coco2014_vocab.json')
    with open(dict_file, 'r') as f:
        dict_data = json.load(f)
        for k, v in dict_data.items():
            data[k] = v

    train_url_file = os.path.join(path, 'train2014_urls.txt')
    with open(train_url_file, 'r') as f:
        train_urls = np.asarray([line.strip() for line in f])
    data['train_urls'] = train_urls

    val_url_file = os.path.join(path, 'val2014_urls.txt')
    with open(val_url_file, 'r') as f:
        val_urls = np.asarray([line.strip() for line in f])
    data['val_urls'] = val_urls
    return data

def image_from_url(url):
    """
    Read an image from a URL. Returns a numpy array with the pixel data.
    We write the image to a temporary file then read it back. Kinda gross.
    """
    # try:
    f = urllib.request.urlopen(url, timeout=30)
    fd, fname = tempfile.mkstemp()
    with open(fname, 'wb') as ff:
        ff.write(f.read())
    img = cv2.imread(fname)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)# .astype(np.float32)
    os.close(fd)
    os.remove(fname)
    return img

class COCODataset(object):
    def __init__(self):
        data_dict = load_coco()
        self.train_captions = data_dict['train_captions']
        self.train_img_idxs = data_dict['train_image_idxs']
        self.val_captions = data_dict['val_captions']
        self.val_img_idxs = data_dict['val_image_idxs']
        self.idx_to_word = data_dict['idx_to_word']
        self.word_to_idx = data_dict['word_to_idx']
        self.train_urls = data_dict['train_urls']
        self.val_urls = data_dict['val_urls']

        self.num_train = len(self.train_captions)
        self.vocab_size = len(self.word_to_idx)

        self.train_cursor = 0
        self.epoch = 0

    def get_num_epoch(self):
        return self.epoch + float(self.train_cursor) / self.num_train


    def load_batch(self, batch_size, kind='train'):
        imgs, captions = [], []
        num = 0
        while num < batch_size:
            if kind =='train':
                idx = np.random.randint(0, self.num_train)
                img_caption = self.train_captions[idx]
                img_url = self.train_urls[self.train_img_idxs[idx]]
                self.train_cursor += 1
                if self.train_cursor == self.num_train:
                    self.train_cursor = 0
                    self.epoch += 1
            else:
                idx = np.random.randint(0, len(self.val_captions))
                img_caption = self.val_captions[idx]
                img_url = self.val_urls[self.val_img_idxs[idx]]

            try:
                img = image_from_url(img_url)

                if len(img.shape) == 3:
                    img = cv2.resize(img, (CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH))
                    imgs.append(img)
                captions.append(img_caption)
                num += 1

            except urllib.error.HTTPError as e:
                print("urllib.error.HTTPError")
            except urllib.error.URLError as e:
                print("urllib.error.HTTPError")
            except Exception as e:
                # 这个是输出错误类别的，如果捕捉的是通用错误，其实这个看不出来什么
                print
                'str(Exception):\t', str(Exception)  # 输出  str(Exception):	<type 'exceptions.Exception'>
                # 这个是输出错误的具体原因，这步可以不用加str，输出
                print
                'str(e):\t\t', str(e)  # 输出 str(e):		integer division or modulo by zero
                print
                'repr(e):\t', repr(e)  # 输出 repr(e):	ZeroDivisionError('integer division or modulo by zero',)
                print
                'traceback.print_exc():';
                # 以下两步都是输出错误的具体位置的
                traceback.print_exc()
                print
                'traceback.format_exc():\n%s' % traceback.format_exc()
        return np.array(imgs), np.array(captions)


    def translate_captions(self, batch_caption_idx):
        batch_sentences = []

        for sentence_caption_idx in batch_caption_idx:
            sentence = ""
            for idx in sentence_caption_idx:
                sentence += " " + self.idx_to_word[idx]
                if idx == self.word_to_idx['<END>']:
                    break
            batch_sentences.append(sentence)
        return batch_sentences

if __name__ == '__main__':
    ds = COCODataset()
    print(ds.num_train)
    print(len(ds.val_captions))
    # m = 10
    # img, captions = ds.load_batch(m)
    # mask = np.zeros_like(captions, dtype=np.float)
    #
    # vals = np.unique(captions)
    # for val in vals:
    #     if val != ds.word_to_idx['<NULL>']:
    #         idx_val = np.where(captions == val)
    #         num_val = np.sum(captions == val)
    #         mask[idx_val] = 1 / num_val
    #

    # captions = ds.translate_captions(ds.train_captions[:1000])
    # for cap in captions:
    #     print(cap)
    # for i in range(m):
    #     plt.imshow(img[i])
    #     plt.title([captions[i]])
    #
    #     plt.show()
