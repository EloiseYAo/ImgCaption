
import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf
import cv2
import math
import numpy as np
from PIL import Image


from models.Captioning import Captioning
from models.Solver import Solver
from utils.data_utils import COCODataset
from utils.config import CONFIG

import matplotlib.pyplot as plt

class Input():
    def __init__(self,path=None):

        self.ds = COCODataset()
        self.model = Captioning(CONFIG.RNN_DIM, CONFIG.WORD2VEC_EMBED_DIM, )
        self.solver = Solver(self.model, self.ds)
        self.path = path
        print('loading data...')
        self.img = self.loac_pic(1)
        print('loaded')

    def loac_pic(self,batch_size):
        imgs = []
        num = 0
        while num < batch_size:
            try:
                if self.path == None:
                    raw_img = cv2.imread("../static/test_imgs/2.png")
                else:
                    raw_img = cv2.imread(self.path)
                raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
                if len(raw_img.shape) == 3:
                    raw_img = cv2.resize(raw_img, (CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH))
                    imgs.append(raw_img)
                num += 1

            except:
                pass
        return np.array(imgs)


    def get_caption(self):

        with self.model.graph.as_default():
            sess = tf.Session(graph=self.model.graph)
            saver = tf.train.Saver()
            saver.restore(sess, save_path='/Users/wujinyao/Desktop/ImgCaption/ckpt79/checkpoint1')

            feed_dict = self.solver.get_inference_feed_dict(self.img)
            caption = sess.run(self.model.output_infer_time, feed_dict=feed_dict)
            caption = self.ds.translate_captions(caption)

        return self.img, caption

if __name__ == "__main__":
    inference = Input()
    img, caption = inference.get_caption()
    plt.imshow(img[0])
    plt.title(caption[0])
    print()
    plt.show()
