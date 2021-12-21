import sys
sys.path.append('..')

import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf

from PIL import Image
import base64
import cv2
from io import BytesIO

import matplotlib.pyplot as plt


from models.Captioning import Captioning
from models.Solver import Solver
from utils.data_utils import COCODataset
from utils.config import CONFIG

class Inference():

    def __init__(self,num):

        self.ds = COCODataset()
        self.model = Captioning(CONFIG.RNN_DIM, CONFIG.WORD2VEC_EMBED_DIM, )
        self.solver = Solver(self.model, self.ds)

        print('loading data...')
        self.imgs, true_captions = self.ds.load_batch(num, kind='val')
        self.true_captions = self.ds.translate_captions(true_captions)
        print('loaded')

    def get_caption(self):

        with self.model.graph.as_default():
            sess = tf.Session(graph=self.model.graph)
            saver = tf.train.Saver()
            saver.restore(sess, save_path='/Users/wujinyao/Desktop/ImgCaption/ckpt79/checkpoint1')

            feed_dict = self.solver.get_inference_feed_dict(self.imgs)
            captions = sess.run(self.model.output_infer_time, feed_dict=feed_dict)
            captions = self.ds.translate_captions(captions)

        return self.imgs, captions

    def get_true_caption(self):

        return self.true_captions


if __name__ == "__main__":

    m = 5
    inference = Inference(m)
    imgs, captions = inference.get_caption()
    true_captions = inference.get_true_caption()

    print(type(imgs[0]))
    print(imgs[0].shape)

    # img = Image.fromarray(imgs[0], 'RGB')
    # print(type(img))
    # img.save('/Users/wujinyao/Desktop/untitled/static/test_imgs/1.png')
    # # img.show()

    # transform to bytes
    # img = imgs[0].tobytes()
    # print(type(img))
    # print(img)
    #
    # base64_img = base64.b64encode(img)  # base64 encode
    # print(type(base64_img))
    # print(base64_img)

    image = Image.fromarray(cv2.cvtColor(imgs[0], cv2.COLOR_BGR2RGB))
    output_buffer = BytesIO()
    image.save(output_buffer, format="JPEG")
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data)
    print(str(base64_str, encoding="utf-8"))


    for i in range(m):
        plt.imshow(imgs[i])
        plt.title(captions[i])
        print("true caption:", true_captions[i])
        print("caption:", captions[i])
        print()
        plt.show()
