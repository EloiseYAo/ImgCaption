import sys
sys.path.append('..')

import os
import tensorflow
tensorflow.compat.v1.disable_v2_behavior()
import tensorflow.compat.v1 as tf

import matplotlib.pyplot as plt
import cv2
import numpy as np


from models.Captioning import Captioning
from models.Solver import Solver
from models.vgg import vgg16
from utils.data_utils import COCODataset
from utils.config import CONFIG


class Inference(Captioning):
    def __init__(self, rnn_dim, embed_dim, ckpt_path, vgg_layer = 'fc2'):
        super(Inference, self).__init__(rnn_dim, embed_dim, None, vgg_layer)

        with self.graph.as_default():
            saver = tf.train.Saver()
            saver.restore(self.sess, ckpt_path)

            self.inference_input_holder = tf.placeholder(dtype=tf.int32, shape=[None])
            self.inference_c_holder = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_dim])
            self.inference_h_holder = tf.placeholder(dtype=tf.float32, shape=[None, self.rnn_dim])

            state = tf.nn.rnn_cell.LSTMStateTuple(self.inference_c_holder, self.inference_h_holder)
            inference_embed = tf.nn.embedding_lookup(self.params['W_word_embed'], self.inference_input_holder)
            inference_rnn_output, (self.next_c, self.next_h) = self.rnn_cell1.call(inference_embed, state)
            print(self.next_h.shape, self.next_c.shape)

            self.inference_output = tf.nn.softmax(
                tf.matmul(inference_rnn_output, self.params['W_output']) + self.params['b_output'], axis=1
            )

    def predict(self, images, beam_width):
        features = self.vgg.get_feature(images, self.vgg_feature_name)
        initial_states = self.sess.run(self.initial_state, feed_dict={self.feature_holder:features})
        # [none, 512]
        rets = []

        for initial_state in initial_states:
            captions = self.beam_search(np.array([initial_state]), beam_width)
            rets.extend(self.coco.translate_captions(captions))
        return rets


    def beam_search(self, state, beam_width=10):
        """
        :param feature:
        :return:
        """
        inputs = np.array([self.start_idx])

        top_outputs = [([self.start_idx], 0, state, state)] # (word_list, log_probability, c, h)

        c = state
        h = state

        while True:
            outputs, c, h = self.sess.run(
                [self.inference_output, self.next_c, self.next_h],
                feed_dict={
                    self.inference_input_holder : inputs,
                    self.inference_c_holder : c,
                    self.inference_h_holder : h
                })

            print(outputs.shape, c.shape, h.shape)

            for i in range(len(top_outputs)):
                outputs[i,:] = np.log(outputs[i,:]) + top_outputs[i][1]

            ridx, cidx = np.unravel_index(np.argsort(outputs, axis=None), outputs.shape)

            next_top_inputs = []
            new_inputs = []
            new_h = []
            new_c = []

            for i in range(beam_width):
                if cidx[-i] == self.end_idx:
                    return [top_outputs[ridx[-i]][0] + [cidx[-i]]]
                next_top_inputs.append(
                    (top_outputs[ridx[-i]][0] + [cidx[-i]], outputs[ridx[-i], cidx[-i]], c[ridx[-i]], h[ridx[-i]])
                )
                new_inputs.append(cidx[-i])
                new_c.append(c[ridx[-i]])
                new_h.append(h[ridx[-i]])

            inputs = np.array(new_inputs)
            top_outputs = next_top_inputs
            c = np.stack(new_c)
            h = np.stack(new_h)






if __name__ == "__main__":
    fnames = [
        '../img/1.png',
        '../img/7.png'
    ]

    imgs = []
    for fname in fnames:
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img = cv2.resize(img, (CONFIG.IMG_HEIGHT, CONFIG.IMG_WIDTH))
        imgs.append(img)
    imgs = np.array(imgs)


    predictor = Inference(CONFIG.RNN_DIM, CONFIG.WORD2VEC_EMBED_DIM, '../ckpt23/checkpoint1')
    predictor.predict(imgs,2)
    # predictor.beam_search(np.random.randn(1,512))
