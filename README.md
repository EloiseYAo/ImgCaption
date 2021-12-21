# ImgCaption

The model is improved based on the traditional Encoder-Decoder framework. The Convolutional Neural Network (CNN) model of encoding layer adopts VGG16 model, which is widely used in the field of image feature extraction, to extract image features and provide more semantic information. The Recurrent Neural Network (RNN) model of decoding layer adopts LSTM model. In the decoding layer, Word to Vector is used to transform the corresponding vectors. The model do not use an independent Word to Vector training model, but combine the Word to Vector training with the LSTM model training at the same time, so that it has a higher degree of fusion and better adaptability with the whole algorithm model.

Finally, by carrying the algorithm model on the web for visualization, users can intuitively experience the task of image caption generation. Users can choose to pull random images for image description generation on the web page, or upload images for image caption generation by themselves, and the server will save the image data uploaded by users and the generated image caption data. When users view it again, there is no need to recalculate, which improves efficiency and saves resources.



![image-20211220203821018](/Users/wujinyao/desktop/ImgCaption/image-20211220203821018.png)



![image-20211220203959161](/Users/wujinyao/desktop/ImgCaption/image-20211220203959161.png)
