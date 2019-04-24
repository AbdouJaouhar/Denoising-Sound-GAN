import numpy as np 
from keras.models import Model, load_model, Sequential
from keras.layers import PReLU,Input,Dropout,BatchNormalization,Activation,Add, Multiply,Reshape, Permute, Concatenate, concatenate,  Conv1D, Dense, Flatten,AtrousConvolution1D
from keras.layers.core import Lambda
from keras.layers.convolutional import SeparableConv1D, UpSampling1D, Conv2DTranspose
from keras.layers.pooling import MaxPooling1D
from keras.layers.merge import concatenate
from keras.layers.advanced_activations import LeakyReLU
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K
from keras.optimizers import Adam, SGD, RMSprop
import tensorflow as tf
import os
import glob, os
from keras import metrics
from random import shuffle
from keras.utils import to_categorical
import itertools

class Denoising_GAN(object):

    def __init__(self, sound_length):
        self.sound_length = sound_length

        self.gen = self.Generator()
        self.gen.summary()
        self.dis = self.Discriminator()
        
        optim = RMSprop(lr=0.008)
        
        self.dis.compile(optimizer=optim,loss="mse",metrics=["accuracy"])
        self.gen.compile(optimizer=optim,loss=self.L1_Loss,metrics=["accuracy"])
        
        self.clear_sounds = []
        self.noisy_sounds = []

        self.combined = Sequential()
        self.combined.add(self.gen)

        self.combined.add(self.dis)
        self.combined.compile(loss=["mse"], optimizer=optim)

    def Generator(self):

        def DownConv1DModule(input, filters, filters_size = 10, BN = True):
 
            x = Conv1D(filters, filters_size, strides = 2, padding='same')(input)
            x = PReLU()(x)

            if BN:
                x = BatchNormalization(momentum=0.2)(x)

            x = MaxPooling1D(pool_size=2,padding='same')(x)

            return x

        def UpConv1DModule(input, skip, filters, filters_size=5, dropout = 0.0):
            def Conv1DTranspose(input, filters, filters_size, strides=2, padding='same'):
                x = Lambda(lambda x: K.expand_dims(x, axis=2))(input)
                x = Conv2DTranspose(filters=filters, kernel_size=(filters_size, 1), strides=(strides, 1), padding=padding)(x)
                x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
                return x

            x = Conv1DTranspose(input, filters, filters_size)
            x = UpSampling1D(size=2)(x)

            x = BatchNormalization(momentum=0.8)(x)
            x = concatenate([x, skip], axis=1)

            return x

        start_filter = 64

        input_sound = Input(shape=(1,self.sound_length))
        inp_resh = Reshape((self.sound_length,1))(input_sound)
        D1 = DownConv1DModule(inp_resh, start_filter)
        D2 = DownConv1DModule(D1, start_filter*2)
        D3 = DownConv1DModule(D2, start_filter*4)
        D4 = DownConv1DModule(D3, start_filter*8)        
        
        U1 = UpConv1DModule(D4, D3, start_filter*4)
        U1 = PReLU()(U1)
        U2 = UpConv1DModule(U1, D2, start_filter*2)
        U2 = PReLU()(U2)
        U3 = UpConv1DModule(U2, D1, start_filter)
        U3 = Activation('tanh')(U3)

        output_sound = Conv1D(1, 1, padding='same', activation='softmax')(U3)
        
        out_resh = Reshape((1,self.sound_length))(output_sound)
        return Model(input_sound, out_resh)

    def Discriminator(self):


        model = Sequential()
        model.add(Conv1D(16 , kernel_size=31, strides=2, padding='same', input_shape=(1,self.sound_length)))
        model.add(PReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(32, kernel_size=31, strides=2, padding='same'))
        model.add(PReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(64, kernel_size=31, strides=2, padding='same'))
        model.add(PReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(128, kernel_size=31, strides=2, padding='same'))
        model.add(PReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(256, kernel_size=31, strides=2, padding='same'))
        model.add(PReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(512, kernel_size=31, strides=2, padding='same'))
        model.add(PReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv1D(1024, kernel_size=31, strides=2, padding='same'))
        model.add(PReLU())
        model.add(BatchNormalization(momentum=0.8))
        model.add(Flatten())
        model.add(Dense(1,activation="tanh"))
        model.summary()

        return model

    
    def generator2(self):
        def make_z(shape, mean=0., std=1., name='z'):
            z = tf.random_normal(shape, mean=mean, stddev=std,name=name, dtype=tf.float32)
            return z
          
        def WaveNetResidualConv1D(num_filters, kernel_size, dilation_rate):
            def build_residual_block(input):

                sigm_conv1d = AtrousConvolution1D(num_filters, kernel_size, dilation_rate=dilation_rate,padding="same", activation="sigmoid")(input)
                tanh_conv1d = AtrousConvolution1D(num_filters, kernel_size, dilation_rate=dilation_rate,padding="same", activation="tanh")(input)

                multiplyLayers = Multiply()([sigm_conv1d, tanh_conv1d])

                skip_connection = Conv1D(1, 1)(multiplyLayers)
                residual = Add()([input, skip_connection])

                return residual, skip_connection

            return build_residual_block

        def UpConv1DModule(input, filters, filters_size=5, dropout = 0.0):
            def Conv1DTranspose(input, filters, filters_size, strides=2, padding='same'):
                x = Lambda(lambda x: K.expand_dims(x, axis=2))(input)
                x = Conv2DTranspose(filters=filters, kernel_size=(filters_size, 1), strides=(strides, 1), padding=padding)(x)
                x = Lambda(lambda x: K.squeeze(x, axis=2))(x)
                return x

            x = Conv1DTranspose(input, filters, filters_size)
            x = UpSampling1D(size=2)(x)

            x = BatchNormalization(momentum=0.8)(x)

            return x
        
        input_sound = Input(shape=(1,self.sound_length))    
        kwidth = 31
        enc_layers = 7
        
        h_i = input_sound
        skip_out = True
        skips = []
        g_enc_depths = [16, 32, 32, 64, 64, 128, 128, 256, 256, 512, 1024]
        
        for layer_idx, layer_depth in enumerate(g_enc_depths):
          
            h_i_dwn = Conv1D(layer_depth, kwidth, data_format='channels_first')(h_i)
            h_i = h_i_dwn
            
            if layer_idx < len(g_enc_depths) - 1:
                skips.append(h_i)
                
            h_i = PReLU()(h_i)

#         z = make_z((input_sound.get_shape().as_list()[0], 1, 16384))
#         h_i = concatenate([z, h_i])


        g_dec_depths = g_enc_depths[:-1][::-1] + [1]
        
        for layer_idx, layer_depth in enumerate(g_dec_depths):
            h_i_dim = h_i.get_shape().as_list()
          
            h_i_dcv = UpConv1DModule(h_i, layer_depth, kwidth)
            
            h_i = h_i_dcv
            if layer_idx < len(g_dec_depths) - 1:
              
                h_i = PReLU()(h_i)
                skip_ = skips[-(layer_idx + 1)]
                h_i = concatenate([Permute((2,1))(h_i), skip_],axis=2)

            else:
                h_i = Activation('tanh')(h_i)

        return Model(input_sound,h_i)
      
    def getDatas(self, Max):
        self.clear_sounds = np.load('/content/drive/My Drive/DeeLearning/clean_sounds.npz', mmap_mode='r')
        self.clear_sounds = self.clear_sounds['arr_0']
        self.noisy_sounds = np.load('/content/drive/My Drive/DeeLearning/noisy_sounds.npz')
        self.noisy_sounds = self.noisy_sounds['arr_0']
        
    def L1_Loss(self, y_true, y_pred):
        HUBER_DELTA = 0.5
        x = K.abs(y_true - y_pred)
        x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
        return  K.sum(x)

    def train(self,epochs,batch_size, n_mini_batch,Max):
        self.getDatas(Max)
        self.clear_sounds = np.array(self.clear_sounds).reshape((11572,1,16384))
        self.noisy_sounds = np.array(self.noisy_sounds).reshape((11572,1,16384))
        print('ALLL LOADEEEEEEEEEDD')
        # print(self.clear_sounds.shape)
        # self.gen.fit(self.clear_sounds, self.noisy_sounds, batch_size=30,epochs=30)
        N = 250
        self.gen.fit(self.noisy_sounds[:N], self.clear_sounds[:N], batch_size=50,epochs=30)
#             for epoch in range(epochs):
#                 print("Epoch : ",epoch)
#                 for index in range(n_mini_batch):
#                     noise = np.random.uniform(0, 1, (batch_size, 1, 16384))
#                     generated_audio = self.gen.predict(noise)

#                     idx = np.random.randint(0, Max, batch_size)
#                     audio_batch = np.array([self.clear_sounds[idx[i]] for i in range(len(idx))]).reshape((batch_size, 1, 16384))

#                     X = np.concatenate((audio_batch, generated_audio))
#                     y = [1] * batch_size + [0] * batch_size

#                     d_loss = self.dis.train_on_batch(X, y)

#                     self.dis.trainable = False

#                     g_loss = self.combined.train_on_batch(noise, [1]*batch_size)
#                     print(d_loss)
#                     self.dis.trainable = True
#                     d_loss = d_loss[0]
#                     print ("%d D loss: %f" % (epoch, d_loss))

print("Creating ...\n")
