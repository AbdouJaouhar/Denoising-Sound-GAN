import numpy as np 
from keras.models import Model, load_model, Sequential
from keras.layers import Input,Dropout,BatchNormalization,Activation,Add, Multiply,Reshape, Concatenate, concatenate,  Conv1D, Dense, Flatten
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
import soundfile as sf
import progressbar
from keras import metrics
from random import shuffle
from keras.utils import to_categorical
import itertools
class GAN(object):

    def __init__(self, sound_length):
        self.sound_length = sound_length

        self.gen = self.Generator()
        self.gen.summary()
        self.dis = self.Discriminator()

        self.dis.compile(optimizer=RMSprop(lr=0.0002, decay=0.2),loss='categorical_crossentropy',metrics=[metrics.binary_accuracy, metrics.mean_squared_error])
        self.gen.compile(optimizer=RMSprop(lr=0.0002, decay=0.2),loss=self.L1_Loss,metrics=[metrics.binary_accuracy, metrics.mean_squared_error])
        
        self.clear_sounds = []
        self.noisy_sounds = []

        # The generator takes noise as input and generates imgs
        noise = Input(shape=(1,self.sound_length))
        sound = self.gen(noise)
        print()

        # For the combined model we will only train the generator
        self.dis.trainable = False

        # The valid takes generated images as input and determines validity
        valid = self.dis(sound)

        # The combined model  (stacked generator and discriminator)
        # Trains generator to fool discriminator
        self.combined = Model(noise, valid)
        self.combined.compile(loss=['binary_crossentropy'], optimizer=RMSprop(lr=0.0002, decay=0.2))

    def Generator(self):

        def DownConv1DModule(input, filters, filters_size = 10, BN = True):
 
            x = Conv1D(filters, filters_size, strides = 2, padding='same')(input)
            x = LeakyReLU(alpha=0.3)(x)

            if BN:
                x = BatchNormalization(momentum=0.8)(x)

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
            if dropout:
                x = Dropout(dropout)(x)

            #x = BatchNormalization(momentum=0.8)(x)
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
        U2 = UpConv1DModule(U1, D2, start_filter*2)
        U3 = UpConv1DModule(U2, D1, start_filter)

        output_sound = Conv1D(1, 1, padding='same', activation='softmax')(U3)
        
        out_resh = Reshape((1,self.sound_length))(output_sound)
        return Model(input_sound, out_resh)

    def Discriminator(self):


        model = Sequential()
        model.add(Conv1D(64 , kernel_size=4, strides=2, padding='same', input_shape=(1,self.sound_length)))
        model.add(LeakyReLU(alpha=0.8))
        model.add(Conv1D(128, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Conv1D(256, kernel_size=4, strides=2, padding='same'))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Flatten())
        model.add(Dense(2,activation="tanh"))
        model.summary()

        return model

    def getDatas(self):
        def min_max_scale(s):
            temp =  (s - s.min()) / (s.max() - s.min())
            return np.array([t[0] for t in temp]).reshape((1,self.sound_length))


        os.chdir("D:/Kaggle/sounds/clean_trainset_wav")
        j= 0
        MAX = 10000
        MAX = 10000
        with progressbar.ProgressBar(max_value=MAX) as bar:        
            for file in glob.glob("*.wav"):
                sound, sr = sf.read("D:/Kaggle/sounds/clean_trainset_wav/"+file, dtype='int32')
                self.clear_sounds.append(min_max_scale(sound[:N].reshape(N, 1).astype('int64')).astype('float32'))
                bar.update(j)
                j+=1
                if j > MAX:
                    break

        os.chdir("D:/Kaggle/sounds/noisy_trainset_wav/")
        j= 0

        MAX = 10000
        with progressbar.ProgressBar(max_value=MAX) as bar:
            for file in glob.glob("*.wav"):
                sound, sr = sf.read("D:/Kaggle/sounds/noisy_trainset_wav/"+file, dtype='int32')
                self.noisy_sounds.append(min_max_scale(sound[:N].reshape(N, 1).astype('int64')).astype('float32'))
                bar.update(j)
                j+=1
                if j > MAX:
                    break
        
    def L1_Loss(self, y_true, y_pred):
        HUBER_DELTA = 0.5
        x = K.abs(y_true - y_pred)
        x = tf.where(x < HUBER_DELTA, 0.5 * x ** 2, HUBER_DELTA * (x - 0.5 * HUBER_DELTA))
        return  K.sum(x)

    def train(self,epochs,batch_size):
        self.getDatas()
        self.clear_sounds = np.array(self.clear_sounds).reshape((10001,1,16384))
        self.noisy_sounds = np.array(self.noisy_sounds).reshape((10001,1,16384))
        # print(self.clear_sounds.shape)
        # self.gen.fit(self.clear_sounds, self.noisy_sounds, batch_size=30,epochs=30)

        for epoch in range(epochs):
            # Train Discriminator with real sounds
            X_train = np.array([*self.noisy_sounds, *self.clear_sounds]).astype(np.float32)
            X_train = np.expand_dims(X_train, axis=1)
            idx = np.random.randint(0, len(X_train), batch_size)

            y_train_true = [1 for _ in range(batch_size)]
            shuffle(X_train)
            real_sounds = np.array([list(itertools.chain(*X_train[idx[i]][0])) for i in range(len(idx))])
            
            # real_sounds = real_sounds.reshape((1,1,16384))
            

            labels_true = to_categorical(y_train_true)
            
            d_loss_real = self.dis.fit(real_sounds.reshape((batch_size,1,16384)), labels_true,  batch_size=batch_size,verbose=0)

            # Train Discriminator with generator made sounds
            noisy_sounds = np.array(self.noisy_sounds)
            noisy_sounds = np.expand_dims(noisy_sounds, axis=1)
            idx_fake = np.random.randint(0, len(noisy_sounds), batch_size)
            

            X_train_fake = np.array([noisy_sounds[idx_fake[i]][0] for i in range(len(idx_fake))])
            

            y_train_fake = [0 for _ in range(batch_size)]
            labels_fake = to_categorical(y_train_fake, num_classes=2)

            fake_sounds = []
            # for i in range(len(idx_fake)):

            #     fake_sounds.append(self.gen.predict(np.array([list(itertools.chain(*X_train_fake[i]))])[i].reshape((batch_size,1,16384))))
                
            # fake_sounds = np.expand_dims(fake_sounds, axis=1)
            fake_sounds = self.gen.predict(X_train_fake)
            fake_sounds = np.array(fake_sounds).reshape((batch_size,1,16384))
            d_loss_fake = self.dis.fit(fake_sounds, labels_fake, batch_size=batch_size,verbose=0)

            # Train Generator on noise/clear sounds 
            clear_sounds =self.clear_sounds
            g_loss = self.combined.fit(X_train_fake, labels_fake, batch_size=batch_size,verbose=0)
            
            d_loss = 0.5 * np.add(d_loss_real.history['loss'][0], d_loss_fake.history['loss'][0])
            if epoch % 10 == 0:
                print ("%d D loss: %f" % (epoch, d_loss))


print("Creating ...\n")

N = 16384
gan = GAN(N)
gan.train(1000,100)

