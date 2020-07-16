import pandas as pd
import numpy as np
from keras.utils import to_categorical
from keras.layers import Dense, LSTM, concatenate
from keras.models import Input, Model, load_model
from matplotlib import pyplot as plt
from keras.utils import plot_model
import time

from nltk import ngrams
from keras.callbacks import EarlyStopping

class LstmCut:
    '''
    Predict the next time gap using the surrounding context. The context is:
        1) {w} activities on left
        2) {w} time gap on left
        3) current activity
        4) {w} activities on right
        5) {w} time gap on right

    In the paper, we have shown that partitioning traces based on the predicted time gaps instead of the
    real time gaps yields better partitioning.
    '''
    def __init__(self, df, case_col, time_diff, activity_col, window=10, name='model', factor=32, noise=0.5):
        '''

        :param df: pandas dataframe (from CSV)
        :param case_col: DF COLUMN: journey id. Ideally, it should be long-running and complex traces otherwise we would not need to partition the journey)
        :param time_diff: DF COLUMN: Time until next event (e.g., in seconds)
        :param activity_col: DF COLUMN: Activity
        :param window: size of the context; i.e., how many on the left and rights should be included in the context.
        :param name: Name of the experiment (will be used to name the model when it is exported)
        :param factor: Size of the neural network
        :param noise: Size of dropout (avoid overfitting)
        '''
        self.df = df
        self.case_col = case_col
        self.time_diff = time_diff
        self.factor = factor
        self.noise = noise
        self.activity_col = activity_col
        self.mapping_activity = {x:y for y,x in pd.Series(df[activity_col].unique()).items()}
        self.mapping_activity['start'] = df[activity_col].nunique()
        self.mapping_activity['end'] = self.mapping_activity['start']+1
        self.df[activity_col] = df[activity_col].map(self.mapping_activity)
        self.name = name
        self.model = None
        self.window = window
        self.activities_left = None
        self.activities_right = None
        self.times_left = None
        self.times_right = None
        self.activities = None
        self.y = None
        self.prepare_feature()
        self.cut()

    def prepare_feature(self):
        '''
        For each activities, retrieve the surrounding context:

        '''

        w = self.window

        # Retrieve the surrounding activities
        self.activities_left = [[y[:-1] for y in ngrams(x, w+1, pad_left=True, left_pad_symbol=self.mapping_activity['start'])] for x in self.df.groupby(self.case_col)[self.activity_col].agg(list)]
        self.activities_right = [[y[1:] for y in ngrams(x, w+1, pad_right=True, right_pad_symbol=self.mapping_activity['end'])] for x in self.df.groupby(self.case_col)[self.activity_col].agg(list)]
        self.activities_left = to_categorical([u for x in self.activities_left for u in x], num_classes=len(self.mapping_activity))
        self.activities_right = to_categorical([u for x in self.activities_right for u in x], num_classes=len(self.mapping_activity))

        # Retrieve the surrounding time gap left and right
        self.times_left = [[y[:-1] for y in ngrams(x, w+1, pad_left=True, left_pad_symbol=0)] for x in self.df.groupby(self.case_col)[self.time_diff].agg(list)]
        self.times_left = np.array([u for x in self.times_left for u in x]).reshape((-1, w, 1))
        self.times_right = [[y[1:] for y in ngrams(x, w+1, pad_right=True, right_pad_symbol=0)] for x in self.df.groupby(self.case_col)[self.time_diff].agg(list)]
        self.times_right = np.array([u for x in self.times_right for u in x]).reshape((-1, w, 1))
        self.activities = self.df[self.activity_col].values
        self.activities = to_categorical(self.activities, num_classes=len(self.mapping_activity))
        self.y = self.df[self.time_diff].values

    def cut(self):
        '''
        Split 80% training and 20% validation
        '''
        n_train = int(self.df[self.case_col].nunique()*.8)
        n_validation = self.df[self.case_col].nunique()-n_train
        n_test = 0
        i = np.array(['training']*n_train+['validation']*n_validation+['test']*n_test)
        np.random.seed(2000)
        np.random.shuffle(i)
        mapping_split = pd.Series(i, index=self.df[self.case_col].unique()).to_dict()
        self.df['split'] = self.df[self.case_col].map(mapping_split)

    def build_model(self, epoch=50):
        '''
        Build the neural network model and train it
        :param epoch: number of epoch
        '''
        start_time = time.time()
        index_training = (self.df['split']=='training').values
        index_validation = (self.df['split']=='validation').values

        activities_left = Input(shape=(self.activities_left[index_training].shape[1], self.activities_left[index_training].shape[2]))
        activities_right = Input(shape=(self.activities_right[index_training].shape[1], self.activities_right[index_training].shape[2]))
        times_left = Input(shape=(self.times_left[index_training].shape[1], self.times_left[index_training].shape[2]))
        times_right = Input(shape=(self.times_right[index_training].shape[1], self.times_right[index_training].shape[2]))

        # Neural Network Architecture
        left = concatenate([activities_left, times_left])
        i3 = LSTM(self.factor, return_sequences=False, dropout=self.noise)(left)
        right = concatenate([activities_right, times_right])
        i4 = LSTM(self.factor, return_sequences=False, dropout=self.noise)(right)
        input_current_activity = Input(shape=(self.activities.shape[1],))
        f1 = concatenate([i3, i4, input_current_activity])
        f1 = Dense(self.factor)(f1)
        output = Dense(1, activation="relu")(f1)

        model = Model([activities_left, activities_right, times_left, times_right, input_current_activity], output)
        model.compile(optimizer='nadam', loss='mean_squared_error', metrics=['mean_squared_error'])

        plot_model(model, show_shapes=True, to_file='{}.png'.format(self.name))

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=20)
        training = model.fit(
            [
                self.activities_left[index_training],
                self.activities_right[index_training],
                self.times_left[index_training],
                self.times_right[index_training],
                self.activities[index_training],
            ],
            y=self.y[index_training],
            shuffle=True,
            epochs=epoch,#
            verbose=1,
            batch_size=128,
            validation_data=(
                [
                    self.activities_left[index_validation],
                    self.activities_right[index_validation],
                    self.times_left[index_validation],
                    self.times_right[index_validation],
                    self.activities[index_validation],
                ],
                self.y[index_validation]
            ),
            callbacks=[es]
        )
        self.model = model
        model.save('{}.h5'.format(self.name))

        # Plot the learning process
        plt.plot(training.history['loss'])
        plt.plot(training.history['val_loss'])
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['loss', 'val_loss'], loc='upper left')
        plt.savefig('{}_accuracy.eps'.format(self.name), format='eps')
        plt.close()

        return time.time() - start_time

    def predict(self):
        '''
        Make prediction about the next time step using neural networks
        :return: prediction about the next timestep
        '''
        if self.model is None:
            exit('The model should be build first')

        self.df['predict'] = self.model.predict([
            self.activities_left,
            self.activities_right,
            self.times_left,
            self.times_right,
            self.activities,
        ])
        self.df.loc[self.df['split']=='last', 'predict'] = np.nan
        d = self.df['predict']
        return d

    def load_model(self, path):
        self.model = load_model(path)


