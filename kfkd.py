import os

import numpy as np
from pandas.io.parsers import read_csv
from sklearn.utils import shuffle
from lasagne import layers
from lasagne.updates import nesterov_momentum
from nolearn.lasagne import NeuralNet
from nolearn.lasagne import BatchIterator
import theano

FTRAIN = "~/cnntest/data/training.csv"
FTEST = "~/cnntest/data/test.csv"

class FlipBatchIterator(BatchIterator):
    flip_indices = [
        (0, 2), (1, 3),
        (4, 8), (5, 9), (6, 10), (7, 11),
        (12, 16), (13, 17), (14, 18), (15, 19),
        (22, 24), (23, 25),
        ]

    def transform(self, Xb, yb):
        Xb, yb = super(FlipBatchIterator, self).transform(Xb, yb)

        # Flip half of the images in this batch at random:
        bs = Xb.shape[0]
        indices = np.random.choice(bs, bs / 2, replace=False)
        Xb[indices] = Xb[indices, :, :, ::-1]

        if yb is not None:
            # Horizontal flip of all x coordinates:
            yb[indices, ::2] = yb[indices, ::2] * -1

            # Swap places, e.g. left_eye_center_x -> right_eye_center_x
            for a, b in self.flip_indices:
                yb[indices, a], yb[indices, b] = (
                    yb[indices, b], yb[indices, a])

        return Xb, yb

class AdjustVariable(object):
    def __init__(self, name, start=0.03, stop=0.001):
        self.name = name
        self.start, self.stop = start, stop
        self.ls = None
    def __call__(self, nn, train_history):
        if self.ls is None:
            self.ls = np.linspace(self.start, self.stop, nn.max_epochs)
        epoch = train_history[-1]['epoch']
        new_value = float32(self.ls[epoch - 1])
        getattr(nn, self.name).set_value(new_value)

def float32(k):
    return np.cast['float32'](k)


def load(test=False,cols =None):
    """Loads data from FTEST if *test* is True, otherwise from FTRAIN.
    Pass a list of *cols* if you're interested in a subset of the
    target columns.
    """
    fname = FTEST if test else FTRAIN
    df = read_csv(os.path.expanduser(fname)) # Load pandas dataframe
    # The Image column has pixel values separated by space; convert
    # the values to numpy arrays:
    df['Image'] = df['Image'].apply(lambda im: np.fromstring(im, sep = ' '))
    
    if cols: # get a subset of columns
        df = df[list(cols)+['Image']]
        
    print(df.count()) # prints the number of values for each column
    df = df.dropna() # drop all rows that have missing values in them
    
    X = np.vstack(df['Image'].values) / 255. # scale pixel values to [0,1]
    X = X.astype(np.float32)
    
    if not test: # only FtRAIN has any target columns
        y = df[df.columns[:-1]].values
        y = (y-48)/48 # scale target coordinates to [-1,1]
        X,y = shuffle(X,y, random_state=42) # suffle train data
        y = y.astype(np.float32)
    else:
        y = None
        
    return X,y

def load2d(test=False, cols=None):
    X, y = load(test=test)
    X = X.reshape(-1, 1, 96, 96)
    return X, y

# X,y = load()
# print("X.shape == {} X.min == {:.3f}; X.max == {:.3f}".format(X.shape,X.min(),X.max()))
# print("y.shape == {}; y.min == {:.3f}; y.max == {:.3f}".format(y.shape,y.min(),y.max()))


# net1 = NeuralNet(
#     layers = [# three layers: one hidden layer
#     ('input', layers.InputLayer),
#     ('hidden',layers.DenseLayer),
#     ('output', layers.DenseLayer),
#     ],
#     # Layer parameters
#     input_shape = (None, 9216), # 96 X 96 input pixels per batch
#     hidden_num_units = 100, # number of units in hidden layer
#     output_nonlinearity = None, # output Layer uses identity function
#     output_num_units = 30, # 30 target values
    
#     # optimization method:
#     update = nesterov_momentum,
#     update_learning_rate = 0.01,
#     update_momentum = 0.9,
    
#     regression=True, # flag to indicate we're dealing with regression problem
#     max_epochs=400, # we want to train this with many epochs
#     verbose = 1,
#     )

# X,y = load()
# net1.fit(X,y)

# net2 = NeuralNet(
#     layers=[
#         ('input', layers.InputLayer),
#         ('conv1', layers.Conv2DLayer),
#         ('pool1', layers.MaxPool2DLayer),
#         ('conv2', layers.Conv2DLayer),
#         ('pool2', layers.MaxPool2DLayer),
#         ('conv3', layers.Conv2DLayer),
#         ('pool3', layers.MaxPool2DLayer),
#         ('hidden4', layers.DenseLayer),
#         ('hidden5', layers.DenseLayer),
#         ('output', layers.DenseLayer),
#         ],
#     input_shape=(None, 1, 96, 96),
#     conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
#     conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
#     conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
#     hidden4_num_units=500, hidden5_num_units=500,
#     output_num_units=30, output_nonlinearity=None,

#     update_learning_rate=0.01,
#     update_momentum=0.9,

#     regression=True,
#     max_epochs=1000,
#     verbose=1,
#     )

# X, y = load2d()  # load 2-d data
# net2.fit(X, y)

    
# # Training for 1000 epochs will take a while.  We'll pickle the
# # trained model so that we can load it back later:
# import cPickle as pickle
# with open('net2.pickle', 'wb') as f:
#     pickle.dump(net2, f, -1)

# net3 = NeuralNet(
#     layers=[
#         ('input', layers.InputLayer),
#         ('conv1', layers.Conv2DLayer),
#         ('pool1', layers.MaxPool2DLayer),
#         ('conv2', layers.Conv2DLayer),
#         ('pool2', layers.MaxPool2DLayer),
#         ('conv3', layers.Conv2DLayer),
#         ('pool3', layers.MaxPool2DLayer),
#         ('hidden4', layers.DenseLayer),
#         ('hidden5', layers.DenseLayer),
#         ('output', layers.DenseLayer),
#         ],
#     input_shape=(None, 1, 96, 96),
#     conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
#     conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
#     conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
#     hidden4_num_units=500, hidden5_num_units=500,
#     output_num_units=30, output_nonlinearity=None,

#     update_learning_rate=0.01,
#     update_momentum=0.9,

#     regression=True,
#     batch_iterator_train=FlipBatchIterator(batch_size=128),
#     max_epochs=3000,
#     verbose=1,
#     )

# X, y = load2d()

# net3.fit(X, y)

# import cPickle as pickle
# with open('net3.pickle', 'wb') as f:
#     pickle.dump(net3, f, -1)

net4 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    hidden4_num_units=500, hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    # batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=3000,
    verbose=1,
    )

X, y = load2d()

net4.fit(X, y)

import cPickle as pickle
with open('net4.pickle', 'wb') as f:
    pickle.dump(net4, f, -1)
    

net6 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),  # !
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),  # !
        ('conv3', layers.Conv2DLayer),
        ('pool3', layers.MaxPool2DLayer),
        ('dropout3', layers.DropoutLayer),  # !
        ('hidden4', layers.DenseLayer),
        ('dropout4', layers.DropoutLayer),  # !
        ('hidden5', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, 96, 96),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2),
    dropout1_p=0.1,  # !
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_pool_size=(2, 2),
    dropout2_p=0.2,  # !
    conv3_num_filters=128, conv3_filter_size=(2, 2), pool3_pool_size=(2, 2),
    dropout3_p=0.3,  # !
    hidden4_num_units=500,
    dropout4_p=0.5,  # !
    hidden5_num_units=500,
    output_num_units=30, output_nonlinearity=None,

    update_learning_rate=theano.shared(float32(0.03)),
    update_momentum=theano.shared(float32(0.9)),

    regression=True,
    batch_iterator_train=FlipBatchIterator(batch_size=128),
    on_epoch_finished=[
        AdjustVariable('update_learning_rate', start=0.03, stop=0.0001),
        AdjustVariable('update_momentum', start=0.9, stop=0.999),
        ],
    max_epochs=3000,
    verbose=1,
    )

import sys
sys.setrecursionlimit(10000)

X, y = load2d()
net6.fit(X, y)

import cPickle as pickle
with open('net6.pickle', 'wb') as f:
    pickle.dump(net6, f, -1)