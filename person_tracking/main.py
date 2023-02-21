import torch
import numpy as np
from data_preparation import load_X, load_y, Dataloader
from train import train
from torch import nn
from lstm import LSTMModel, init_weights
from util import plot, evaluate
from config import Model
import sys


# Data file to load X and y values
# df = Dataloader()
# X_train_signals_paths = df.X_train_signals_paths
# X_test_signals_paths = df.X_test_signals_paths

# y_train_path = df.y_train_path
# y_test_path = df.y_test_path

# LSTM Neural Network's internal structure

n_hidden = Model.n_hidden
n_classes = Model.n_classes
epochs = Model.n_epochs
learning_rate = Model.learning_rate
weight_decay = Model.weight_decay
clip_val = Model.clip_val
diag = Model.diag

# Training
# check if GPU is available

#train_on_gpu = torch.cuda.is_available()
if (torch.cuda.is_available() ):
    print('Training on GPU')
else:
    print('GPU not available! Training on CPU. Try to keep n_epochs very small')


def main():

    # X_train = load_X(X_train_signals_paths)
    # X_test = load_X(X_test_signals_paths)

    # y_train = load_y(y_train_path)
    # y_test = load_y(y_test_path)
    df = Dataloader()
    X_train = df.x_train
    X_test = df.x_test
    y_train = df.y_train
    y_test = df.y_train

    # Input Data

    training_data_count = len(X_train)  # num training series
    test_data_count = len(X_test)  # num testing series
    n_steps = len(X_train[0])  # 128 timesteps per series
    n_input = len(X_train[0][0])  # num input parameters per timestep


    # Some debugging info

    # print("Some useful info to get an insight on dataset's shape and normalisation:")
    # print("(X shape, y shape, every X's mean, every X's standard deviation)")
    # print(X_test.shape, y_test.shape, np.mean(X_test), np.std(X_test))
    # print("The dataset is therefore properly normalised, as expected, but not yet one-hot encoded.")

    for lr in learning_rate:
        arch = Model.arch
        net = LSTMModel()
        # if arch['name'] == 'LSTM1' or arch['name'] == 'LSTM2':
        #     net = LSTMModel()
        # elif arch['name'] == 'Res_LSTM':
        #     net = Res_LSTMModel()
        # elif arch['name'] == 'Res_Bidir_LSTM':
        #     net = Res_Bidir_LSTMModel()
        # elif arch['name'] == 'Bidir_LSTM1' or arch['name'] == 'Bidir_LSTM2':
        #     net = Bidir_LSTMModel()
        # else:
        #     print("Incorrect architecture chosen. Please check architecture given in config.py. Program will exit now! :( ")
        #     sys.exit()
        net.apply(init_weights)
        print(diag)
        opt = torch.optim.Adam(net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        net = net.float()
        params = train(net, X_train, y_train, X_test, y_test, opt=opt, criterion=criterion, epochs=epochs, clip_val=clip_val)
        evaluate(params['best_model'], X_test, y_test, criterion)
        plot(params['epochs'], params['train_loss'], params['test_loss'], 'loss', lr)
        plot(params['epochs'], params['train_accuracy'], params['test_accuracy'], 'accuracy', lr)

        #plot(params['lr'], params['train_loss'], params['test_loss'], 'loss_lr', lr)


main()
