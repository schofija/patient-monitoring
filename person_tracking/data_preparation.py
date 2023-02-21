import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

from config import Data, Model

class Dataloader:
    #This will return all data for one video
    def __init__(self,  folder=Data.PATH_TO_FOLDER):
        
        self.x_train, self.y_train, self.x_test, self.y_test = Dataloader.__load_data()
        self.folder = folder
        print("Data Loaded Successfully")
    
    @staticmethod
    def __load_data():
        def load_X(self):
            all_files = [(os.path.join(self.folder, f)) for f in self.folder if f.endswith('.csv')]
            df = pd.concat(map(pd.read_csv, all_files))
            #If using a single file use line below
            #X = pd.read_csv(path, header=None).values
            chunks = int(len(X) / Model.SEQUENCE_LEN)
            X = np.array(np.split(X, chunks))
            return X


        def load_Y(self):
            #path = os.path.join(self.folder,)
            Y = pd.read_excel(self.folder, header=None, index_col=None).values
            chunks = int(len(Y) / Model.SEQUENCE_LEN)
            Y = np.array(np.split(Y, chunks))
            return Y

        def load_dataset(self):
            X = load_X(self.path)
            Y = load_Y(self.path)

            #y_ = load_Y(self.path)
            x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 0.2, random_state=1)
            #zero offset done in load_Y
            # y_test = y_test - 1
            # y_train = y_train - 1
            
            # y_test = to_categorical(y_test)
            # y_train = to_categorical(y_train)

            return x_train, y_train, x_test, y_test
        return load_dataset()


if __name__ == '__main__':
    d = Dataloader()
    print(d.x_train.shape[0])