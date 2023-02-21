

class Model:
    N_CLASSES = 4
    INPUT_DIM = 52
    HIDDEN_DIM = 32
    BATCH_SIZE = 64
    EPOCHS = 100
    LSTM_ARCH = {
	'name' : 'LSTM',
	'bidir' : False,
	'clip_val' : 10,
	'drop_prob' : 0.5,
	'n_epochs_hold' : 100,
	'n_layers' : 2,
	'learning_rate' : [0.0015],
	'weight_decay' : 0.001,
	'n_residual_layers' : 0,
	'n_highway_layers' : 1,
	'diag' : 'Architecure chosen is baseline LSTM with 1 layer',
	'save_file' : 'results_lstm.txt'
}


class Data:
    PATH_TO_FOLDER = 'Data/'
    DIMENSION = ['_x', '_y', '_z', '_visibility']
    FEATURE_TYPE = ['nose', 'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist', 'left_hip', 'right_hip', 'left_knee', 'right_knee']
    LABELS = ["WALKING", "SITTING", "STANDING", "LAYING"]

# arch = Model.LSTM_ARCH

# # This will set the values according to that architecture
# bidir = arch['bidir']
# clip_val = arch['clip_val']
# drop_prob = arch['drop_prob']
# n_epochs_hold = arch['n_epochs_hold']
# n_layers = arch['n_layers']
# learning_rate = arch['learning_rate']
# weight_decay = arch['weight_decay']
# n_highway_layers = arch['n_highway_layers']
# n_residual_layers = arch['n_residual_layers']