import torch
import torch.nn as nn
import torch.nn.functional as F
from config import Model

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
arch = Model.LSTM_ARCH
def init_weights(m):
    if type(m) == nn.LSTM:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'weight_hh' in name:
                torch.nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    elif type(m) == nn.Linear:
        torch.nn.init.orthogonal_(m.weight)
        m.bias.data.fill_(0)

class LSTMModel(nn.Module):

    def __init__(self, n_input, n_hidden, n_layers,
                 n_classes, drop_prob):
        super(LSTMModel, self).__init__()

        self.n_layers = arch['n_layers']
        self.n_hidden = Model.HIDDEN_DIM
        self.n_classes = Model.N_CLASSES
        self.drop_prob = arch['drop_prob']
        self.n_input = Model.INPUT_DIM

        self.lstm1 = nn.LSTM(n_input, n_hidden, n_layers, dropout=self.drop_prob)
        self.lstm2 = nn.LSTM(n_hidden, n_hidden, n_layers, dropout=self.drop_prob)
        self.fc = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(drop_prob)

    def forward(self, x, hidden):
        x = x.permute(1, 0, 2)
        x, hidden1 = self.lstm1(x, hidden)
        for i in range(arch['n_highway_layers']):
            #x = F.relu(x)
            x, hidden2 = self.lstm2(x, hidden)
        x = self.dropout(x)
        out = x[-1]
        out = out.contiguous().view(-1, self.n_hidden)
        out = self.fc(out)
        out = F.softmax(out)

        return out
# class LSTM(nn.Module):
    
#     def __init__(self,input_dim,hidden_dim,output_dim,layer_num):
#         super(LSTM,self).__init__()
#         self.hidden_dim = hidden_dim
#         self.output_dim = output_dim
#         self.lstm = torch.nn.LSTM(input_dim,hidden_dim,layer_num,batch_first=True)
#         self.fc = torch.nn.Linear(hidden_dim,output_dim)
#         self.bn = nn.BatchNorm1d(32)
        
#     def forward(self,inputs):
#         x = self.bn(inputs)
#         lstm_out,(hn,cn) = self.lstm(x)
#         out = self.fc(lstm_out[:,-1,:])
#         return out