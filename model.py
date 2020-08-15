import torch
import torch.nn as nn
from torch.autograd import Variable
from MobileNetV2 import MobileNetV2


class EventDetector(nn.Module):
    def __init__(self, pretrain, width_mult, lstm_layers, lstm_hidden, device, use_no_element, bidirectional=True, dropout=True):
        super(EventDetector, self).__init__()
        self.width_mult = width_mult
        self.lstm_layers = lstm_layers
        self.lstm_hidden = lstm_hidden
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.device = device
        self.use_no_element = use_no_element

        # TODO : 19~25をコメント
        # net = MobileNetV2(width_mult=width_mult)
        # state_dict_mobilenet = torch.load('mobilenet_v2.pth.tar')
        # if pretrain:
        #     net.load_state_dict(state_dict_mobilenet,strict=False)
        #     #net.load_state_dict(state_dict_mobilenet)

        # self.cnn = nn.Sequential(*list(net.children())[0][:19])
        self.rnn = nn.LSTM(int(36*width_mult if width_mult > 1.0 else 36),
                           self.lstm_hidden, self.lstm_layers,
                           batch_first=True, bidirectional=bidirectional)
        if self.use_no_element == False:
            if self.bidirectional:
                self.lin = nn.Linear(2*self.lstm_hidden, 12)
            else:
                self.lin = nn.Linear(self.lstm_hidden, 12)
        else:
            if self.bidirectional:
                self.lin = nn.Linear(2*self.lstm_hidden, 13)
            else:
                self.lin = nn.Linear(self.lstm_hidden, 13)


        if self.dropout:
            self.drop = nn.Dropout(0.5)

    def init_hidden(self, batch_size):
        if self.bidirectional:
            return torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(self.device),torch.zeros(2*self.lstm_layers, batch_size, self.lstm_hidden).to(self.device)
        else:
            return torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device),torch.zeros(self.lstm_layers, batch_size, self.lstm_hidden).to(self.device)

            

    def forward(self, x, lengths=None):
        # TODO : 55行目追加
        # batch_size, timesteps, C, H, W = x.size()  ##torch.Size([8, 300, 3, 224, 224])
        batch_size, timesteps, one_person_coordinates, each_coordinates = x.size() ##torch.Size([8, 300, 18, 2])
        self.hidden = self.init_hidden(batch_size)

        # # CNN forward
        # c_in = x.view(batch_size * timesteps, C, H, W)  ##torch.Size([2400, 3, 224, 224])
        # c_out = self.cnn(c_in)
        # c_out = c_out.mean(3).mean(2)  ##torch.Size([2400, 1280])  ##Global average pooling
        # if self.dropout:
        #     c_out = self.drop(c_out)

        # TODO : c.outを座標データに変えるはず
        # LSTM forward

        r_in = x.view(batch_size, timesteps, one_person_coordinates * each_coordinates)  ##torch.Size([8, 300, 1280])  ##torch.Size([8, 300, 36])
        r_out, states = self.rnn(r_in, self.hidden)   ##r_out:torch.Size([8, 300, 512]),  len(states)=2
        out = self.lin(r_out)  ##torch.Size([8, 300, 12])

        if self.use_no_element == False:
            out = out.view(batch_size*timesteps, 12)
        else:
            out = out.view(batch_size*timesteps, 13)
        # out.shape => torch.Size([2400, 13])
        return out
