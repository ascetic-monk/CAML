import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
import math
import torch.nn.functional as F


class RNN(nn.Module):
    def __init__(self, hidden_size, in_channel, encoding_size, out_channel,
                 cell_type='GRU', num_layers=1, device='cpu', dropout=0, bidirectional=True, outseq=False):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.in_channel = in_channel
        self.num_layers = num_layers
        self.cell_type = cell_type
        self.encoding_size = encoding_size
        self.bidirectional = bidirectional
        self.device = device
        self.outseq = outseq
        self.out_channel = out_channel

        if cell_type == 'GRU':
            self.rnn = nn.GRU(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                              batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)

        elif cell_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=self.in_channel, hidden_size=self.hidden_size, num_layers=num_layers,
                               batch_first=False, dropout=dropout, bidirectional=bidirectional).to(self.device)
        else:
            raise ValueError('Cell type not defined, must be one of the following {GRU, LSTM, RNN}')

        self.nn = nn.Sequential(
            nn.Linear(self.hidden_size * (int(self.bidirectional) + 1), self.encoding_size)).to(self.device)
        self.bn = nn.BatchNorm1d(self.encoding_size)
        self.nnout = nn.Linear(self.encoding_size, self.output_size)

    def forward(self, x):
        x = x.permute(2, 0, 1)
        if self.cell_type == 'GRU':
            past = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), x.shape[1], self.hidden_size).to(
                self.device)
        elif self.cell_type == 'LSTM':
            h_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(
                self.device)
            c_0 = torch.zeros(self.num_layers * (int(self.bidirectional) + 1), (x.shape[1]), self.hidden_size).to(
                self.device)
            past = (h_0, c_0)
        out, _ = self.rnn(x.to(self.device), past)  # out shape = [seq_len, batch_size, num_directions*hidden_size]
        encodings = self.nn(out[-1].squeeze(0))

        output = self.nnout(self.bn(encodings))
        return output


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, group=1):
        """
        相当于一个Residual block

        :param n_inputs: int, 输入通道数
        :param n_outputs: int, 输出通道数
        :param kernel_size: int, 卷积核尺寸
        :param stride: int, 步长，一般为1
        :param dilation: int, 膨胀系数
        :param padding: int, 填充系数
        :param dropout: float, dropout比率
        """
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=group))
        # 经过conv1，输出的size其实是(Batch, input_channel, seq_len + padding)
        self.chomp1 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, groups=group))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, groups=group) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        """
        参数初始化

        :return:
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        :param x: size of (Batch, input_channel, seq_len)
        :return:
        """
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, group=1):
        """
        TCN，目前paper给出的TCN结构很好的支持每个时刻为一个数的情况，即sequence结构，
        对于每个时刻为一个向量这种一维结构，勉强可以把向量拆成若干该时刻的输入通道，
        对于每个时刻为一个矩阵或更高维图像的情况，就不太好办。

        :param num_inputs: int， 输入通道数
        :param num_channels: list，每层的hidden_channel数，例如[25,25,25,25]表示有4个隐层，每层hidden_channel数为25
        :param kernel_size: int, 卷积核尺寸
        :param dropout: float, drop_out比率
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i  # 膨胀系数：1，2，4，8……
            in_channels = num_inputs if i == 0 else num_channels[i - 1]  # 确定每一层的输入通道数
            out_channels = num_channels[i]  # 确定每一层的输出通道数
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout, group=group)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """
        输入x的结构不同于RNN，一般RNN的size为(Batch, seq_len, channels)或者(seq_len, Batch, channels)，
        这里把seq_len放在channels后面，把所有时间步的数据拼起来，当做Conv1d的输入尺寸，实现卷积跨时间步的操作，
        很巧妙的设计。

        :param x: size of (Batch, input_channel, seq_len)
        :return: size of (Batch, output_channel, seq_len)
        """
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size,
                 dropout, reduce='mean', group=1):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size, dropout=dropout, group=group)
        self.linear = nn.Linear(num_channels[-1], group*output_size)
        self.reduce = reduce
        # self.sig = nn.Sigmoid()

    def forward(self, x, seqout=False):
        # x needs to have dimension (N, C, L) in order to be passed into CNN
        output = self.tcn(x.transpose(1, 2)).transpose(1, 2)
        outputl = self.linear(output).double()

        if self.reduce == 'mean':
            outputr = torch.mean(outputl, 1)
        elif self.reduce == 'max':
            outputr = torch.max(outputl, 1)

        if seqout:
            return output, outputr
        else:
            return outputr  # self.sig(output)


class SimConv4m(torch.nn.Module):
    def __init__(self, input_size, output_size, feature_size=64, groups=1):
        super(SimConv4m, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.name = "conv4"
        self.groups = groups

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(input_size, feature_size, 3, 1, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            # nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=1, bias=False),
            # torch.nn.BatchNorm1d(feature_size),
            # torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()

        self.sup_head = torch.nn.Sequential(
            nn.Linear(feature_size, output_size)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, seq_out=False, mask=None): #
        # x_ = x.view(x.shape[0], 1, -1)
        x = x.permute(0, 2, 1)
        if not (mask == None):
            msk1, msk2, msk3 = mask

        h = self.layer1(x)  # (B, 1, D)->(B, 8, D/2)
        if not (mask == None):
            h = h * msk1
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        if not (mask == None):
            h = h * msk2
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        if not (mask == None):
            h = h * msk3
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
        h_nor = self.avgpool(h)
        h_nor = self.flatten(h_nor)
        h_nor = F.normalize(h_nor, dim=1)

        out = self.sup_head(h_nor)
        if seq_out:
            return h, out
        else:
            return out


class SimConv4(torch.nn.Module):
    def __init__(self, input_size, output_size, feature_size=64, groups=1):
        super(SimConv4, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.name = "conv4"
        self.groups = groups

        self.layer1 = torch.nn.Sequential(
            # nn.Conv1d(input_size, feature_size, 3, 1, 1, groups=1, bias=False),
            # torch.nn.BatchNorm1d(feature_size),
            # torch.nn.ReLU(),
            nn.Conv1d(input_size, feature_size, 3, 2, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            # nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=groups, bias=False),
            # torch.nn.BatchNorm1d(feature_size),
            # torch.nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            # nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=1, bias=False),
            # torch.nn.BatchNorm1d(feature_size),
            # torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()

        self.sup_head = torch.nn.Sequential(
            nn.Linear(feature_size, output_size)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, seq_out=False, mask=None): #
        # x_ = x.view(x.shape[0], 1, -1)

        x = x.permute(0, 2, 1)
        if not (mask == None):
            msk1, msk2, msk3 = mask

        h = self.layer1(x)  # (B, 1, D)->(B, 8, D/2)
        if not (mask == None):
            h = h * msk1
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        if not (mask == None):
            h = h * msk2
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        if not (mask == None):
            h = h * msk3
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
        h_nor = self.avgpool(h)
        h_nor = self.flatten(h_nor)
        h_nor = F.normalize(h_nor, dim=1)

        out = self.sup_head(h_nor)
        if seq_out:
            return h, out
        else:
            return out


class SimConv6(torch.nn.Module):
    def __init__(self, input_size, output_size, feature_size=64, groups=1):
        super(SimConv6, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.name = "conv4"
        self.groups = groups

        self.layer1 = torch.nn.Sequential(
            # nn.Conv1d(input_size, feature_size, 3, 1, 1, groups=1, bias=False),
            # torch.nn.BatchNorm1d(feature_size),
            # torch.nn.ReLU(),
            nn.Conv1d(input_size, feature_size, 3, 2, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            # nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=groups, bias=False),
            # torch.nn.BatchNorm1d(feature_size),
            # torch.nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            # nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=1, bias=False),
            # torch.nn.BatchNorm1d(feature_size),
            # torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )

        self.layer5 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )

        self.layer6 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()

        self.sup_head = torch.nn.Sequential(
            nn.Linear(feature_size, output_size)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, seq_out=False, mask=None): #
        # x_ = x.view(x.shape[0], 1, -1)

        x = x.permute(0, 2, 1)

        h = self.layer1(x)  # (B, 1, D)->(B, 8, D/2)
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
        h = self.layer5(h)
        h = self.layer6(h)
        h_nor = self.avgpool(h)
        h_nor = self.flatten(h_nor)
        h_nor = F.normalize(h_nor, dim=1)

        out = self.sup_head(h_nor)
        if seq_out:
            return h, out
        else:
            return out



class SimConv4d(torch.nn.Module):
    def __init__(self, input_size, output_size, feature_size=64, groups=1):
        super(SimConv4d, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.feature_size = feature_size
        self.name = "conv4"
        self.groups = groups

        self.layer1 = torch.nn.Sequential(
            nn.Conv1d(input_size, feature_size, 3, 1, 1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=1, bias=False),
            # nn.Dropout(0.2),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=1, bias=False),
            # nn.Dropout(0.2),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer3 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 1, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=1, bias=False),
            # nn.Dropout(0.2),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer4 = torch.nn.Sequential(
            nn.Conv1d(feature_size, feature_size, 3, 2, 1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )

        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.flatten = torch.nn.Flatten()

        self.sup_head = torch.nn.Sequential(
            nn.Linear(feature_size, output_size)
        )

        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, torch.nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            if isinstance(m, nn.Conv1d):
                nn.init.xavier_normal_(m.weight.data)
            #        nn.init.xavier_normal_(m.bias.data)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, seq_out=False, mask=None): #
        # x_ = x.view(x.shape[0], 1, -1)

        x = x.permute(0, 2, 1)
        if not (mask == None):
            msk1, msk2, msk3 = mask

        h = self.layer1(x)  # (B, 1, D)->(B, 8, D/2)
        if not (mask == None):
            h = h * msk1
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        if not (mask == None):
            h = h * msk2
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        if not (mask == None):
            h = h * msk3
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
        h_nor = self.avgpool(h)
        h_nor = self.flatten(h_nor)
        h_nor = F.normalize(h_nor, dim=1)

        out = self.sup_head(h_nor)
        if seq_out:
            return h, out
        else:
            return out


class DeConv4(torch.nn.Module):
    def __init__(self, input_size, feature_size=64, groups=1):
        super(DeConv4, self).__init__()
        self.input_size = input_size
        # self.output_size = output_size
        self.feature_size = feature_size
        self.name = "deconv4"
        self.groups = groups

        self.layer1 = torch.nn.Sequential(
            nn.ConvTranspose1d(feature_size, feature_size, 3, 2, 1,
                               output_padding=1, groups=1, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU()
        )

        self.layer2 = torch.nn.Sequential(
            nn.ConvTranspose1d(feature_size, feature_size, 3, 2, 1,
                               output_padding=1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )

        self.layer3 = torch.nn.Sequential(
            nn.ConvTranspose1d(feature_size, feature_size, 3, 2, 1,
                               output_padding=1, groups=groups, bias=False),
            torch.nn.BatchNorm1d(feature_size),
            torch.nn.ReLU(),
        )

        self.layer4 = torch.nn.Sequential(
            nn.ConvTranspose1d(feature_size, input_size, 3, 2, 1,
                               output_padding=1, groups=groups, bias=False),
        )

        # for m in self.modules():
        #     if isinstance(m, torch.nn.Conv2d):
        #         n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #         m.weight.data.normal_(0, math.sqrt(2. / n))
        #     elif isinstance(m, torch.nn.BatchNorm2d):
        #         m.weight.data.fill_(1)
        #         m.bias.data.zero_()
        #     if isinstance(m, nn.Conv1d):
        #         nn.init.xavier_normal_(m.weight.data)
        #     #        nn.init.xavier_normal_(m.bias.data)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)
        #     elif isinstance(m, nn.Linear):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x, seq_out=False):
        # x_ = x.view(x.shape[0], 1, -1)
        # x = x.permute(0,2,1)

        h = self.layer1(x)  # (B, 1, D)->(B, 8, D/2)
        h = self.layer2(h)  # (B, 8, D/2)->(B, 16, D/4)
        h = self.layer3(h)  # (B, 16, D/4)->(B, 32, D/8)
        h = self.layer4(h)  # (B, 32, D/8)->(B, 64, 1)
        # h = self.flatten(h)
        h = h.permute(0, 2, 1)
        return h
        # h = F.normalize(h, dim=1)
        # out = self.sup_head(h)
        # if seq_out:
        #     return h, out
        # else:
        #     return out


def model_select(model_name, args):
    if model_name == 'Conv':
        model = SimConv4(input_size=args.channel_num,
                         output_size=args.act_num,
                         feature_size=args.feature_size,
                         groups=1)
    if model_name == 'Conv6':
        model = SimConv4(input_size=args.channel_num,
                         output_size=args.act_num,
                         feature_size=args.feature_size,
                         groups=1)
    if model_name == 'Convd':
        model = SimConv4d(input_size=args.channel_num,
                          output_size=args.act_num,
                          feature_size=args.feature_size,
                          groups=1)
    if model_name == 'Convmul':
        model = SimConv4m(input_size=args.channel_num,
                         output_size=args.act_num,
                         feature_size=args.feature_size,
                         groups=1)
    if model_name == 'Convsub':
        model = SimConv4(input_size=args.channel_num,
                         output_size=args.act_num,
                         feature_size=int(args.feature_size),
                         groups=1)
    elif model_name == 'TCN':
        model = TCN(input_size=args.channel_num,
                    output_size=args.act_num,
                    num_channels=[args.feature_size, args.feature_size,
                                  args.feature_size, args.feature_size],
                    kernel_size=2,
                    dropout=0.2)
    elif model_name == 'DeConv':
        model = DeConv4(input_size=args.channel_num,
                        feature_size=args.feature_size,
                        groups=1)

    return model