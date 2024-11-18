from torch import nn
import torch.nn.functional as F


class BidirectGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectGRU, self).__init__()
        self.rnn = nn.GRU(input_size, hidden_size, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, output_size)

    def forward(self, x):
        r, _ = self.rnn(x)
        t, b, h = r.size()
        x = r.view(t*b, h)
        out = self.fc(x)
        return out.view(t, b, -1)


class R(nn.Sequential):
    def __init__(self, input_size, hidden_size, output_size):
        super(R, self).__init__(
            BidirectGRU(input_size, hidden_size, hidden_size),
            BidirectGRU(hidden_size, hidden_size, output_size)
        )


class ConvBNRelu(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, bn=False):
        # super(ConvBNRelu, self).__init__()
        if bn:
            super(ConvBNRelu, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
            )
        else:
            super(ConvBNRelu, self).__init__(
                nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
                nn.ReLU(inplace=True)
            )


class C(nn.Sequential):
    def __init__(self, height, in_channels):
        super(C, self).__init__()
        cs = [1, 64, 128, 256, 256, 512, 512, 512]
        ps = [1, 1, 1, 1, 1, 1, 0]
        ks = [3, 3, 3, 3, 3, 3, 2]
        cnn = nn.Sequential()
        for i in range(7):
            if i in [0, 1, 2, 3, 6]:
                cnn.add_module('conv{}'.format(i),
                               ConvBNRelu(cs[i], cs[i+1], ks[i], 1, ps[i]))
            if i in [4, 5]:
                cnn.add_module('conv{}'.format(i),
                               ConvBNRelu(cs[i], cs[i+1], ks[i], 1, ps[i], bn=True))
            if i in [0, 1]:
                cnn.add_module('pool{}'.format(i), nn.MaxPool2d(2, 2))

            if i in [3, 5]:
                cnn.add_module('pool{}'.format(i), nn.MaxPool2d(2, (2, 1), (0, 1)))

        self.cnn = cnn

    def forward(self, x):
        return self.cnn(x)


class CRNN(nn.Module):
    def __init__(self, height, in_channels, input_size, hidden_size, output_size):
        super(CRNN, self).__init__()
        self.cnn = C(height, in_channels)
        self.rnn = R(input_size, hidden_size, output_size)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)
        output = self.rnn(conv)
        return F.log_softmax(output, dim=2)


if __name__ == '__main__':
    import torch
    net = CRNN(32, 1, 512, 256, 256)
    print(net)

    x = torch.randn(1, 1, 32, 100)
    out = net(x)
    print(out.shape)