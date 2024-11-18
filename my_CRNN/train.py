from model import CRNN
from data import train_dl, test_dl, char_list
import torch
from torch import nn, optim
from tqdm import tqdm
from config import device, ckpt
import os.path as osp

net = CRNN(32, 1, 512, 256, len(char_list)+1)


class strLabelConverter(object):
    def __init__(self, alphabet):
        self.alphabet = alphabet + 'c'

    def encode(self, labels):
        length = []
        result = []
        for label in labels:
            length.append(len(label))
            for index in label:
                result.append(index.item())
        text = result
        return torch.IntTensor(text), torch.IntTensor(length)

    def decode(self, t, length):
        char_list = []
        for i in range(length):
            if t[i] != 0 and (not (i > 0 and t[i-1] == t[i])):
                char_list.append(self.alphabet[t[i] - 1])
        return ''.join(char_list)


converter = strLabelConverter(''.join(char_list))


def train():
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-3)
    criteron = nn.CTCLoss(reduction='sum')
    if osp.exists(ckpt):
        c = torch.load(ckpt)
        net.load_state_dict(c['state_dict'])
        best_loss = c['best_loss']
    else:
        best_loss = 1e9

    for m in range(100):
        epoch_loss = 0.0
        for n, (image, label) in tqdm(enumerate(train_dl), total=len(train_dl)):
            optimizer.zero_grad()
            image = image.to(device)
            out = net(image)
            text, lengths = converter.encode(label)
            pred_lengths = torch.IntTensor([out.size(0)] * out.shape[1])
            loss = criteron(out, text, pred_lengths, lengths)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= len(train_dl.dataset)
        print('epoch{}_loss'.format(m), epoch_loss)

        val_loss = 0.0
        with torch.no_grad():
            for m, (image, label) in tqdm(enumerate(test_dl), total=len(test_dl)):
                image = image.to(device)
                out = net(image)
                text, lengths = converter.encode(label)
                pred_lengths = torch.IntTensor([out.size(0)] * out.shape[1])
                loss = criteron(out, text, pred_lengths, lengths)
                val_loss += loss.item()
        val_loss /= len(test_dl.dataset)
        print('val{}_loss'.format(m), val_loss)
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(
                {
                    'state_dict': net.state_dict(),
                    'best_loss': best_loss
                },
                ckpt
            )


if __name__ == '__main__':
    train()