from torchvision import transforms
import torch
import numpy as np

from data import test_dl
import matplotlib.pyplot as plt
from config import device, ckpt, char_list
from train import converter, net

if __name__ == '__main__':
    params = torch.load(ckpt)
    net.load_state_dict(params['state_dict'])
    print('current loss: {}'.format(params['best_loss']))

    net.to(device)
    col = 0
    row = 1
    for d in test_dl.dataset:
        img = d[0].convert('L')
        h, w = img.size
        img = img.resize((int(h*(32/w)), 32))
        img_tensor = transforms.ToTensor()(img).unsqueeze(0)
        label = d[1].int()
        label = [char_list[i - 1] for i in label]
        preds = net(img_tensor.to(device))

        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)
        pred_size = torch.IntTensor([preds.size(0)])
        sim_pred = converter.decode(preds.data, pred_size.data)
        plt.subplot(330+col+1)
        plt.title(''.join(sim_pred))
        plt.imshow(np.array(img))
        plt.axis('off')
        col += 1
        if col == 9:
            break
    plt.show()