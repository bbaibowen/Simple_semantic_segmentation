import torch
import torch.nn as nn
import numpy as np
from nn_utils import Network
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from PIL import Image
from torchvision import transforms
from vis import labelTopng


if __name__ == '__main__':
    image_path = './2.JPEG'
    im = Image.open(image_path)
    im = transforms.Resize((256*2,256*2))(im)
    im = transforms.ToTensor()(im)
    im = transforms.Normalize([.485, .456, .406], [.229, .224, .225])(im)
    model = Network(21,is_train=False,is_PixelShuffle=False)
    model.load_state_dict(torch.load('./save_pth2/epoch_173.pth'))
    model = model.cuda()
    model.eval()
    im = im.unsqueeze(0).cuda()
    pred = model(im)[-1]
    label = pred.data.max(1)[1].squeeze_(1).squeeze_(0)
    labelTopng(label, img_name='test2.png')

