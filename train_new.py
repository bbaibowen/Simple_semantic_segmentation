import torch
import torch.nn as nn
import torch.optim as optim
from pascal_voc import VOCSegmentation
# from network import Network\
import numpy as np
from nn_utils import Network
from lr_scheduler import LR_Scheduler
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from torch.utils import data
# from losses import SegmentationMultiLosses,FocalLoss2d
from focal_loss import Focalloss,OhemCrossEntropy,OHEM_Focal_Loss
from eval import Evaluator
import time
import os
import math
from radam import RAdam
from torchtoolbox.optimizer import Lookahead

SAVER_PATH = './save_pth2/'

if not os.path.exists(SAVER_PATH):
    os.mkdir(SAVER_PATH)
class Trainer():
    def __init__(self):
        self.base_lr = 1e-3
        self.moment = 0.9
        self.wd = 1e-4
        self.maxepoch = 200
        self.batch = 16
        self.num_class = 21
        self.WR_EPOCH = 10
        self.cuda = True
        self.base_size = 280
        self.crop_size = 256
        self.root = 'D:\\data\\VOCtrainval_11-May-2012\\'
        self.lr_mode = 'cos'
        self.datasets = VOCSegmentation(root=self.root,mode='train',split='train',base_size=self.base_size, crop_size=self.crop_size)
        self.datasets_val = VOCSegmentation(root=self.root,split='val', mode='val',base_size=self.base_size, crop_size=self.crop_size)
        self.trainloader = data.DataLoader(dataset=self.datasets,batch_size=self.batch,
                                           drop_last=False,shuffle=True)
        self.valloader = data.DataLoader(dataset=self.datasets_val,batch_size=16,shuffle=False)
        self.Eval = Evaluator(self.num_class)
        self.Eval_val = Evaluator(self.num_class)
        self.model = Network(self.num_class,is_train=True,is_PixelShuffle=False)

        self.loss_layer = OHEM_Focal_Loss(ignore_label=-1)
        params_list = [{'params': self.model.encoder.parameters(), 'lr': self.base_lr}, ]
        if hasattr(self.model, 'decoder'):
            params_list.append({'params': self.model.decoder.parameters(), 'lr': self.base_lr * 10})

        # self.optimizer = optim.SGD(params_list, lr=self.base_lr, momentum=self.moment, weight_decay=self.wd)
        self.optimizer = RAdam(params_list,lr=self.base_lr,weight_decay=self.wd)
        self.optimizer = Lookahead(self.optimizer)
        self.scheduler = LR_Scheduler(mode=self.lr_mode, base_lr=self.base_lr, num_epochs=self.maxepoch,
                                      iters_per_epoch=len(self.trainloader), warmup_epochs=self.WR_EPOCH)
        # self.model.init_weight()
        if self.cuda:
            cudnn.benchmark = True
            self.model = self.model.cuda()
            self.loss_layer = self.loss_layer.cuda()
        print(len(self.trainloader))
        self.best = 0.0

    def train(self,epoch,loss_list,miou_list):
        train_loss = 0.0
        self.model.train()
        t0 = time.time()
        for i ,(image,mask) in enumerate(self.trainloader):
            # t1 = time.time()

            if self.cuda:
                im = Variable(image).cuda()
                label = Variable(mask).cuda()
            else:
                im = Variable(image)
                label = Variable(mask)
            self.scheduler(self.optimizer,i,epoch)
            self.optimizer.zero_grad()
            outs = self.model(im)
            loss = self.loss_layer(outs,label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()


            # preds = self.get_pred(outs).cuda()
            self.Eval.add_batch(label,torch.argmax(outs[-1].data,dim=1))
            # self.Eval.add_batch(label,preds.data)
            # print(time.time() - t1,loss.item())

        val_miou = self.val()
        miou = float(self.Eval.mIoU(clear=True))
        print('cost:{},epoch:{},loss:{},miou:{},val_miou:{}'.format(time.time() - t0,epoch,train_loss / i,miou,val_miou))
        # if (epoch % 5 == 0 and epoch != 0):

        #
        # if (epoch % 5 == 0 and epoch <50 and epoch != 0) or (epoch >= 50):
        #     torch.save(self.model.state_dict(), SAVER_PATH + 'epoch_' + repr(epoch) + '.pth')
        if val_miou > self.best:
            self.best = val_miou
            torch.save(self.model.state_dict(), SAVER_PATH + 'best.pth')

        loss_list.append(train_loss / i)
        miou_list.append(miou)


    def val(self):
        self.model.eval()
        for i ,(image,mask) in enumerate(self.valloader):
            if self.cuda:
                im = Variable(image).cuda()
                label = Variable(mask).cuda()
            else:
                im = Variable(image)
                label = Variable(mask)
            with torch.no_grad():
                outs = self.model(im)
            # pred = self.get_pred(outs)
            self.Eval_val.add_batch(label,torch.argmax(outs[-1].data,dim=1))
            # self.Eval_val.add_batch(label,pred.data)

        val_miou = self.Eval_val.mIoU(clear=True)
        return float(val_miou)


    def weights(self,input):
        mask = np.array(input)
        b = np.zeros((self.num_class,))
        for i in range(self.num_class):
            beta = (mask[i] + 1) / (self.batch * self.crop_size * self.crop_size)
            b[i] = math.sqrt((1 - beta) / beta)


        return torch.Tensor(b)

    def weights2(self,input):
        classWeights = torch.zeros(self.num_class, dtype=torch.float)
        normHist = input / torch.sum(input)
        for i in range(self.num_class):
            classWeights[i] = 1 / (torch.log(1.10 + normHist[i]))
        return classWeights

    def get_pred(self,outs):
        _num = len(outs)
        meger = np.zeros((self.batch,256, 256, _num), dtype=np.int)
        for i, j in enumerate(outs):
            meger[:,:, :, i] = np.array(torch.argmax(j.cpu(), dim=1)[0])
        pred = np.zeros((self.batch,self.crop_size, self.crop_size), dtype=np.int)
        for b in range(self.batch):
            for i in range(self.crop_size):
                for j in range(self.crop_size):
                    out = np.argmax(np.bincount(meger[b,i, j, :]))
                    pred[b,i,j] = out
        return torch.Tensor(pred)

    def torch_get_pred(self,outs): #slowly
        _num = len(outs)
        meger = torch.zeros((self.batch, 256, 256, _num), dtype=torch.int)
        for i, j in enumerate(outs):
            meger[:, :, :, i] = torch.argmax(j, dim=1)[0]
        pred = torch.zeros((self.batch, self.crop_size, self.crop_size), dtype=torch.int)
        for b in range(self.batch):
            for i in range(self.crop_size):
                for j in range(self.crop_size):
                    out = torch.argmax(torch.bincount(meger[b, i, j, :]))
                    pred[b, i, j] = out
        return pred



if __name__ == '__main__':
    trainer = Trainer()
    loss_list = list()
    miou_list = list()
    for ep in range(0,trainer.maxepoch):
        torch.cuda.empty_cache()
        # trainer.val()
        trainer.train(ep,loss_list,miou_list)
    torch.save(trainer.model.state_dict(), SAVER_PATH + 'last.pth')
    from matplotlib import pyplot as plt
    plt.plot(range(len(loss_list)), loss_list, color='black')
    plt.plot(range(len(miou_list)), loss_list, color='red')
    plt.savefig('./res.jpg')

