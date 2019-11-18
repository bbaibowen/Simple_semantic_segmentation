import torch
import numpy as np

def MIOU(num_cls,pred,label):

    mat = torch.zeros((num_cls,num_cls))
    mat += genmat(num_cls,pred,label).cpu()
    print(mat)
    iou = torch.diag(mat) / (torch.sum(mat,0) + torch.sum(mat,1) - torch.diag(mat))
    print(iou)
    miou = torch.mean(iou[torch.isfinite(iou)])
    return miou

def genmat(num_cls,pred,label):
    gt_im = label.data.float()
    pre = pred.data.float()
    mask = (gt_im >= 0) & (gt_im < num_cls)
    lb = num_cls * gt_im[mask] + pre[mask]
    count = torch.bincount(lb.int(),minlength=num_cls ** 2)
    confusion_matrix = count[:num_cls ** 2].reshape(num_cls, num_cls)
    return confusion_matrix.float()



class Evaluator(object):
    def __init__(self, num_class):
        self.num_class = num_class
        self.matrix = torch.zeros((self.num_class, self.num_class))

    def add_batch(self, gt_image, pre_image):
        assert gt_image.shape == pre_image.shape
        self.matrix += self.generate_matrix(gt_image, pre_image).cpu()

    def mIoU(self, clear=False):
        iou = torch.diag(self.matrix) / (
                torch.sum(self.matrix, 0) + torch.sum(self.matrix, 1) - torch.diag(self.matrix))
        miou = torch.mean(iou[torch.isfinite(iou)])
        if clear:
            self.matrix = torch.zeros((self.num_class, self.num_class))
        return miou.item()

    def acc(self, clear=False):
        acc = torch.sum(torch.diag(self.matrix))/torch.sum(self.matrix)
        if clear:
            self.matrix = torch.zeros((self.num_class, self.num_class))
        return acc

    def generate_matrix(self, gt_image, pre_image):
        gt_image = gt_image.data.float()
        pre_image = pre_image.data.float()
        mask = (gt_image >= 0) & (gt_image < self.num_class)
        label = self.num_class * gt_image[mask] + pre_image[mask]
        count = torch.bincount(label.int(), minlength=self.num_class ** 2)
        confusion_matrix = count[:self.num_class ** 2].reshape(self.num_class, self.num_class)  # int can't div 0
        return confusion_matrix.float()


class ConfusionMatrix(object):

    def __init__(self, nclass, classes=None):
        self.nclass = nclass
        self.classes = classes
        self.M = np.zeros((nclass, nclass))

    def add(self, gt, pred):
        assert(np.max(pred) <= self.nclass)
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if not gt[i] == 255:
                self.M[gt[i], pred[i]] += 1.0

    def addM(self, matrix):
        assert(matrix.shape == self.M.shape)
        self.M += matrix

    def __str__(self):
        pass

    def recall(self):
        recall = 0.0
        for i in range(self.nclass):
            recall += self.M[i, i] / np.sum(self.M[:, i])

        return recall/self.nclass

    def accuracy(self):
        accuracy = 0.0
        for i in range(self.nclass):
            accuracy += self.M[i, i] / np.sum(self.M[i, :])

        return accuracy/self.nclass

    def jaccard(self):
        jaccard = 0.0
        jaccard_perclass = []
        for i in range(self.nclass):
            jaccard_perclass.append(self.M[i, i] / (np.sum(self.M[i, :]) + np.sum(self.M[:, i]) - self.M[i, i]))

        return np.sum(jaccard_perclass)/len(jaccard_perclass), jaccard_perclass, self.M

    def generateM(self, item):
        gt, pred = item
        m = np.zeros((self.nclass, self.nclass))
        assert(len(gt) == len(pred))
        for i in range(len(gt)):
            if gt[i] < self.nclass: #and pred[i] < self.nclass:
                m[gt[i], pred[i]] += 1.0
        return m

if __name__ == '__main__':
    num = 1
    x = torch.randn(1,3,256,256)
    y = torch.ones(1,256,256)
    x = torch.argmax(x,1)
    print(x.shape)
    mio = MIOU(num,x,y)
    print(mio)