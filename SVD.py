import numpy as np
import random


class SVD:
    def __init__(self, mat, K=20):
        self.mat = np.array(mat)
        # mat: 评分矩阵 首先对数据进行数组化
        self.K = K
        self.bi = {}
        self.bu = {}
        self.qi = {}
        self.pu = {}
        self.avg = np.mean(self.mat[:2])
        for i in range(self.mat.shape[0]):
            uid = self.mat[i, 0]
            iid = self.mat[i, 1]
            self.bi.setdefault(iid, 0)
            self.bu.setdefault(uid, 0)
            self.qi.setdefault(iid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))
            self.pu.setdefault(uid, np.random.random((self.K, 1)) / 10 * np.sqrt(self.K))

    def predict(self, uid, iid):
        self.bi.setdefault(iid, 0)
        self.bu.setdefault(uid, 0)
        self.qi.setdefault(iid, np.zeros(self.K, 1))
        self.pu.setdefault(uid, np.zeros(self.K, 1))
        rating = self.avg + self.bi[iid] + self.bu[uid] + np.sum(self.qi[iid] * self.pu[uid])
        if rating > 5:
            rating = 5
        if rating < 1:
            rating = 1
        return rating

    def train(self, steps=30, gamma=0.04, Lambda=0.15):
        print('train data size', self.mat.shape)
        for step in range(steps):
            print('step', step + 1, 'is running')
            """
            shuffle与permutation的区别:
            函数shuffle与permutation都是对原来的数组进行重新洗牌（即随机打乱原来的元素顺序）；
            区别在于shuffle直接在原来的数组上进行操作，改变原来数组的顺序，无返回值。
            而permutation不直接在原来的数组上进行操作，而是返回一个新的打乱顺序的数组，并不改变原来的数组。
            """
            KK = np.random.permutation(self.mat.shape[0])
            RMSE = 0.0
            for i in range(self.mat.shape[0]):
                j = KK[i]
                uid = self.mat[j, 0]
                iid = self.mat[j, 1]
                rating = self.mat[j, 2]
                eui = rating - self.predict(uid, iid)
                RMSE += eui ** 2
                self.bu[uid] += gamma * (eui - Lambda * self.bu[uid])
                self.bi[iid] += gamma * (eui - Lambda * self.bi[iid])
                tmp = self.qi[iid]
                self.qi[iid] += gamma * (eui * self.pu[uid] - Lambda * self.qi[iid])
                self.pu[uid] += gamma * (eui * tmp - Lambda * self.pu[uid])
            gamma = 0.93 * gamma
            # gamma以0.93的学习率递减
        print('RMSE', np.sqrt(RMSE / self.mat.shape[0]))

    def test(self, test_data):
        test_data = np.array(test_data)
        print('test_data size:', test_data.shape)
        RMSE = 0.0
        for i in range(test_data.shape[0]):
            uid = test_data[i, 0]
            iid = test_data[i, 1]
            rating = test_data[i, 2]
            eui = rating - self.predict(uid, iid)
            RMSE += eui ** 2
        print('RMSE of test data is', np.sqrt(RMSE / test_data.shape[0]))


def getData():
    with open('your path', 'r')as f:
        lines = f.readlines()
        data = []
        for line in lines:
            list = line.split('\t\n')
            if int(list[2] != 0):
                data.append([int(i) for i in list[:3]])
        random.shuffle(data)
        train_data = data[:int(len(data) * 0.7)]
        test_data = data[int(len(data)) * 0.7:]
        print('load data fininshed')
        print("data size:", len(data))
        return train_data, test_data


if __name__ == '__main__':
    train_data, test_data = getData()
    a = SVD(train_data, 30)
    a.train()
    a.test(test_data)
