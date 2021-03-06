import sys, os
sys.path.append('../common/')
import numpy as np
import time
import matplotlib.pyplot as plt
from optimizer import *

class Trainer:
    '''
    ニューラルネットの学習を行うクラス
    '''

    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer

        self.loss_list = []
        self.test_loss_list = []
        self.eval_interval = None
        self.current_epoch = 0

    def fit(self, x_train, t_train, x_test=None, t_test=None, max_epoch=10, batch_size=32, max_grad=None, eval_interval=20):
        data_size = len(x_train)
        max_iters = data_size // batch_size
        self.eval_interval = eval_interval
        model, optimizer = self.model, self.optimizer
        total_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # データのシャッフル
            idx = np.random.permutation(np.arange(data_size))
            x_train = x_train[idx]
            t_train = t_train[idx]

            for iters in range(max_iters):
                batch_x = x_train[iters*batch_size : (iters+1)*batch_size]
                batch_t = t_train[iters*batch_size : (iters+1)*batch_size]

                # 勾配を求め、パラメータを更新
                loss = model.forward(batch_x, batch_t)
                model.backward()
                params, grads = remove_duplicate(model.params, model.grads)    # 共有された重みを1つに集約
                if max_grad is not None:
                    clip_grads(grads, max_grad)
                optimizer.update(params, grads)
                total_loss += loss
                loss_count += 1
                
                # 評価
                if (eval_interval is not None) and (iters % eval_interval) == 0:
                    avg_loss = total_loss / loss_count
                    elapsed_time = time.time() - start_time
                    if (x_test is not None) and (t_test is not None):
                        test_loss = model.forward(x_test, t_test)
                        self.test_loss_list.append(test_loss)
                        print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f | test_loss %.2f'
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss, test_loss))
                    else :
                        print('| epoch %d |  iter %d / %d | time %d[s] | loss %.2f '
                          % (self.current_epoch + 1, iters + 1, max_iters, elapsed_time, avg_loss))
                    self.loss_list.append(float(avg_loss))
                    total_loss, loss_count = 0, 0
            
            self.current_epoch += 1

    def plot(self, ylim=None):
        x = np.arange(len(self.loss_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train', color='blue')
        if len(self.test_loss_list) != 0:
            plt.plot(x, self.test_loss_list, label='test',color='red')
        plt.xlabel('iterations (x' + str(self.eval_interval) + ')')
        plt.ylabel('loss')
        plt.legend()
        plt.show()


def remove_duplicate(params, grads):
    '''
    パラメータ配列中の重複する重みをひとつに集約し、
    その重みに対応する勾配を加算する
    '''
    params, grads = params[:], grads[:]  # copy list

    while True:
        find_flg = False
        L = len(params)

        for i in range(0, L - 1):
            for j in range(i + 1, L):
                # 重みを共有する場合
                if params[i] is params[j]:
                    grads[i] += grads[j]  # 勾配の加算
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)
                # 転置行列として重みを共有する場合（weight tying）
                elif params[i].ndim == 2 and params[j].ndim == 2 and \
                     params[i].T.shape == params[j].shape and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    find_flg = True
                    params.pop(j)
                    grads.pop(j)

                if find_flg: break
            if find_flg: break

        if not find_flg: break

    return params, grads