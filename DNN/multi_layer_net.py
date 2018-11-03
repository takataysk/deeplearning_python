import sys, os
sys.path.append('../common/')
import numpy as np
from layers import *

class Multi_layer_net:
    '''
    多層全結合ニューラルネットワーク

    Parameters
    -------------
    input_size : 入力サイズ
    hidden_size_list : 隠れ層のニューロンの数のリスト(e.g. [100, 100, 100])
    output_size : 出力サイズ
    activation : 'relu' or 'sigmoid'
    weight_init_std : 重みの標準偏差を指定(e.g. 0.01)
        'relu'または'he'を指定した場合は「Xavierの初期値」を設定
        'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
    
    '''

    def __init__(self, input_size, hidden_size_list, output_size,
                activation='relu', weight_init_std='relu'):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size_list = hidden_size_list
        self.hidden_layer_num = len(hidden_size_list)
        self.params = {}

        # 重みの初期化
        self.__init_weight(weight_init_std)

        # レイヤの生成
        actvation_layer = {'sigmoid' : Sigmoid, 'relu' : Relu}
        self.layers = []
        for idx in range(1, self.hidden_layer_num+2):
            if idx == self.hidden_layer_num:    # 最終行の場合活性化関数は不要
                self.layers.append(Affine(self.params['W' + str(idx)], self.params['b' + str(idx)]))
            else:
                self.layers.append(Affine(self.params['W' + str(idx)], self.params['b' + str(idx)]))
                self.layers.append(actvation_layer[activation]())

        self.loss_layer = SoftmaxWithLoss()

        # 全ての重みと勾配をリストにまとめる
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads

    def __init_weight(self, weight_init_std='relu'):
        """重みの初期値設定
        Parameters
        ----------
        weight_init_std : 重みの標準偏差を指定（e.g. 0.01）
            'relu'または'he'を指定した場合は「Heの初期値」を設定
            'sigmoid'または'xavier'を指定した場合は「Xavierの初期値」を設定
        """
        all_size_list = [self.input_size] + self.hidden_size_list + [self.output_size]
        for idx in range(1, len(all_size_list)):
            scale = weight_init_std
            if str(weight_init_std).lower() in ('relu', 'he'):
                scale = np.sqrt(2.0 / all_size_list[idx - 1])  # ReLUを使う場合に推奨される初期値
            elif str(weight_init_std).lower() in ('sigmoid', 'xavier'):
                scale = np.sqrt(1.0 / all_size_list[idx - 1])  # sigmoidを使う場合に推奨される初期値
            self.params['W' + str(idx)] = scale * np.random.randn(all_size_list[idx-1], all_size_list[idx])
            self.params['b' + str(idx)] = np.zeros(all_size_list[idx])
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
    
    def forward(self, x, t):
        score = self.predict(x)
        loss = self.loss_layer.forward(score, t)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy