import sys
sys.path.append('../common/')
sys.path.append('../dataset/')
import numpy as np
import matplotlib.pyplot as plt
from mnist import load_mnist
from simple_convnet import SimpleConvNet
from optimizer import *
from trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

# 処理に時間のかかる場合はデータを削減 
x_train, t_train = x_train[:5000], t_train[:5000]
x_test, t_test = x_test[:1000], t_test[:1000]

# ハイパーパラメータの設定
max_epochs = 20
batch_size = 50

model = SimpleConvNet(input_dim=(1,28,28), 
                        conv_param = {'filter_num': 30, 'filter_size': 5, 'pad': 0, 'stride': 1},
                        hidden_size=100, output_size=10, weight_init_std=0.01)
optimizer = AdaGrad(lr=0.001)                     
trainer = Trainer(model, optimizer)
trainer.fit(x_train, t_train, x_test, t_test, max_epochs, batch_size, eval_interval=10)

# グラフの描画
trainer.plot()

print('test accuracy : ' + str(model.accuracy(x_test, t_test)))