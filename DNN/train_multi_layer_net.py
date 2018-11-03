import sys
sys.path.append('../common/')
sys.path.append('../dataset/')
from optimizer import *
from trainer import Trainer
import spiral
from mnist import load_mnist
from multi_layer_net import Multi_layer_net
import matplotlib.pyplot as plt

# ハイパーパラメータの設定
max_epoch = 10
batch_size = 100
hidden_size_list = [50, 50]
learning_rate = 0.01

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

model = Multi_layer_net(x_train.shape[1], hidden_size_list, t_train.shape[1])
optimizer = AdaGrad(lr=learning_rate)

trainer = Trainer(model, optimizer)
trainer.fit(x_train, t_train, x_test, t_test, max_epoch, batch_size, eval_interval=100)
trainer.plot()

print('test accuracy : ' + str(model.accuracy(x_test, t_test)))



''' spiralデータセット用
# 境界領域のプロット
h = 0.001
x_min, x_max = x[:, 0].min() - .1, x[:, 0].max() + .1
y_min, y_max = x[:, 1].min() - .1, x[:, 1].max() + .1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
X = np.c_[xx.ravel(), yy.ravel()]
score = model.predict(X)
predict_cls = np.argmax(score, axis=1)
Z = predict_cls.reshape(xx.shape)
plt.contourf(xx, yy, Z)
plt.axis('off')

# データ点のプロット
x, t = spiral.load_data()
N = 100
CLS_NUM = 3
markers = ['o', 'x', '^']
for i in range(CLS_NUM):
    plt.scatter(x[i*N:(i+1)*N, 0], x[i*N:(i+1)*N, 1], s=40, marker=markers[i])
plt.show()
'''