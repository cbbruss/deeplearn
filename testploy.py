import numpy as np


# Do on server
train_loss = np.array([i["train_loss"] for i in net1.train_history_])
valid_loss = np.array([i["valid_loss"] for i in net1.train_history_])

np.savetxt("train_loss.csv", train_loss, delimiter=",")
np.savetxt("valid_loss.csv", valid_loss, delimiter=",")

from numpy import genfromtxt
train_loss = genfromtxt('train_loss.csv', delimiter=',')
valid_loss = genfromtxt('valid_loss.csv', delimiter=',')

# Transfer to local machine
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.plot(train_loss, linewidth=3, label="train")
plt.plot(valid_loss, linewidth=3, label="valid")
plt.grid()
plt.legend()
plt.xlabel("epoch")
plt.ylabel("loss")
plt.ylim(1e-3, 1e-2)
plt.yscale("log")
plt.savefig('display.png')