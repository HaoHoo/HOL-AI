import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import io

url="https://raw.githubusercontent.com/HaoHoo/HOL-AI/master/data/MNIST Test 10k 28x28 dense.csv"
stream=requests.get(url).content
frame=pd.read_csv(io.StringIO(stream.decode('utf-8')), sep='\t', header=0)
frame

data_np_array=frame.values
print(data_np_array)
print(data_np_array.shape)

select_img_array=data_np_array[999,1:]
print(select_img_array)
print(select_img_array.shape)

img_data=select_img_array.reshape((28,28))
print(img_data)
print(img_data.shape)

plt.gray()
plt.imshow(img_data)

arrdat=np.loadtxt("https://raw.githubusercontent.com/HaoHoo/HOL-AI/master/data/HOL_MNIST_test.csv", 'i2', delimiter=",")
imgdat=arrdat.reshape(10,28,28)
fig, axs = plt.subplots(2, 5)
fig.subplots_adjust(left=0.06, right=0.75, top=0.43, bottom=0.06, hspace=0, wspace=0.07)
images = []
k=0
for i in range(2):
    for j in range(5):
        images.append(axs[i,j].imshow(imgdat[k,]))
        axs[i,j].set_xticklabels('')
        axs[i,j].set_yticklabels('')
        axs[i,j].set_xticks([0,13,27])
        axs[i,j].set_yticks([0,13,27])
        k=k+1      
axs[1,0].set_xticklabels((0,13,27))
axs[1,0].set_yticklabels((27,13,0))
plt.show()