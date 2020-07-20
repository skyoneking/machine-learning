import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO
import math

# 获取图片尺寸
def img_size(img):
    img_file = BytesIO()
    image = Image.fromarray(np.uint8(img))
    print(len(set(image.getdata())))
    image.save(img_file, 'jpeg')
    return img_file.tell()/1024

# 图片源
img_ori = Image.open('test.jpg')
X = np.array(img_ori)

# 3个颜色通道(高为观察值，宽为特征值)
X = X.transpose((2,0,1))
X_means = X.mean(axis=1).mean(axis=1).reshape(-1,1,1)
X = X - X_means # 均值0化
h = X.shape[1]
u_R, s_R, _v = np.linalg.svd(np.dot(X[0].T, X[0]) / h)
u_G, s_G, _v = np.linalg.svd(np.dot(X[1].T, X[1]) / h)
u_B, s_B, _v = np.linalg.svd(np.dot(X[2].T, X[2]) / h)

k = 400
u_R_reduce = u_R[:,:k]
u_G_reduce = u_G[:,:k]
u_B_reduce = u_B[:,:k]

R = np.dot(X[0], u_R_reduce).dot(u_R_reduce.T)
G = np.dot(X[1], u_G_reduce).dot(u_G_reduce.T)
B = np.dot(X[2], u_B_reduce).dot(u_B_reduce.T)

R = ((R - R.min())/(R.max()-R.min()))
G = ((G - G.min())/(G.max()-G.min()))
B = ((B - B.min())/(B.max()-B.min()))

img_temp = (np.array([R,G,B]).transpose((1,2,0))*255).astype(np.uint8)
print(img_size(img_ori))
print(img_size(img_temp))
plt.subplot(121)
plt.imshow(np.array(img_ori))
plt.axis('off')
plt.subplot(122)
plt.imshow(img_temp)
plt.axis('off')
plt.show()