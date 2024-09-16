import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imsave

img = np.load(r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_segnext_infer_npy\000001.npy')
plt.imshow(img)
plt.show()
