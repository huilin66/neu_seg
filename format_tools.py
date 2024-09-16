import os
import numpy as np
from skimage import io
from pathlib import Path
from tqdm import tqdm


def images_to_npy(image_dir, output_dir):
    img_list = os.listdir(image_dir)

    os.makedirs(output_dir, exist_ok=True)
    for image_name in tqdm(img_list):
        image_path = os.path.join(image_dir, image_name)
        img = io.imread(image_path)
        np.save(os.path.join(output_dir, Path(image_name).stem+'.npy'), img)




if __name__ == '__main__':
    # 使用示例
    image_dir = r'E:\repository\mmsegmentation\data\NEU_Seg\annotations\test'
    output_dir = r'E:\repository\mmsegmentation\data\NEU_Seg\images\test_gt'
    images_to_npy(image_dir, output_dir)
