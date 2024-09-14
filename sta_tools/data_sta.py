# -*- coding: utf-8 -*-
# @Author : Zhao HL
# @contact: huilin16@qq.com
# @Time   : 2021/1/30 10:05
# @File   : data_sta.py
# @Description
'''
数据统计方法，包括：
    *统计影像shape分布；
    *统计影像各通道mean、std统计值；
    *统计gt各类数量分布；
    *统计gt类别唯一值；
    *统计obj信息：obj面积分布、各gt含有的obj个数分布
'''

from __future__ import print_function
from __future__ import division
import os
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from skimage import io
import matplotlib.pyplot as plt
from collections import namedtuple

Cls = namedtuple('cls', ['name', 'id', 'color'])
Clss = [
    Cls('bg', 0, (0, 0, 0)),
    Cls('cls1', 1, (255, 128, 128)),
    Cls('cls2', 2, (255, 255, 128)),
    Cls('cls3', 3, (128, 255, 128)),
]
Img_chs = 3


def img_read(img_path):
    img = io.imread(img_path)
    return img
def gt_read(img_path):
    img = io.imread(img_path)
    return img




# region shape sta

def dir_shape_sta(imgs_path, save_path):
    '''
    统计影像shape分布，并绘制二维分布图，以文件夹方式进行
    :param imgs_path: 影像文件夹路径
    :param save_path: csv结果路径
    :return:
    '''
    print('images shape sta:')
    imgs_list = [os.path.join(imgs_path, file_name) for file_name in os.listdir(imgs_path)]
    list_shape_sta(imgs_list, save_path)
    print('finish\n')


def list_shape_sta(imgs_list, save_path):
    '''
    统计影像shape分布，并绘制二维分布图，以list方式进行
    :param imgs_list: 影像文件路径list
    :param save_path: csv结果路径
    :return:
    '''
    shape_df = pd.DataFrame(None, columns=['height', 'width'])
    with tqdm(imgs_list) as pbar:
        pbar.set_description('shape sta ')
        for index, img_path in enumerate(pbar):
            img = img_read(img_path)
            shape_df.loc[index] = [img.shape[0], img.shape[1]]

    sns.jointplot(x='height', y='width', data=shape_df)
    plt.savefig(save_path.replace('.csv', '.png'))

    shape_df['filename'] = [os.path.basename(gt_path) for gt_path in imgs_list]
    shape_df.to_csv(save_path)
    print('save to %s' % save_path)

# endregion


# region mean、std sta

def dir_ms_sta(imgs_path, save_path, img_chs=Img_chs):
    '''
    统计影像各通道mean、std统计值，以文件夹方式进行
    :param imgs_path: 影像文件夹路径
    :param save_path: csv结果路径
    :return:
    '''
    print('images mean std sta:')
    imgs_list = [os.path.join(imgs_path, file_name) for file_name in os.listdir(imgs_path)]
    list_ms_sta(imgs_list, save_path, img_chs=img_chs)
    print('finish\n')


def list_ms_sta(imgs_list, save_path, img_chs=Img_chs):
    '''
    统计影像各通道mean、std统计值，以list方式进行
    :param imgs_list: 影像文件路径list
    :param save_path: csv结果路径
    :param img_chs: 影像通道数
    :return:
    '''
    ms_df = pd.DataFrame(None, columns=['mean' + str(i + 1) for i in range(img_chs)] +
                                       ['std' + str(i + 1) for i in range(img_chs)])
    with tqdm(imgs_list) as pbar:
        pbar.set_description('mean std sta ')
        for index, img_path in enumerate(pbar):
            img = img_read(img_path)
            mean = np.mean(img, axis=(0, 1))
            std = np.std(img, axis=(0, 1))
            ms_df.loc[index] = np.append(mean, std)


    ms_df.loc['Col_avg'] = ms_df.apply(lambda x: x.mean())
    ms_df.loc['Col_avgTensor'] = ms_df.loc['Col_avg'].apply(lambda x: x / 255)
    ms_df.to_csv(save_path)
    print(ms_df[-2:])
    print('save to %s' % save_path)

# endregion


# region class sta

def dir_class_sta(gts_path, save_path, clss=Clss):
    '''
    统计gt各类数量分布，以文件夹方式进行
    :param gts_path: gt文件夹路径
    :param save_path: csv结果保存路径
    :param clss: 类别映射表
    :return:
    '''
    print('gts class sta:')
    gts_list = [os.path.join(gts_path, file_name) for file_name in os.listdir(gts_path)]
    list_class_sta(gts_list, save_path, clss=clss)
    print('finish\n')


def list_class_sta(gts_list, save_path, clss=Clss):
    '''
    统计gt各类数量分布，以list方式进行
    :param gts_list: gt文件路径list
    :param save_path: csv结果保存路径
    :param clss: 类别映射表
    :return:
    '''
    class_df = pd.DataFrame(None, columns=[cls.name for cls in clss])
    with tqdm(gts_list) as pbar:
        pbar.set_description('class sta ')
        for index, gt_path in enumerate(pbar):
            gt = gt_read(gt_path)
            cls_sta = []
            for cls in clss:
                cls_sta.append(np.sum(gt == cls.id))
            class_df.loc[index] = cls_sta


    class_df.loc['Pixel_sum'] = class_df.apply(lambda x: x.sum())
    class_df.loc['Pixel_pct'] = class_df.loc['Pixel_sum'] / class_df.loc['Pixel_sum'].sum()
    class_df.loc['Pixel_pct'].plot(kind='bar', title='Pixel_pct')
    plt.savefig(save_path.replace('.csv', '.png'))
    plt.close()

    class_df.loc['Sample_sum'] = class_df.apply(lambda x: np.sum(x > 0) - 1)
    class_df.loc['Sample_pct'] = class_df.loc['Sample_sum'] / class_df.loc['Sample_sum'].sum()
    class_df.loc['Sample_pct'].plot(kind='bar', title='Sample_pct')
    plt.savefig(save_path.replace('.csv', '_sample.png'))
    plt.close()

    class_df['filename'] = [os.path.basename(gt_path) for gt_path in gts_list] + ['', '', '', '']
    class_df.to_csv(save_path)
    class_df.pop('filename')
    print(class_df[-4:])
    print('save to %s' % save_path)



def dir_range_sta(imgs_path, save_path, img_chs=Img_chs):
    '''
    统计影像各通道range，以文件夹方式进行
    :param imgs_path: 影像文件夹路径
    :param save_path: csv结果路径
    :return:
    '''
    print('images pixel range:')
    imgs_list = [os.path.join(imgs_path, file_name) for file_name in os.listdir(imgs_path)]
    list_range_sta1(imgs_list, save_path, img_chs=img_chs)
    print('finish\n')


def list_range_sta1(imgs_list, save_path, img_chs=Img_chs):
    '''
    统计影像各通道range，以list方式进行
    :param imgs_list: 影像文件路径list
    :param save_path: csv结果路径
    :param img_chs: 影像通道数
    :return:
    '''
    ms_df = pd.DataFrame(None, columns=['max%d'%i for i in range(img_chs)] +
                                       ['min%d'%i for i in range(img_chs)])
    with tqdm(imgs_list) as pbar:
        pbar.set_description('pixel range sta ')
        for index, img_path in enumerate(pbar):
            img = img_read(img_path)

            max_value = np.max(img, axis=(0, 1))
            min_value = np.min(img, axis=(0, 1))
            ms_df.loc[index] = np.append(max_value, min_value)
    colors = ['r', 'g', 'b']
    for i in range(img_chs):
        print('band %d max value range: %g~%g'%(i, ms_df['max%d'%i].min(), ms_df['max%d'%i].max()))
        sns.kdeplot(data=ms_df, x="max%d"%i, color=colors[i], label=f'band %d Max'%i, fill=True)
    plt.legend()
    plt.title('Distribution of Max Values for Each Band')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(save_path.replace('.csv', '_bands_max.png'))
    plt.close()


    for i in range(img_chs):
        print('band %d min value range: %g~%g'%(i, ms_df['min%d'%i].min(), ms_df['min%d'%i].max()))
        sns.kdeplot(data=ms_df, x="min%d"%i, color=colors[i], label=f'band %d Min'%i, fill=True)
    plt.legend()
    plt.title('Distribution of Min Values for Each Band')
    plt.xlabel('Value')
    plt.ylabel('Density')
    plt.savefig(save_path.replace('.csv', '_bands_min.png'))
    plt.close()

# endregion



if __name__ == '__main__':
    pass
    img_dir_train = r'D:\202410_defecttest\NEU_Seg_Data\images\training'
    img_dir_test = r'D:\202410_defecttest\NEU_Seg_Data\images\test'
    gt_dir_train = r'D:\202410_defecttest\NEU_Seg_Data\annotations\training'
    gt_dir_test = r'D:\202410_defecttest\NEU_Seg_Data\annotations\test'
    os.makedirs(r'D:\202410_defecttest\NEU_Seg_Data\sta', exist_ok=True)

    # dir_shape_sta(img_dir_train, r'D:\202410_defecttest\NEU_Seg_Data\sta\shape_train.csv')
    # dir_shape_sta(img_dir_test, r'D:\202410_defecttest\NEU_Seg_Data\sta\shape_test.csv')
    # dir_class_sta(gt_dir_train, r'D:\202410_defecttest\NEU_Seg_Data\sta\gt_train.csv')
    # dir_class_sta(gt_dir_test, r'D:\202410_defecttest\NEU_Seg_Data\sta\gt_test.csv')
    # dir_ms_sta(img_dir_train, r'D:\202410_defecttest\NEU_Seg_Data\sta\ms_train.csv')
    # dir_ms_sta(img_dir_test, r'D:\202410_defecttest\NEU_Seg_Data\sta\ms_test.csv')
    # dir_range_sta(img_dir_train, r'D:\202410_defecttest\NEU_Seg_Data\sta\range_train.csv')
    # dir_range_sta(img_dir_test, r'D:\202410_defecttest\NEU_Seg_Data\sta\range_test.csv')