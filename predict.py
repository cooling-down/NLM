from cdmss import *

import pandas as pd
import numpy as np
import cv2

# 无需训练的NLM算法
def make_kernel(f):
    kernel = np.zeros((2*f+1, 2*f+1))
    for d in range(1, f+1):
        kernel[f-d:f+d+1, f-d:f+d+1] += (1.0/((2*d+1)**2))
    return kernel/kernel.sum()

def NLmeansfilter(I, h_=1, templateWindowSize=2,  searchWindowSize=4):
    f = templateWindowSize//2
    t = searchWindowSize//2
    height, width = I.shape[:2]
    padLength = f+t
    I2 = np.pad(I, padLength, 'symmetric')
    kernel = make_kernel(f)
    h = (h_**2)
    I_ = I2[padLength-f:padLength+f+height, padLength-f:padLength+f+width]

    average = np.zeros(I.shape)
    sweight = np.zeros(I.shape)
    wmax =  np.zeros(I.shape)
    for i in range(-t, t+1):
        for j in range(-t, t+1):
            if i==0 and j==0:
                continue
            I2_ = I2[padLength+i-f:padLength+i+f+height, padLength+j-f:padLength+j+f+width]
            w = np.exp(-cv2.filter2D((I2_ - I_)**2, -1, kernel)/h)[f:f+height, f:f+width]
            sweight += w
            wmax = np.maximum(wmax, w)
            average += (w*I2_[f:f+height, f:f+width])
    return (average+wmax*I)/(sweight+wmax)

def predict(cdmss):
    # cdmss---数据集
    data_x, data_y = cdmss.get_data(output_type='df')
    # cdmss---超参数
    sigma = cdmss.get_parameters()['sigma']
    templateWindowSize = cdmss.get_parameters()['templateWindowSize']
    searchWindowSize = cdmss.get_parameters()['searchWindowSize']
    y_predict=NLmeansfilter(data_y.to_frame().values, sigma,templateWindowSize, searchWindowSize)
    #########################################################################
    # 举例：预测绘图
    import matplotlib.pyplot as plt
    plt.figure(figsize=(20, 20))
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    font2 = {'family': 'Times New Roman',
             'weight': 'normal', 'size': 40, }
    plt.xlabel("TestSet", font2)
    plt.xticks(fontsize=30)
    plt.ylabel('Error', font2)
    plt.yticks(fontsize=30)
    # plt.scatter(range(len(bp_y_predict)), error, s=200, color='k', label='error')
    plt.plot(range(len(y_predict)), y_predict, color='r', label='error')
    # 预测结果 dataframe、numpy-array或者list形式
    y_predict_df = pd.DataFrame(y_predict)
    # cdmss---保存结果
    dic = {'img': [plt],
           'image_type': ['plt'],
           'result': y_predict_df,
           'result_type': 'df'}
    cdmss.save_result(dic)
    # cdmss---运行信息
    cdmss.save_message(code=10200, message='运行成功！')

    return "done"

if __name__ == "__main__":
    # cdmss = cdmss.cdmss(sys.argv)
    # 测试时使用的代码
    cdmss = cdmss([0, './param.json', './exampleWrite.json'])
    predict(cdmss)
