#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author zym

import numpy as np
import pandas as pd
from utils import tan

# 时间数据
dt_range = pd.date_range('2007-01-01', '2013-12-31', freq='D')
# LAI数据
lai_perGrid_perMonth = np.loadtxt(
    r'F:\我的论文\python程序\网格型分布式水文模型\lai_everyGrid_everyMonth.txt')
# 坡度数据
slope_data = np.loadtxt(
    r'F:\我的论文\数据\3km数据\slope_data.txt',
    dtype=np.float32,
    skiprows=6)
slope_data = slope_data.flatten()  # 格子都是压平了算的


class Runoff():

    def __init__(self, wm, kc, ki, kg, b):
        self.WM = wm  # 土壤蓄水容量
        self.KC = kc  # 蒸散发折算系数
        self.KI = ki  # 壤中径流出流系数
        self.KG = kg  # 地下径流出流系数
        self.B = b  # 反映坡度对出流的影响

    def eva_runoff(self, grid_id, P, EM):
        '''

        :param grid_id: flatten之后的序列号
        :param P: 降雨序列
        :param EM: 蒸发序列
        :return: 三水源序列
        '''
        # 土壤水容量
        W = np.zeros(EM.shape[0], dtype=np.float32)
        W[0] = self.WM  # 先假设么一个格子的土壤水容量相同为150
        # 流域蒸散发能力数组
        EP = self.KC * EM
        # 三水源产流
        RS = np.zeros(EM.shape[0], dtype=np.float32)  # 地表径流
        RI = np.zeros(EM.shape[0], dtype=np.float32)  # 壤中径流
        RG = np.zeros(EM.shape[0], dtype=np.float32)  # 地下径流

        for i in range(EM.shape[0]):  # 锁定到每一天
            date = dt_range[i]  # 拿到这一天的时间进而拿到今天所属的月份
            # 蒸发计算
            ia = 0.15 * lai_perGrid_perMonth[grid_id, date.month - 1]  # 冠层截流
            if ia < EP[i]:
                e1 = ia
                e2 = (EP[i] - e1) * W[i] / self.WM
                e = e1 + e2
            else:
                e = EP[i]

            # 产流计算 分未蓄满、达到田间持水量、达到饱和含水量三个过程
            if i <= P.shape[0] - 2:
                if W[i] + P[i] - e <= self.WM:  # 此时尚未达到田间持水量
                    W[i + 1] = W[i] + P[i] - e
                    RS[i] = 0  # 不产流
                    RI[i] = 0
                    RG[i] = 0
                elif W[i] + P[i] - e <= self.WM / 0.7:  # 此时已达到田间持水量尚未达到饱和含水量
                    RS[i] = 0
                    RI[i] = self.\
                                KI * W[i] * tan(slope_data[grid_id]) ** self.B
                    RG[i] = self.KG * W[i] * tan(slope_data[grid_id]) ** self.B
                    W[i + 1] = W[i] + P[i] - e - (RI[i] + RG[i])
                else:  # 此时达到饱和含水量
                    RI[i] = self.KI * W[i] * tan(slope_data[grid_id]) ** self.B
                    RG[i] = self.KG * W[i] * tan(slope_data[grid_id]) ** self.B
                    RS[i] = W[i] + P[i] - e - RI[i] - RG[i] - self.WM / 0.7
                    W[i + 1] = self.WM / 0.7
        return [RS, RI, RG]  # 到底需不需要做单元格内坡面汇流呢 todo
