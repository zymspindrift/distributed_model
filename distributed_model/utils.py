#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author zym

import numpy as np

def find_input(i, j, fdr):
    '''
    利用流向矩阵判断有几个网格水流流向A
    :param i: 网格行号
    :param j: 网格列号
    :param fdr: 流向矩阵
    :return: k:input个数 output：具体的网格
    '''

    if fdr[i, j] == -9999:
        return 0, []
    k = 0  # k个网格指向i,j
    location = []  # 指向当前网格的点
    if fdr[i - 1, j - 1] == 2:
        k += 1
        location.append([i - 1, j - 1])
    if fdr[i - 1, j] == 4:
        k += 1
        location.append([i - 1, j])
    if fdr[i - 1, j + 1] == 8:
        k += 1
        location.append([i - 1, j + 1])
    if fdr[i, j - 1] == 1:
        k += 1
        location.append([i, j - 1])
    if fdr[i, j + 1] == 16:
        k += 1
        location.append([i, j + 1])
    if fdr[i + 1, j - 1] == 128:
        k += 1
        location.append([i + 1, j - 1])
    if fdr[i + 1, j] == 64:
        k += 1
        location.append([i + 1, j])
    if fdr[i + 1, j + 1] == 32:
        k += 1
        location.append([i + 1, j + 1])

    return k, location


def id_toij(grid_id, ncols=35, nrows=None):
    return [grid_id // ncols, grid_id % ncols]


def ij_toid(ij_list, ncols=35, n_rows=None):
    return (ij_list[0]) * ncols + ij_list[1]


def dc_calculate(vec1,vec2):
    '''
    计算确定性系数
    :param vec1: array1
    :param vec2: array2
    :return: dc
    '''

    SSE = np.square(vec1 - vec2)
    SST = np.square(vec2 - vec2.mean())
    dc = 1 - SSE.sum() / SST.sum()

    return dc


def time_lag(func):
    def inner(*args,**kwargs):
        import time
        t1 = time.time()
        func(*args,**kwargs)
        t2 = time.time()
        print('执行时间为：%fs' % (t2-t1))
    return inner


# 弧度转角度，并计算正切（tan）值
def tan(degree):
    return np.tan(np.radians(degree))