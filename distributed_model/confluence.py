#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author zym

import numpy as np


class Confluence():

    def __init__(self, dt, dx, a3, b3, a4, b4, cs, ci, cg):
        # 时间步长、空间步长
        self.dt = dt
        self.dx = dx
        # 波速计算的经验参数
        self.a3 = a3
        self.b3 = b3
        # 河道宽度计算的经验参数
        self.a4 = a4
        self.b4 = b4
        # 地表水线性水库汇流系数 todo
        self.CS = cs
        # 壤中流线性水库退水系数
        self.CI = ci
        # 地下水线型水库退水系数
        self.CG = cg
        # 非线性水库蓄泄系数
        self.a = 0.85

        # 单位转换系数
        self.U = 3 * 3 / (3.6 * 24)
        # 平均河底比降（河床坡度，查资料所得）
        self.s0 = 1.35 / 1000

    def __repr__(self):
        return "几种汇流演算方法"

    # 马斯京根-康吉演算法
    def musking_cunge(self, in1, in2, o1, qlat, fac):

        # 假设流速为流量的函数，流量以上游栅格的出流代替
        c = self.a3 * in2 ** self.b3
        if c < 0.01:
            c = 0.01
        # print(c)
        # 假设河宽是集水面积的函数
        b = self.a4 * np.log(fac) + self.b4
        k = self.dx / c
        x = 0.5 * (1 - qlat / (c * self.dx * b * self.s0))
        c1 = (self.dt / k + 2 * x) / (self.dt / k + 2 * (1 - x))
        c2 = (self.dt / k - 2 * x) / (self.dt / k + 2 * (1 - x))
        c3 = (2 * (1 - x) - self.dt / k) / (self.dt / k + 2 * (1 - x))
        c4 = 2 * self.dt / k / (self.dt / k + 2 * (1 - x))
        o2 = c1 * in1 + c2 * in2 + c3 * o1 + c4 * qlat

        return max(o2, 0)

    # 壤中流线性水库法汇流计算
    def interflow_confluence(self, q):
        trss = np.zeros(q.size)
        for i in range(1, q.size):
            trss[i] = trss[i - 1] * self.CI + q[i] * (1 - self.CI) * self.U

        return trss

    # 地下水线型水库法汇流计算
    def underground_confluence(self, q):
        trg = np.zeros(q.size)
        for i in range(1, q.size):
            trg[i] = trg[i - 1] * self.CG + q[i] * (1 - self.CG) * self.U

        return trg

    # 地下水非线性水库演算法  # todo
    def underground_confluence_1(self, q):
        trg = np.zeros(q.size)
        for i in range(1, q.size):
            trg[i] = (max((2 * self.a / self.dt * trg[i - 1]**0.5 - trg[i - 1] +
                       self.a**2 / self.dt**2 + 2 * q[i]),0)**0.5 - self.a / self.dt)**2
        return trg

    # 地表水线性水库演算法
    def surface_confluence(self, q):
        trs = np.zeros(q.size)
        for i in range(1, q.size):
            trs[i] = self.CS * trs[i - 1] + (1 - self.CS) * q[i] * self.U

        return trs
