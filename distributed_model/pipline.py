#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# author zym

from confluence import Confluence
from runoff_generation import Runoff,dt_range
from utils import *

from numba import jit
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from bayes_opt import BayesianOptimization

nrows = 38
ncols = 35
grid_nums = nrows * ncols

# tensor

# 日蒸发数据
day_evap = np.loadtxt(r'F:\我的论文\python程序\数据预处理\蒸发数据处理\day_evap.txt')
# 降雨数据
grid_rain = np.loadtxt(r'F:\我的论文\python程序\数据预处理\3000数据\网格雨量数据计算\grid_rain.txt')
# 读取真实流量数据
real_q = np.loadtxt(r'F:\我的论文\python程序\数据预处理\真实流量数据\真实流量数据.txt')

# 汇水栅格标记数据
grid_order = np.loadtxt(
    r'F:\我的论文\python程序\数据预处理\3000数据\汇水栅格统计\汇水栅格次序.txt',
    dtype=np.int)
grid_order = grid_order.flatten()  # 按格子数展平数据
# 河道栅格标记数据
grid_river = np.loadtxt(
    r'F:\我的论文\python程序\数据预处理\3000数据\riverway.txt',
    dtype=np.int)
rivers = np.argwhere(grid_river == 1)
river_mark_list = [[rivers[i, 0], rivers[i, 1]]
                   for i in range(rivers.shape[0])]
# 栅格流向数据，建立递归关系
fdr = np.loadtxt(r'F:\我的论文\python程序\数据预处理\3000数据\fdr3000.txt', skiprows=6)
# 栅格汇水累积量数据，估算河道宽度
fac = np.loadtxt(r'F:\我的论文\python程序\数据预处理\3000数据\fac3000.txt',skiprows=6)


# flow

class Distribute_predict():

    def __init__(self,wm, kc, ki, kg, b,a3, b3, a4, b4, cs, ci, cg,dt=24, dx=3):
        # 产流参数
        self.WM = wm  # 土壤蓄水容量
        self.KC = kc  # 蒸散发折算系数
        self.KI = ki  # 壤中径流出流系数
        self.KG = kg  # 地下径流出流系数
        self.B = b  # 反映坡度对出流的影响

        # 汇流参数
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

        # 时间步长、空间步长
        self.dt = dt
        self.dx = dx
        # 单位转换系数
        self.U = 3 * 3 / (3.6 * 24)
        # 出口位置
        self.output_i = 29
        self.output_j = 7

    # 模型产流计算
    @jit
    def _distributed_runoff(self):
        runoff = Runoff(self.WM, self.KC, self.KI, self.KG, self.B)
        runoff_result = []
        for grid_id in range(grid_nums):
            i,j = id_toij(grid_id)
            if fac[i,j] >= 0:
                r_all = runoff.eva_runoff(grid_id, grid_rain[grid_id, :], day_evap)
                runoff_result.append(r_all)
            else:
                runoff_result.append(-9999)  # 对于非汇水栅格标记为-9999，表示NoData

        return runoff_result

    # 模型汇流计算
    def distributed_confluence(self,output_ij_list):

        # 获取产流数据
        grid_runoff = self._distributed_runoff()

        # 汇流计算实例化
        confluence = Confluence(self.dt, self.dx, self.a3, self.b3, self.a4, self.b4, self.CS, self.CI, self.CG)

        # 河道栅格汇流计算

        def river_flow(i, j):
            k, grid_input = find_input(i, j, fdr)  # 找到所有指向该河道栅格的栅格
            not_river_list = [
                item for item in grid_input if item not in river_mark_list]
            river_list = [item for item in grid_input if item in river_mark_list]
            # 线性叠加所有非河道栅格的坡地汇流之后的过程
            R_not_river = np.zeros(day_evap.size)
            for ij in not_river_list:
                grid_id = ij_toid(ij, ncols)
                RS, RI, RG = grid_runoff[grid_id]
                # 坡面汇流之后的结果线性叠加（序列值）
                RS_slope = confluence.surface_confluence(RS)
                RI_slope = confluence.interflow_confluence(RI)
                RG_slope = confluence.underground_confluence_1(RG)
                R_not_river += (RS_slope + RI_slope + RG_slope)
            if not river_list:  # 到了河道栅格的源头了
                # 该栅格本身所产生的净雨作为旁侧入流处理
                grid_id = ij_toid([i,j])
                RS,RI,RG = grid_runoff[grid_id]
                qlat  = (RS + RI + RG) * self.U  # 旁侧入流
                # 此时所有上游栅格线性叠加作为计算栅格的入流过程
                R = np.zeros(day_evap.size)
                for dt_id in range(1,day_evap.size):
                    R[dt_id] = confluence.musking_cunge(
                        R_not_river[dt_id-1],R_not_river[dt_id],R[dt_id-1],qlat[dt_id],fac[i,j])
                return R  # 返回该河道栅格出流过程
            else:
                # 如果不是源头河道栅格，即该栅格上游仍有河道栅格汇入
                # 上游河道栅格的出流过程线性叠加作为计算栅格的入流过程
                R_in = np.zeros(day_evap.size)
                for ij in river_list:
                    R = river_flow(ij[0],ij[1])  # 递归运算，算法精髓！！
                    R_in += R
                # 坡面栅格的出流过程与栅格本身净雨（产流）作为旁侧入流
                grid_id = ij_toid([i,j])
                RS,RI,RG = grid_runoff[grid_id]  # 本身产流
                qlat = R_not_river + (RS + RI + RG) * self.U  # 旁侧入流
                R = np.zeros(day_evap.size)
                for dt_id in range(1,day_evap.size):
                    R[dt_id] = confluence.musking_cunge(
                        R_in[dt_id-1],R_in[dt_id],R[dt_id-1],qlat[dt_id],fac[i,j])
                return R

        q = np.zeros(day_evap.size)
        for ij_tuple in output_ij_list:
            q += river_flow(ij_tuple[0],ij_tuple[1])

        return q

        # # 非河道栅格汇流计算，栅格距离河道栅格最多也就五步，可以忽略  #todo
        # def overland_flow(i,j):
        #     k,grid_input = find_input(i,j,fdr)
        #     # 找出非河道集水栅格
        #     grid_input = [item for item in grid_input if item not in river_mark_list]
        #     if not grid_input: # 当此栅格为源头栅格时

# 可视化
def q_vision(q,real_q):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y%m'))

    ax.set_xlabel('时间')
    ax.set_ylabel('流量' + '$\mathrm{m}^{3}$' + '/s')
    ax.plot(dt_range,q,label='预测流量')
    ax.plot(dt_range,real_q,label='实测流量')
    ax.legend(loc=1)
    plt.show()

@time_lag
def main(wm, kc, ki, kg, b, a3, b3, a4, b4, cs, ci, cg):
    distribute_predict = Distribute_predict(wm, kc, ki, kg, b,a3, b3, a4, b4, cs, ci, cg)
    q = distribute_predict.distributed_confluence([(29, 7), (31, 11), (28, 16), (8,31), (34,8)])
    q_vision(q,real_q)
    dc = dc_calculate(q,real_q)
    print(f'计算总径流：{q.sum()}')
    print(f'实测总径流：{real_q.sum()}')
    print(f'确定性系数dc值为：{dc}')

    # return dc

# 贝叶斯优化调参
def bayes_optimize():
    bo = BayesianOptimization(main,{'wm':(100,220),
                                    'kc':(0.6,1.2),
                                    'ki':(0.2,0.6),
                                    'kg':(0.2,0.6),
                                    'b':(2,2),
                                    'a3':(0.1,1),
                                    'b3':(0.1,1),
                                    'a4':(0.5,2),
                                    'b4':(1,3),
                                    'cs':(0.4,0.7),
                                    'ci':(0.5,0.9),
                                    'cg':(0.9,0.998)})
    bo.explore({'wm':[150],'kc':[0.6],'ki':[0.35],'kg':[0.35],'b':[2],
                'a3':[0.5],'b3':[0.5],'a4':[1.15],'b4':[1.78],
                'cs':[0.6],'ci':[0.75],'cg':[0.995]})
    bo.maximize(init_points=10,acq='poi')
    print(bo.res['max'])

if __name__ == '__main__':
    main(150,0.6,0.35,0.35,2,0.5,0.5,1.15,1.78,0.6,0.75,0.995)
    # bayes_optimize()
