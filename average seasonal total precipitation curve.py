import numpy as np  # 用于处理数组和数值计算
import xarray as xr  # 用于处理NetCDF文件
import matplotlib.pyplot as plt  # 用于绘图
import pandas as pd  # 用于数据处理和分析
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 设置文件路径
ncFilePath1 = "ERA5_1940-2022_Total_precipitation_China_PCR.nc"
ncFilePath2 = "ERA5_1940-2022_Total_precipitation_China_TT.nc"
ncFilePath3 = "ERA5_1940-2022_Total_precipitation_China_SC.nc"
outputfilePath = 'average seasonal total precipitation curve.png'

# 加载nc数据
ncDataset1 = xr.open_dataset(ncFilePath1)
ncDataset2 = xr.open_dataset(ncFilePath2)
ncDataset3 = xr.open_dataset(ncFilePath3)

# 数据处理 ------------------------------------------------------------------------------------
# 转换时间数据为 Pandas 的时间戳格式
time_stamps_pd = pd.to_datetime(ncDataset1["time"])
# 计算每个月的天数
days_in_each_month = time_stamps_pd.to_series().dt.days_in_month
# 将每个时间步的降水数据乘以对应月份的天数并转换为毫米
for i in range(len(days_in_each_month.values)):
    ncDataset1["tp"][i, :, :] = ncDataset1["tp"][i, :, :] * days_in_each_month.values[i] * 1000
    ncDataset2["tp"][i, :, :] = ncDataset2["tp"][i, :, :] * days_in_each_month.values[i] * 1000
    ncDataset3["tp"][i, :, :] = ncDataset3["tp"][i, :, :] * days_in_each_month.values[i] * 1000

mean_data_each_time1 = ncDataset1["tp"].mean(dim=('latitude', 'longitude'))
mean_data_each_time2 = ncDataset2["tp"].mean(dim=('latitude', 'longitude'))
mean_data_each_time3 = ncDataset3["tp"].mean(dim=('latitude', 'longitude'))

data1 = mean_data_each_time1.groupby('time.year').sum(dim='time')
data2 = mean_data_each_time2.groupby('time.year').sum(dim='time')
data3 = mean_data_each_time3.groupby('time.year').sum(dim='time')

# 根据时间划分为四个季节（假设时间以月为单位）
spring1 = mean_data_each_time1.sel(
    time=(mean_data_each_time1['time.month'] >= 3) & (mean_data_each_time1['time.month'] <= 5))
summer1 = mean_data_each_time1.sel(
    time=(mean_data_each_time1['time.month'] >= 6) & (mean_data_each_time1['time.month'] <= 8))
autumn1 = mean_data_each_time1.sel(
    time=(mean_data_each_time1['time.month'] >= 9) & (mean_data_each_time1['time.month'] <= 11))
winter1 = mean_data_each_time1.sel(
    time=(mean_data_each_time1['time.month'] == 12) | (mean_data_each_time1['time.month'] <= 2))

spring2 = mean_data_each_time2.sel(
    time=(mean_data_each_time2['time.month'] >= 3) & (mean_data_each_time2['time.month'] <= 5))
summer2 = mean_data_each_time2.sel(
    time=(mean_data_each_time2['time.month'] >= 6) & (mean_data_each_time2['time.month'] <= 8))
autumn2 = mean_data_each_time2.sel(
    time=(mean_data_each_time2['time.month'] >= 9) & (mean_data_each_time2['time.month'] <= 11))
winter2 = mean_data_each_time2.sel(
    time=(mean_data_each_time2['time.month'] == 12) | (mean_data_each_time2['time.month'] <= 2))

spring3 = mean_data_each_time3.sel(
    time=(mean_data_each_time3['time.month'] >= 3) & (mean_data_each_time3['time.month'] <= 5))
summer3 = mean_data_each_time3.sel(
    time=(mean_data_each_time3['time.month'] >= 6) & (mean_data_each_time3['time.month'] <= 8))
autumn3 = mean_data_each_time3.sel(
    time=(mean_data_each_time3['time.month'] >= 9) & (mean_data_each_time3['time.month'] <= 11))
winter3 = mean_data_each_time3.sel(
    time=(mean_data_each_time3['time.month'] == 12) | (mean_data_each_time3['time.month'] <= 2))


data_spring1 = spring1.groupby('time.year').sum(dim='time')
data_summer1 = summer1.groupby('time.year').sum(dim='time')
data_autumn1 = autumn1.groupby('time.year').sum(dim='time')
data_winter1 = winter1.groupby('time.year').sum(dim='time')

data_spring2 = spring2.groupby('time.year').sum(dim='time')
data_summer2 = summer2.groupby('time.year').sum(dim='time')
data_autumn2 = autumn2.groupby('time.year').sum(dim='time')
data_winter2 = winter2.groupby('time.year').sum(dim='time')

data_spring3 = spring3.groupby('time.year').sum(dim='time')
data_summer3 = summer3.groupby('time.year').sum(dim='time')
data_autumn3 = autumn3.groupby('time.year').sum(dim='time')
data_winter3 = winter3.groupby('time.year').sum(dim='time')


# 定义一个函数用于绘制趋势线
def plot_trend(data_var, ax):
    time = np.arange(len(data_var))  # 创建时间序列
    model = LinearRegression()
    model.fit(time.reshape(-1, 1), data_var)
    coefficients = np.polyfit(time, data_var, 1)  # 计算线性回归系数
    trend = np.polyval(coefficients, time)  # 计算线性趋势线
    ax.plot(time + 1940, trend, label='线性趋势', color='red')  # 绘制线性趋势线
    # 显示斜率和R2值
    slope = model.coef_[0]
    r2 = r2_score(data_var, trend)
    ax.text(0.1, 0.9, f'Slope: {slope:.2f} R2: {r2:.2f}',
            horizontalalignment='left', transform=ax.transAxes, fontsize=10)


# 设置全局的字体和字号
title_font = {'family': 'Microsoft YaHei', 'size': 10}
# 设置全局的字体和字号
plt.rcParams.update({'font.family': 'SimHei', 'font.size': 10})
# 创建一个 2x2 的子图布局
fig, axs = plt.subplots(5, 3, figsize=(15, 15))

# 1总 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data1.plot(ax=axs[0, 0], label='年平均降水量')
plot_trend(data1, axs[0, 0])
# 添加坐标轴和图例
axs[0, 0].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[0, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[0, 0].set_ylabel('年平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[0, 0].set_title('高原气候区', loc='left',  fontdict=title_font)

# 1春 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_spring1.plot(ax=axs[1, 0], label='春季平均降水量')
plot_trend(data_spring1, axs[1, 0])
# 添加坐标轴和图例
axs[1, 0].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[1, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[1, 0].set_ylabel('春季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[1, 0].set_title('高原气候区——春', loc='left',  fontdict=title_font)

# 1夏 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_summer1.plot(ax=axs[2, 0], label='夏季平均降水量')
plot_trend(data_summer1, axs[2, 0])
# 添加坐标轴和图例
axs[2, 0].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[2, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[2, 0].set_ylabel('夏季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[2, 0].set_title('高原气候区——夏', loc='left',  fontdict=title_font)

# 1秋 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_autumn1.plot(ax=axs[3, 0], label='秋季平均降水量')
plot_trend(data_autumn1, axs[3, 0])
# 添加坐标轴和图例
axs[3, 0].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[3, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[3, 0].set_ylabel('秋季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[3, 0].set_title('高原气候区——秋', loc='left',  fontdict=title_font)

# 1冬 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_winter1.plot(ax=axs[4, 0], label='冬季平均降水量')
plot_trend(data_winter1, axs[4, 0])
# 添加坐标轴和图例
axs[4, 0].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[4, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[4, 0].set_ylabel('冬季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[4, 0].set_title('高原气候区——冬', loc='left',  fontdict=title_font)

# 2总 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data2.plot(ax=axs[0, 1], label='年平均降水量')
plot_trend(data2, axs[0, 1])
# 添加坐标轴和图例
axs[0, 1].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[0, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[0, 1].set_ylabel('年平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[0, 1].set_title('温带气候区', loc='left',  fontdict=title_font)

# 2春 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_spring2.plot(ax=axs[1, 1], label='春季平均降水量')
plot_trend(data_spring2, axs[1, 1])
# 添加坐标轴和图例
axs[1, 1].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[1, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[1, 1].set_ylabel('春季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[1, 1].set_title('温带气候区——春', loc='left',  fontdict=title_font)

# 2夏 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_summer2.plot(ax=axs[2, 1], label='夏季平均降水量')
plot_trend(data_summer2, axs[2, 1])
# 添加坐标轴和图例
axs[2, 1].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[2, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[2, 1].set_ylabel('夏季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[2, 1].set_title('温带气候区——夏', loc='left',  fontdict=title_font)

# 2秋 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_autumn2.plot(ax=axs[3, 1], label='秋季平均降水量')
plot_trend(data_autumn2, axs[3, 1])
# 添加坐标轴和图例
axs[3, 1].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[3, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[3, 1].set_ylabel('秋季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[3, 1].set_title('温带气候区——秋', loc='left',  fontdict=title_font)

# 2冬 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_winter2.plot(ax=axs[4, 1], label='冬季平均降水量')
plot_trend(data_winter2, axs[4, 1])
# 添加坐标轴和图例
axs[4, 1].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[4, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[4, 1].set_ylabel('冬季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[4, 1].set_title('温带气候区——冬', loc='left',  fontdict=title_font)

# 3总 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data3.plot(ax=axs[0, 2], label='年平均降水量')
plot_trend(data3, axs[0, 2])
# 添加坐标轴和图例
axs[0, 2].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[0, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[0, 2].set_ylabel('年平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[0, 2].set_title('亚热带、热带气候区', loc='left',  fontdict=title_font)

# 3春 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_spring3.plot(ax=axs[1, 2], label='春季平均降水量')
plot_trend(data_spring3, axs[1, 2])
# 添加坐标轴和图例
axs[1, 2].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[1, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[1, 2].set_ylabel('春季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[1, 2].set_title('亚热带、热带气候区——春', loc='left',  fontdict=title_font)

# 3夏 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_summer3.plot(ax=axs[2, 2], label='夏季平均降水量')
plot_trend(data_summer3, axs[2, 2])
# 添加坐标轴和图例
axs[2, 2].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[2, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[2, 2].set_ylabel('夏季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[2, 2].set_title('亚热带、热带气候区——夏', loc='left',  fontdict=title_font)

# 3秋 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_autumn3.plot(ax=axs[3, 2], label='秋季平均降水量')
plot_trend(data_autumn3, axs[3, 2])
# 添加坐标轴和图例
axs[3, 2].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[3, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[3, 2].set_ylabel('秋季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[3, 2].set_title('亚热带、热带气候区——秋', loc='left',  fontdict=title_font)

# 3冬 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_winter3.plot(ax=axs[4, 2], label='冬季平均降水量')
plot_trend(data_winter3, axs[4, 2])
# 添加坐标轴和图例
axs[4, 2].legend(loc='upper right', prop={'size': 10})  # 添加图例
axs[4, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[4, 2].set_ylabel('冬季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[4, 2].set_title('亚热带、热带气候区——冬', loc='left',  fontdict=title_font)

# 调整布局，使得子图之间不重叠
plt.tight_layout()
plt.savefig(outputfilePath)
plt.show()
