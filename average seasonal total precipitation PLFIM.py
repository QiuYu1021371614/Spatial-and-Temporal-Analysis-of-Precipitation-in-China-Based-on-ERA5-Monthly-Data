import numpy as np  # 用于处理数组和数值计算
import xarray as xr  # 用于处理NetCDF文件
import matplotlib.pyplot as plt  # 用于绘图
import pandas as pd  # 用于数据处理和分析
import pwlf

# 设置文件路径
ncFilePath1 = "ERA5_1940-2022_Total_precipitation_China_PCR.nc"
ncFilePath2 = "ERA5_1940-2022_Total_precipitation_China_TT.nc"
ncFilePath3 = "ERA5_1940-2022_Total_precipitation_China_SC.nc"
outputfilePath = 'average seasonal total precipitation PLFIM.png'

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
data1_values = data1.values
data2 = mean_data_each_time2.groupby('time.year').sum(dim='time')
data2_values = data2.values
data3 = mean_data_each_time3.groupby('time.year').sum(dim='time')
data3_values = data3.values

years = data1.year.values

# 计算降水量值的累积和
data1_cumulative = np.cumsum(data1_values - np.mean(data1_values))
data2_cumulative = np.cumsum(data2_values - np.mean(data2_values))
data3_cumulative = np.cumsum(data3_values - np.mean(data3_values))

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
data_spring1_values = data_spring1.values
data_summer1 = summer1.groupby('time.year').sum(dim='time')
data_summer1_values = data_summer1.values
data_autumn1 = autumn1.groupby('time.year').sum(dim='time')
data_autumn1_values = data_autumn1.values
data_winter1 = winter1.groupby('time.year').sum(dim='time')
data_winter1_values = data_winter1.values

data_spring2 = spring2.groupby('time.year').sum(dim='time')
data_spring2_values = data_spring2.values
data_summer2 = summer2.groupby('time.year').sum(dim='time')
data_summer2_values = data_summer2.values
data_autumn2 = autumn2.groupby('time.year').sum(dim='time')
data_autumn2_values = data_autumn2.values
data_winter2 = winter2.groupby('time.year').sum(dim='time')
data_winter2_values = data_winter2.values

data_spring3 = spring3.groupby('time.year').sum(dim='time')
data_spring3_values = data_spring3.values
data_summer3 = summer3.groupby('time.year').sum(dim='time')
data_summer3_values = data_summer3.values
data_autumn3 = autumn3.groupby('time.year').sum(dim='time')
data_autumn3_values = data_autumn3.values
data_winter3 = winter3.groupby('time.year').sum(dim='time')
data_winter3_values = data_winter3.values

# 计算降水量值的累积和
data_spring1_cumulative = np.cumsum(data_spring1_values - np.mean(data_spring1_values))
data_summer1_cumulative = np.cumsum(data_summer1_values - np.mean(data_summer1_values))
data_autumn1_cumulative = np.cumsum(data_autumn1_values - np.mean(data_autumn1_values))
data_winter1_cumulative = np.cumsum(data_winter1_values - np.mean(data_winter1_values))

data_spring2_cumulative = np.cumsum(data_spring2_values - np.mean(data_spring2_values))
data_summer2_cumulative = np.cumsum(data_summer2_values - np.mean(data_summer2_values))
data_autumn2_cumulative = np.cumsum(data_autumn2_values - np.mean(data_autumn2_values))
data_winter2_cumulative = np.cumsum(data_winter2_values - np.mean(data_winter2_values))

data_spring3_cumulative = np.cumsum(data_spring3_values - np.mean(data_spring3_values))
data_summer3_cumulative = np.cumsum(data_summer3_values - np.mean(data_summer3_values))
data_autumn3_cumulative = np.cumsum(data_autumn3_values - np.mean(data_autumn3_values))
data_winter3_cumulative = np.cumsum(data_winter3_values - np.mean(data_winter3_values))


# 定义一个函数用于绘制趋势线
def plot_trend(data_var, ax):
    time = np.arange(len(data_var))  # 创建时间序列
    coefficients = np.polyfit(time, data_var, 1)  # 计算线性回归系数
    trend = np.polyval(coefficients, time)  # 计算线性趋势线
    ax.plot(years, trend, '--', color='orange', label='线性趋势')  # 绘制线性趋势线


# 定义一个函数用于线性拟合和计算突变点
def PLFIM(data_cumulative, data_values, ax):
    # 使用你的x和y数据初始化分段线性拟合
    my_pwlf = pwlf.PiecewiseLinFit(years, data_cumulative)
    # 使用期望的线段数拟合数据
    # 这里，我们可以尝试用两个线段（一个变化点）来拟合
    my_pwlf.fit(3)
    # 使用分段线性拟合的结果（变化点）在原始年平均数据上进行拟合
    pwlf_segments = pwlf.PiecewiseLinFit(years, data_values)
    pwlf_segments.fit_with_breaks(my_pwlf.fit_breaks)
    # 预测拟合线
    x_hat = np.linspace(years.min(), years.max(), 100)
    y_hat = pwlf_segments.predict(x_hat)
    ax.plot(x_hat, y_hat, '-', color='r', label='分段线性拟合')
    legend_labels = {}
    # 变化点在 my_pwlf.fit_breaks 中
    for point in my_pwlf.fit_breaks[1:-1]:  # 跳过第一个和最后一个点，因为它们是边界
        label = '变化点' if '变化点' not in legend_labels else ""
        ax.scatter(point, pwlf_segments.predict(point), facecolors='none', edgecolors='b', s=100,
                   label=label)
        ax.annotate(str(int(point)),  # 这将浮点数年份转换为整数然后转换为字符串
                    (point, pwlf_segments.predict(point)),
                    textcoords="offset points", color='b',
                    xytext=(0, 10),
                    ha='center')
        legend_labels[label] = None  # 标记该标签已添加


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
PLFIM(data1_cumulative, data1_values, axs[0, 0])
# 添加坐标轴和图例
axs[0, 0].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[0, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[0, 0].set_ylabel('年平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[0, 0].set_title('高原气候区', loc='left', fontdict=title_font)

# 1春 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_spring1.plot(ax=axs[1, 0], label='春季平均降水量')
plot_trend(data_spring1, axs[1, 0])
PLFIM(data_spring1_cumulative, data_spring1_values, axs[1, 0])
# 添加坐标轴和图例
axs[1, 0].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[1, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[1, 0].set_ylabel('春季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[1, 0].set_title('高原气候区——春', loc='left', fontdict=title_font)

# 1夏 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_summer1.plot(ax=axs[2, 0], label='夏季平均降水量')
plot_trend(data_summer1, axs[2, 0])
PLFIM(data_summer1_cumulative, data_summer1_values, axs[2, 0])
# 添加坐标轴和图例
axs[2, 0].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[2, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[2, 0].set_ylabel('夏季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[2, 0].set_title('高原气候区——夏', loc='left', fontdict=title_font)

# 1秋 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_autumn1.plot(ax=axs[3, 0], label='秋季平均降水量')
plot_trend(data_autumn1, axs[3, 0])
PLFIM(data_autumn1_cumulative, data_autumn1_values, axs[3, 0])
# 添加坐标轴和图例
axs[3, 0].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[3, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[3, 0].set_ylabel('秋季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[3, 0].set_title('高原气候区——秋', loc='left', fontdict=title_font)

# 1冬 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_winter1.plot(ax=axs[4, 0], label='冬季平均降水量')
plot_trend(data_winter1, axs[4, 0])
PLFIM(data_winter1_cumulative, data_winter1_values, axs[4, 0])
# 添加坐标轴和图例
axs[4, 0].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[4, 0].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[4, 0].set_ylabel('冬季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[4, 0].set_title('高原气候区——冬', loc='left', fontdict=title_font)

# 2总 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data2.plot(ax=axs[0, 1], label='年平均降水量')
plot_trend(data2, axs[0, 1])
PLFIM(data2_cumulative, data2_values, axs[0, 1])
# 添加坐标轴和图例
axs[0, 1].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[0, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[0, 1].set_ylabel('年平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[0, 1].set_title('温带气候区', loc='left', fontdict=title_font)

# 2春 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_spring2.plot(ax=axs[1, 1], label='春季平均降水量')
plot_trend(data_spring2, axs[1, 1])
PLFIM(data_spring2_cumulative, data_spring2_values, axs[1, 1])
# 添加坐标轴和图例
axs[1, 1].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[1, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[1, 1].set_ylabel('春季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[1, 1].set_title('温带气候区——春', loc='left', fontdict=title_font)

# 2夏 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_summer2.plot(ax=axs[2, 1], label='夏季平均降水量')
plot_trend(data_summer2, axs[2, 1])
PLFIM(data_summer2_cumulative, data_summer2_values, axs[2, 1])
# 添加坐标轴和图例
axs[2, 1].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[2, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[2, 1].set_ylabel('夏季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[2, 1].set_title('温带气候区——夏', loc='left', fontdict=title_font)

# 2秋 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_autumn2.plot(ax=axs[3, 1], label='秋季平均降水量')
plot_trend(data_autumn2, axs[3, 1])
PLFIM(data_autumn2_cumulative, data_autumn2_values, axs[3, 1])
# 添加坐标轴和图例
axs[3, 1].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[3, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[3, 1].set_ylabel('秋季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[3, 1].set_title('温带气候区——秋', loc='left', fontdict=title_font)

# 2冬 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_winter2.plot(ax=axs[4, 1], label='冬季平均降水量')
plot_trend(data_winter2, axs[4, 1])
PLFIM(data_winter2_cumulative, data_winter2_values, axs[4, 1])
# 添加坐标轴和图例
axs[4, 1].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[4, 1].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[4, 1].set_ylabel('冬季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[4, 1].set_title('温带气候区——冬', loc='left', fontdict=title_font)

# 3总 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data3.plot(ax=axs[0, 2], label='年平均降水量')
plot_trend(data3, axs[0, 2])
PLFIM(data3_cumulative, data3_values, axs[0, 2])
# 添加坐标轴和图例
axs[0, 2].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[0, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[0, 2].set_ylabel('年平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[0, 2].set_title('亚热带、热带气候区', loc='left', fontdict=title_font)

# 3春 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_spring3.plot(ax=axs[1, 2], label='春季平均降水量')
plot_trend(data_spring3, axs[1, 2])
PLFIM(data_spring3_cumulative, data_spring3_values, axs[1, 2])
# 添加坐标轴和图例
axs[1, 2].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[1, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[1, 2].set_ylabel('春季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[1, 2].set_title('亚热带、热带气候区——春', loc='left', fontdict=title_font)

# 3夏 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_summer3.plot(ax=axs[2, 2], label='夏季平均降水量')
plot_trend(data_summer3, axs[2, 2])
PLFIM(data_summer3_cumulative, data_summer3_values, axs[2, 2])
# 添加坐标轴和图例
axs[2, 2].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[2, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[2, 2].set_ylabel('夏季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[2, 2].set_title('亚热带、热带气候区——夏', loc='left', fontdict=title_font)

# 3秋 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_autumn3.plot(ax=axs[3, 2], label='秋季平均降水量')
plot_trend(data_autumn3, axs[3, 2])
PLFIM(data_autumn3_cumulative, data_autumn3_values, axs[3, 2])
# 添加坐标轴和图例
axs[3, 2].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[3, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[3, 2].set_ylabel('秋季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[3, 2].set_title('亚热带、热带气候区——秋', loc='left', fontdict=title_font)

# 3冬 ------------------------------------------------------------------------------------
# 显示每个季节的数据
data_winter3.plot(ax=axs[4, 2], label='冬季平均降水量')
plot_trend(data_winter3, axs[4, 2])
PLFIM(data_winter3_cumulative, data_winter3_values, axs[4, 2])
# 添加坐标轴和图例
axs[4, 2].legend(loc='upper right', prop={'size': 8})  # 添加图例
axs[4, 2].set_xlabel('年份', fontsize=10)  # 设置x轴标签
axs[4, 2].set_ylabel('冬季平均降水量（mm）', fontsize=10)  # 设置y轴标签
axs[4, 2].set_title('亚热带、热带气候区——冬', loc='left', fontdict=title_font)

# 调整布局，使得子图之间不重叠
plt.tight_layout()
plt.savefig(outputfilePath)
plt.show()
