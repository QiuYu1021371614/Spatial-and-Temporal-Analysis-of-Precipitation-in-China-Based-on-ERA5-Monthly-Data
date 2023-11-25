import numpy as np  # 用于处理数组和数值计算
import xarray as xr  # 用于处理NetCDF文件
import matplotlib.pyplot as plt  # 用于绘图
import pandas as pd  # 用于数据处理和分析
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


# 设置文件路径
ncFilePath = "ERA5_1940-2022_Total_precipitation_China.nc"
outputfilePath = 'total precipitation curve'

# 加载nc数据
ncDataset_China = xr.open_dataset(ncFilePath)

# 数据处理 ------------------------------------------------------------------------------------
# 转换时间数据为 Pandas 的时间戳格式
time_stamps_pd = pd.to_datetime(ncDataset_China["time"])
# 计算每个月的天数
days_in_each_month = time_stamps_pd.to_series().dt.days_in_month
# 将每个时间步的降水数据乘以对应月份的天数并转换为毫米
for i in range(len(days_in_each_month.values)):
    ncDataset_China["tp"][i, :, :] = ncDataset_China["tp"][i, :, :] * days_in_each_month.values[i] * 1000

mean_data_each_time = ncDataset_China["tp"].mean(dim=('latitude', 'longitude'))
ncDataset_year = mean_data_each_time.groupby('time.year').sum(dim='time')

# 设置全局的字体和字号
plt.rcParams.update({'font.family': 'SimHei', 'font.size': 10})

# 绘制数据时间序列图
plt.figure(figsize=(6, 3.6))
ncDataset_year.plot(label='全年降水量')  # 绘制数据

# 计算时间序列的线性趋势线
time = np.arange(len(ncDataset_year))  # 创建时间序列
model = LinearRegression()
model.fit(time.reshape(-1, 1), ncDataset_year)
coefficients = np.polyfit(time, ncDataset_year, 1)  # 计算线性回归系数
trend = np.polyval(coefficients, time)  # 计算线性趋势线0
slope = model.coef_[0]
r2 = r2_score(ncDataset_year, trend)


# 绘制线性趋势线
plt.plot(time+1940, trend, label='线性趋势', color='red')
plt.text(0.1, 0.9, f'Slope: {slope:.2f} R2: {r2:.2f}', horizontalalignment='left', transform=plt.gca().transAxes, fontsize=10)
# 添加坐标轴和图例
plt.xlabel('年份', fontsize=10)
plt.ylabel('年平均降水量（mm）', fontsize=10)
plt.legend(prop={'size': 10})
plt.savefig(outputfilePath)
plt.show()
