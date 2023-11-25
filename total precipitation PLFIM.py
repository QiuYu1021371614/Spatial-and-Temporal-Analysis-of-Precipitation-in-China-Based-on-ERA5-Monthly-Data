import xarray as xr
import numpy as np
import pwlf
import matplotlib.pyplot as plt
import pandas as pd  # 用于数据处理和分析
from scipy.stats import linregress

plt.rcParams['axes.unicode_minus'] = False

# 设置文件路径
ncFilePath = "ERA5_1940-2022_Total_precipitation_China.nc"
outputfilePath = 'total precipitation PLFIM.png'

# 加载nc数据
ncDataset = xr.open_dataset(ncFilePath)

# 数据处理 ------------------------------------------------------------------------------------
# 转换时间数据为 Pandas 的时间戳格式
time_stamps_pd = pd.to_datetime(ncDataset["time"])
# 计算每个月的天数
days_in_each_month = time_stamps_pd.to_series().dt.days_in_month
# 将每个时间步的降水数据乘以对应月份的天数并转换为毫米
for i in range(len(days_in_each_month.values)):
    ncDataset["tp"][i, :, :] = ncDataset["tp"][i, :, :] * days_in_each_month.values[i] * 1000

mean_ncDataset = ncDataset["tp"].mean(dim=('latitude', 'longitude'))
data = mean_ncDataset.groupby('time.year').sum(dim='time')
data_values = data.values
years = data.year.values

# 计算降水量值的累积和
data_cumulative = np.cumsum(data_values - np.mean(data_values))

# 使用你的x和y数据初始化分段线性拟合
my_pwlf = pwlf.PiecewiseLinFit(years, data_cumulative)

# 使用期望的线段数拟合数据
# 这里，我们可以尝试用两个线段（一个变化点）来拟合
res = my_pwlf.fit(3)

# 使用分段线性拟合的结果（变化点）在原始年平均数据上进行拟合
pwlf_segments = pwlf.PiecewiseLinFit(years, data_values)
pwlf_segments.fit_with_breaks(my_pwlf.fit_breaks)

# 预测拟合线
x_hat = np.linspace(years.min(), years.max(), 100)
y_hat = pwlf_segments.predict(x_hat)

# 计算整个数据集的线性趋势线
slope, intercept, r_value, p_value, std_err = linregress(years, data_values)
trend_line = intercept + slope * years

# 设置全局的字体和字号
plt.rcParams.update({'font.family': 'SimHei', 'font.size': 10})

plt.figure(figsize=(6, 3.6))
plt.plot(years, data, label='全年降水量')
plt.plot(x_hat, y_hat, '-', color='r', label='分段线性拟合')
plt.plot(years, trend_line, '--', color='orange', label='线性趋势线')  # 添加线性趋势线

# 变化点在 my_pwlf.fit_breaks 中
for point in my_pwlf.fit_breaks[1:-1]:  # 跳过第一个和最后一个点，因为它们是边界
    plt.scatter(point, pwlf_segments.predict(point), facecolors='none', edgecolors='b', s=100,
                label='突变点' if '突变点' not in plt.gca().get_legend_handles_labels()[1] else "")
    plt.annotate(str(int(point)),  # 这将浮点数年份转换为整数然后转换为字符串
                 (point, pwlf_segments.predict(point)),
                 textcoords="offset points", color='b',
                 xytext=(0, 10),
                 ha='center')

plt.xlabel('年份')
plt.ylabel('年平均降水量（mm）')
plt.legend(prop={'size': 8})
plt.savefig(outputfilePath)
plt.show()
