# 导入必要的库
import numpy as np  # 用于处理数组和数值计算
import xarray as xr  # 用于处理NetCDF文件
import geopandas as gpd  # 用于处理地理空间数据
from shapely.geometry import mapping  # 用于将几何对象转换为GeoJSON格式
import matplotlib.pyplot as plt  # 用于绘图
import matplotlib.colors as mcolors  # 用于处理颜色映射和色彩等
import matplotlib as mpl  # 用于绘图
import cartopy.crs as ccrs  # 用于地图投影
import cartopy.feature as cfeature  # 用于地图要素
import gma  # 用于绘制地图边界
import pandas as pd  # 用于数据处理和分析


# 设置文件路径
ncFilePath = "ERA5_1940-2022_Total_precipitation_China.nc"
chinaMapFilePath = "china_subarea.geojson"
chinaNineLineFilePath = "九段线GS（2019）1719号.geojson"
outputfilePath = 'annual mean total precipitation.png'

# 加载nc数据、中国整体范围数据和南海九段线数据
ncDataset_China = xr.open_dataset(ncFilePath)
chinaMap = gpd.read_file(chinaMapFilePath)
chinaNineLine = gpd.read_file(chinaNineLineFilePath)
# 读取nc数据经纬度信息
lat = ncDataset_China.latitude
lon = ncDataset_China.longitude

# 数据处理 ------------------------------------------------------------------------------------
# 转换时间数据为 Pandas 的时间戳格式
time_stamps_pd = pd.to_datetime(ncDataset_China["time"])
# 计算每个月的天数
days_in_each_month = time_stamps_pd.to_series().dt.days_in_month
# 将每个时间步的降水数据乘以对应月份的天数并转换为毫米
for i in range(len(days_in_each_month.values)):
    ncDataset_China["tp"][i, :, :] = ncDataset_China["tp"][i, :, :] * days_in_each_month.values[i] * 1000
# 计算年均降水量
annual_mean_tp = ncDataset_China["tp"].mean(dim='time') * 12
data = annual_mean_tp


# 创建一个绘图窗口
figure = plt.figure(figsize=(4, 3), dpi=240, facecolor="w")

# 设置不同降水水平的颜色和数值范围
level1 = np.linspace(0, 2500, 15)
level2 = (2500, 12000)
title_font_s = {'family': 'Microsoft YaHei', 'size': 7}

# 添加中国地图子图
ax_ChinaMap = figure.add_subplot(projection=ccrs.PlateCarree())
# 设置地图范围和细节
ax_ChinaMap.set_global()
ax_ChinaMap.set_extent([72, 136, 16, 56], crs=ccrs.PlateCarree())
ax_ChinaMap.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.2)
ax_ChinaMap.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.3)
gls_ChinaMap = ax_ChinaMap.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                                     color='gray', alpha=0, linestyle='dashed', linewidth=0.3,
                                     y_inline=False, x_inline=False,
                                     xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 10),
                                     xlabel_style={"size": 7},
                                     ylabel_style={"size": 7})
gls_ChinaMap.top_labels = False
gls_ChinaMap.right_labels = False
contourf_ChinaMap = ax_ChinaMap.contourf(lon, lat, data, levels=level1, cmap="Blues", extend='both')
ax_ChinaMap.contourf(lon, lat, data, levels=level2, colors='#063676')
ax_ChinaMap.text(0.235, 0.42, "高原气候区", transform=ax_ChinaMap.transAxes, fontdict=title_font_s)
ax_ChinaMap.text(0.55, 0.55, "温带气候区", transform=ax_ChinaMap.transAxes, fontdict=title_font_s)
ax_ChinaMap.text(0.475, 0.27, "亚热带、热带气候区", transform=ax_ChinaMap.transAxes, fontdict=title_font_s)

# 添加九段线子图
ax_chinaNineLine = figure.add_axes([0.779, 0.25, 0.15, 0.3], projection=ccrs.PlateCarree())
ax_chinaNineLine.set_extent([104.5, 125, 0, 26])
ax_chinaNineLine.spines['geo'].set_linewidth(.5)
ax_chinaNineLine.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.3)
ax_chinaNineLine.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.5)
ax_chinaNineLine.contourf(lon, lat, data, levels=level1, cmap="Blues")
ax_chinaNineLine.contourf(lon, lat, data, levels=level2, colors='#063676')

# 设置颜色条和标签
cbar = plt.colorbar(contourf_ChinaMap, ax=ax_ChinaMap, ticks=[0, 500, 1000, 1500, 2000, 2500], orientation="horizontal",
                    shrink=0.85, aspect=34, pad=.08)
cbar.ax.set_xticklabels(['0', '500', '1000', '1500', '2000', '2500'],
                        fontproperties='Microsoft YaHei')
cbar.ax.tick_params(labelsize=8, direction="in", bottom=False)
cbar.set_label("total precipitation(mm)", fontsize=8)

# 调整布局并展示图像
plt.tight_layout()
plt.savefig(outputfilePath)
plt.show()

