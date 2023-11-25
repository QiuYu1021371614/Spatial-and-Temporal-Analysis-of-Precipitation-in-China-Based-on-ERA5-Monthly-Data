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
outputfilePath = 'average seasonal total precipitation.png'

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
# 按照季节重分类并求平均
ncDataset_season_mean = ncDataset_China.groupby('time.season').mean(dim='time')
# 划分四季数据
data_spring = ncDataset_season_mean["tp"][2, :, :] * 3
data_summer = ncDataset_season_mean["tp"][1, :, :] * 3
data_autumn = ncDataset_season_mean["tp"][3, :, :] * 3
data_winter = ncDataset_season_mean["tp"][0, :, :] * 3


# 创建一个绘图窗口
figure = plt.figure(figsize=(8, 6), dpi=120, facecolor="w")

# 设置不同降水水平的颜色和数值范围
level1 = np.linspace(0, 1200, 9)
level2 = (1200, 12000)
# 设置字体样式
title_font = {'family': 'Microsoft YaHei', 'size': 10}
title_font_s = {'family': 'Microsoft YaHei', 'size': 7}

# 春 ------------------------------------------------------------------------------------
ax_spring_ChinaMap = figure.add_axes([0.06, 0.55, 0.43, 0.43], projection=ccrs.PlateCarree())
ax_spring_ChinaMap.set_global()
ax_spring_ChinaMap.set_extent([72, 136, 16, 56], crs=ccrs.PlateCarree())
ax_spring_ChinaMap.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.2)
ax_spring_ChinaMap.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                  linewidth=.3)
gls_spring_ChinaMap = ax_spring_ChinaMap.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                                                   color='gray', alpha=0, linestyle='dashed', linewidth=0.3,
                                                   y_inline=False, x_inline=False,
                                                   xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 10),
                                                   xlabel_style={"size": 8},
                                                   ylabel_style={"size": 8})
gls_spring_ChinaMap.top_labels = False
gls_spring_ChinaMap.right_labels = False
contourf_spring_ChinaMap = ax_spring_ChinaMap.contourf(lon, lat, data_spring, levels=level1, cmap='Blues', extend='both')
ax_spring_ChinaMap.contourf(lon, lat, data_spring, levels=level2, colors='#063676')
ax_spring_ChinaMap.set_title('春', loc='left', fontdict=title_font)
ax_spring_ChinaMap.text(0.235, 0.42, "高原气候区", transform=ax_spring_ChinaMap.transAxes, fontdict=title_font_s)
ax_spring_ChinaMap.text(0.55, 0.55, "温带气候区", transform=ax_spring_ChinaMap.transAxes, fontdict=title_font_s)
ax_spring_ChinaMap.text(0.475, 0.27, "亚热带、热带气候区", transform=ax_spring_ChinaMap.transAxes, fontdict=title_font_s)

# 添加九段线子图
ax_spring_chinaNineLine = figure.add_axes([0.418, 0.5755, 0.072, 0.144], projection=ccrs.PlateCarree())
ax_spring_chinaNineLine.set_extent([104.5, 125, 0, 26])
ax_spring_chinaNineLine.spines['geo'].set_linewidth(.5)
ax_spring_chinaNineLine.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.3)
ax_spring_chinaNineLine.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.5)
ax_spring_chinaNineLine.contourf(lon, lat, data_spring, levels=level1, cmap='Blues')
ax_spring_chinaNineLine.contourf(lon, lat, data_spring, levels=level2, colors='#063676')

# 夏 ------------------------------------------------------------------------------------
ax_summer_ChinaMap = figure.add_axes([0.545, 0.55, 0.43, 0.43], projection=ccrs.PlateCarree())
ax_summer_ChinaMap.set_global()
ax_summer_ChinaMap.set_extent([72, 136, 16, 56], crs=ccrs.PlateCarree())
ax_summer_ChinaMap.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.2)
ax_summer_ChinaMap.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                  linewidth=.3)
gls_summer_ChinaMap = ax_summer_ChinaMap.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                                                   color='gray', alpha=0, linestyle='dashed', linewidth=0.3,
                                                   y_inline=False, x_inline=False,
                                                   xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 10),
                                                   xlabel_style={"size": 8},
                                                   ylabel_style={"size": 8})
gls_summer_ChinaMap.top_labels = False
gls_summer_ChinaMap.right_labels = False
contourf_summer_ChinaMap = ax_summer_ChinaMap.contourf(lon, lat, data_summer, levels=level1, cmap='Blues')
ax_summer_ChinaMap.contourf(lon, lat, data_summer, levels=level2, colors='#063676')
ax_summer_ChinaMap.set_title('夏', loc='left', fontdict=title_font)
ax_summer_ChinaMap.text(0.235, 0.42, "高原气候区", transform=ax_summer_ChinaMap.transAxes, fontdict=title_font_s)
ax_summer_ChinaMap.text(0.55, 0.55, "温带气候区", transform=ax_summer_ChinaMap.transAxes, fontdict=title_font_s)
ax_summer_ChinaMap.text(0.475, 0.27, "亚热带、热带气候区", transform=ax_summer_ChinaMap.transAxes, fontdict=title_font_s)

# 添加九段线子图
ax_summer_chinaNineLine = figure.add_axes([0.903, 0.5755, 0.072, 0.144], projection=ccrs.PlateCarree())
ax_summer_chinaNineLine.set_extent([104.5, 125, 0, 26])
ax_summer_chinaNineLine.spines['geo'].set_linewidth(.5)
ax_summer_chinaNineLine.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.3)
ax_summer_chinaNineLine.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.5)
ax_summer_chinaNineLine.contourf(lon, lat, data_summer, levels=level1, cmap='Blues')
ax_summer_chinaNineLine.contourf(lon, lat, data_summer, levels=level2, colors='#063676')

# 秋 ------------------------------------------------------------------------------------
ax_autumn_ChinaMap = figure.add_axes([0.06, 0.1, 0.43, 0.43], projection=ccrs.PlateCarree())
ax_autumn_ChinaMap.set_global()
ax_autumn_ChinaMap.set_extent([72, 136, 16, 56], crs=ccrs.PlateCarree())
ax_autumn_ChinaMap.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.2)
ax_autumn_ChinaMap.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                  linewidth=.3)
gls_autumn_ChinaMap = ax_autumn_ChinaMap.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                                                   color='gray', alpha=0, linestyle='dashed', linewidth=0.3,
                                                   y_inline=False, x_inline=False,
                                                   xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 10),
                                                   xlabel_style={"size": 8},
                                                   ylabel_style={"size": 8})
gls_autumn_ChinaMap.top_labels = False
gls_autumn_ChinaMap.right_labels = False
contourf_autumn_ChinaMap = ax_autumn_ChinaMap.contourf(lon, lat, data_autumn, levels=level1, cmap='Blues')
ax_autumn_ChinaMap.contourf(lon, lat, data_autumn, levels=level2, colors='#063676')
ax_autumn_ChinaMap.set_title('秋', loc='left', fontdict=title_font)
ax_autumn_ChinaMap.text(0.235, 0.42, "高原气候区", transform=ax_autumn_ChinaMap.transAxes, fontdict=title_font_s)
ax_autumn_ChinaMap.text(0.55, 0.55, "温带气候区", transform=ax_autumn_ChinaMap.transAxes, fontdict=title_font_s)
ax_autumn_ChinaMap.text(0.475, 0.27, "亚热带、热带气候区", transform=ax_autumn_ChinaMap.transAxes, fontdict=title_font_s)

# 添加九段线子图
ax_autumn_chinaNineLine = figure.add_axes([0.418, 0.1255, 0.072, 0.144], projection=ccrs.PlateCarree())
ax_autumn_chinaNineLine.set_extent([104.5, 125, 0, 26])
ax_autumn_chinaNineLine.spines['geo'].set_linewidth(.5)
ax_autumn_chinaNineLine.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.3)
ax_autumn_chinaNineLine.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.5)
ax_autumn_chinaNineLine.contourf(lon, lat, data_autumn, levels=level1, cmap='Blues')
ax_autumn_chinaNineLine.contourf(lon, lat, data_autumn, levels=level2, colors='#063676')

# 冬 ------------------------------------------------------------------------------------
ax_winter_ChinaMap = figure.add_axes([0.545, 0.1, 0.43, 0.43], projection=ccrs.PlateCarree())
ax_winter_ChinaMap.set_global()
ax_winter_ChinaMap.set_extent([72, 136, 16, 56], crs=ccrs.PlateCarree())
ax_winter_ChinaMap.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.2)
ax_winter_ChinaMap.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                  linewidth=.3)
gls_winter_ChinaMap = ax_winter_ChinaMap.gridlines(draw_labels=True, crs=ccrs.PlateCarree(),
                                                   color='gray', alpha=0, linestyle='dashed', linewidth=0.3,
                                                   y_inline=False, x_inline=False,
                                                   xlocs=range(-180, 180, 10), ylocs=range(-90, 90, 10),
                                                   xlabel_style={"size": 8},
                                                   ylabel_style={"size": 8})
gls_winter_ChinaMap.top_labels = False
gls_winter_ChinaMap.right_labels = False
contourf_winter_ChinaMap = ax_winter_ChinaMap.contourf(lon, lat, data_winter, levels=level1, cmap='Blues')
ax_winter_ChinaMap.contourf(lon, lat, data_winter, levels=level2, colors='#063676')
ax_winter_ChinaMap.set_title('冬', loc='left', fontdict=title_font)
ax_winter_ChinaMap.text(0.235, 0.42, "高原气候区", transform=ax_winter_ChinaMap.transAxes, fontdict=title_font_s)
ax_winter_ChinaMap.text(0.55, 0.55, "温带气候区", transform=ax_winter_ChinaMap.transAxes, fontdict=title_font_s)
ax_winter_ChinaMap.text(0.475, 0.27, "亚热带、热带气候区", transform=ax_winter_ChinaMap.transAxes, fontdict=title_font_s)

# 添加九段线子图
ax_winter_chinaNineLine = figure.add_axes([0.903, 0.1255, 0.072, 0.144], projection=ccrs.PlateCarree())
ax_winter_chinaNineLine.set_extent([104.5, 125, 0, 26])
ax_winter_chinaNineLine.spines['geo'].set_linewidth(.5)
ax_winter_chinaNineLine.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.3)
ax_winter_chinaNineLine.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black",
                                       linewidth=.5)
ax_winter_chinaNineLine.contourf(lon, lat, data_winter, levels=level1, cmap='Blues')
ax_winter_chinaNineLine.contourf(lon, lat, data_winter, levels=level2, colors='#063676')

# 设置颜色条和标签
cbar_ax = figure.add_axes([0.27, 0.075, 0.5, 0.015])
cbar = plt.colorbar(contourf_spring_ChinaMap, cax=cbar_ax, extend='both', ticks=[0, 300, 600, 900, 1200],
                    orientation="horizontal")
cbar.ax.set_xticklabels(['0', '300', '600', '900', '1200'], fontproperties='Microsoft YaHei')
cbar.ax.tick_params(labelsize=10, direction="in", bottom=False)
cbar.set_label("total precipitation(mm)", fontsize=10)

# 调整布局并展示图像
plt.tight_layout()
plt.savefig(outputfilePath)
plt.show()

