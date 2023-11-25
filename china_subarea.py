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
ncFilePath1 = "ERA5_1940-2022_Total_precipitation_China_PCR.nc"
ncFilePath2 = "ERA5_1940-2022_Total_precipitation_China_TT.nc"
ncFilePath3 = "ERA5_1940-2022_Total_precipitation_China_SC.nc"

chinaMapFilePath = "china_subarea.geojson"
chinaNineLineFilePath = "九段线GS（2019）1719号.geojson"
outputfilePath = 'china subarea.png'

# 加载nc数据、中国整体范围数据和南海九段线数据
ncDataset1 = xr.open_dataset(ncFilePath1)
ncDataset2 = xr.open_dataset(ncFilePath2)
ncDataset3 = xr.open_dataset(ncFilePath3)

chinaMap = gpd.read_file(chinaMapFilePath)
chinaNineLine = gpd.read_file(chinaNineLineFilePath)

# 读取nc数据经纬度信息
lat1 = ncDataset1.latitude
lon1 = ncDataset1.longitude
lat2 = ncDataset2.latitude
lon2 = ncDataset2.longitude
lat3 = ncDataset3.latitude
lon3 = ncDataset3.longitude

data1 = ncDataset1["tp"].mean(dim='time')
data2 = ncDataset2["tp"].mean(dim='time')
data3 = ncDataset3["tp"].mean(dim='time')

# 创建一个绘图窗口
figure = plt.figure(figsize=(4, 3), dpi=240, facecolor="w")

# 设置不同降水水平的颜色和数值范围
level1 = [0, 100, 200, 300]
level2 = [-1, 0, 200, 300]
level3 = [-2, -1, 0, 200]

colors = ['#8772C1', '#66C97D', '#FDD187']
# 为不同水平创建颜色映射
cmap = mcolors.ListedColormap(colors)
norm1 = mcolors.BoundaryNorm(level1, cmap.N)
norm2 = mcolors.BoundaryNorm(level2, cmap.N)
norm3 = mcolors.BoundaryNorm(level3, cmap.N)

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
contourf_ChinaMap = ax_ChinaMap.contourf(lon1, lat1, data1, levels=level1, cmap=cmap, norm=norm1)
ax_ChinaMap.contourf(lon1, lat1, data1, levels=level1, cmap=cmap, norm=norm1)
ax_ChinaMap.contourf(lon2, lat2, data2, levels=level2, cmap=cmap, norm=norm2)
ax_ChinaMap.contourf(lon2, lat2, data2, levels=level2, cmap=cmap, norm=norm2)
ax_ChinaMap.contourf(lon3, lat3, data3, levels=level3, cmap=cmap, norm=norm3)
ax_ChinaMap.contourf(lon3, lat3, data3, levels=level3, cmap=cmap, norm=norm3)

# 添加九段线子图
ax_chinaNineLine = figure.add_axes([0.79, 0.234, 0.15, 0.3], projection=ccrs.PlateCarree())
ax_chinaNineLine.set_extent([104.5, 125, 0, 26])
ax_chinaNineLine.spines['geo'].set_linewidth(.5)
ax_chinaNineLine.add_geometries(chinaMap["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.3)
ax_chinaNineLine.add_geometries(chinaNineLine["geometry"], crs=ccrs.PlateCarree(), fc="None", ec="black", linewidth=.5)
ax_chinaNineLine.contourf(lon1, lat1, data1, levels=level1, cmap=cmap, norm=norm1)
ax_chinaNineLine.contourf(lon1, lat1, data1, levels=level1, cmap=cmap, norm=norm1)
ax_chinaNineLine.contourf(lon2, lat2, data2, levels=level2, cmap=cmap, norm=norm2)
ax_chinaNineLine.contourf(lon2, lat2, data2, levels=level2, cmap=cmap, norm=norm2)
ax_chinaNineLine.contourf(lon3, lat3, data3, levels=level3, cmap=cmap, norm=norm3)
ax_chinaNineLine.contourf(lon3, lat3, data3, levels=level3, cmap=cmap, norm=norm3)

# 设置颜色条和标签
cbar = plt.colorbar(contourf_ChinaMap, ax=ax_ChinaMap, ticks=[50, 150, 250], orientation="horizontal",
                    shrink=0.85, aspect=34, pad=.08)
cbar.ax.set_xticklabels(['高原气候区', '温带气候区', '亚热带、热带气候区'],
                        fontproperties='Microsoft YaHei')
cbar.ax.tick_params(which="both", labelsize=8, direction="in", bottom=False)

# 调整布局并展示图像
plt.tight_layout()
plt.savefig(outputfilePath)
plt.show()
