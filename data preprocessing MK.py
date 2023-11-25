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
from scipy.stats import norm

# 设置文件路径
ncFilePath = "ERA5_1940-2022_Total_precipitation.nc"

# 加载nc数据、中国整体范围数据和南海九段线数据
ncDataset = xr.open_dataset(ncFilePath)

selected_data = ncDataset.sel(latitude=slice(54, 18), longitude=slice(73, 135))

# 数据处理 ------------------------------------------------------------------------------------
# 转换时间数据为 Pandas 的时间戳格式
time_stamps_pd = pd.to_datetime(selected_data["time"])
# 计算每个月的天数
days_in_each_month = time_stamps_pd.to_series().dt.days_in_month
# 将每个时间步的降水数据乘以对应月份的天数并转换为毫米
for i in range(len(days_in_each_month.values)):
    selected_data["tp"][i, :, :] = selected_data["tp"][i, :, :] * days_in_each_month.values[i] * 1000
# 计算每年总降水量
annual_mean_tp = selected_data["tp"].groupby('time.year').sum(dim='time')


def mk_trend_analysis(data):
    """
    进行MK趋势分析，返回Kendall's Tau指数

    参数：
    data (xarray.DataArray)：输入的NC数据数组

    返回值：
    kendall_tau (xarray.DataArray)：Kendall's Tau指数，与输入数据的维度相同
    """

    latitude_vals = data['latitude'].values
    longitude_vals = data['longitude'].values
    year_vals = data['year'].values

    latitudes, longitudes, years = len(latitude_vals), len(longitude_vals), len(year_vals)

    z_values = np.zeros((latitudes, longitudes))

    for lat_idx in range(latitudes):
        for lon_idx in range(longitudes):
            time_series = data.sel(latitude=latitude_vals[lat_idx], longitude=longitude_vals[lon_idx])
            time_series = time_series.values

            valid_indices = ~np.isnan(time_series)
            if np.any(valid_indices):
                valid_data = time_series[valid_indices]
                n = len(valid_data)

                # 计算S值
                s = 0
                for i in range(n - 1):
                    for j in range(i + 1, n):
                        s += np.sign(valid_data[j] - valid_data[i])

                # 计算方差
                var_s = (n * (n - 1) * (2 * n + 5)) / 18

                # 计算Z值
                if s > 0:
                    z = (s - 1) / np.sqrt(var_s)
                elif s < 0:
                    z = (s + 1) / np.sqrt(var_s)
                else:
                    z = 0

                z_values[lat_idx, lon_idx] = z
                print(z_values[lat_idx, lon_idx])

    # 创建新的xarray.DataArray对象以保存Z值
    z_values_da = xr.DataArray(z_values, coords={'latitude': latitude_vals, 'longitude': longitude_vals},
                               dims=('latitude', 'longitude'))

    return z_values_da


# 进行MK趋势分析
z_values_result = mk_trend_analysis(annual_mean_tp)

print(z_values_result)


shpFilePath = "national_boundary.shp"
outputfilePath = 'ERA5_1940-2022_Total_precipitation_China_Zvalues.nc'

# 加载shp文件
shp = gpd.read_file(shpFilePath)

# 根据您的nc数据集的字段，选择降水变量 'z' 并重新排列维度
ncDataset = z_values_result.transpose('latitude', 'longitude')
# 指定经度（longitude）和纬度（latitude）维度指定为空间维度，以便在之后的空间操作中能够明确识别这些维度
ncDataset.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# 添加空间参考坐标系（CRS），这里使用WGS 84坐标系 (EPSG:4326)
ncDataset.rio.write_crs("EPSG:4326", inplace=True)

result = ncDataset.rio.clip(shp.geometry.apply(mapping), shp.crs)

result.to_netcdf(outputfilePath)


