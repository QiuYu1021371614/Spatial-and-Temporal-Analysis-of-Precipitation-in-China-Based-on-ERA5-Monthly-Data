# 导入必要的库
import xarray as xr  # 用于处理NetCDF文件
import geopandas as gpd  # 用于处理地理空间数据
from shapely.geometry import mapping  # 用于将几何对象转换为GeoJSON格式

# 设置文件路径
ncFilePath = "ERA5_1940-2022_Total_precipitation.nc"
shpFilePath1 = "china_subarea_高原气候区.shp"
shpFilePath2 = "china_subarea_温带气候区.shp"
shpFilePath3 = "china_subarea_亚热热带气候区.shp"
outputfilePath1 = 'ERA5_1940-2022_Total_precipitation_China_PCR.nc'
outputfilePath2 = 'ERA5_1940-2022_Total_precipitation_China_TT.nc'
outputfilePath3 = 'ERA5_1940-2022_Total_precipitation_China_SC.nc'

# 加载nc数据
ncDataset = xr.open_dataset(ncFilePath)
# 根据您的nc数据集的字段，选择降水变量 'tp' 并重新排列维度
ncDataset = ncDataset[['tp']].transpose('time', 'latitude', 'longitude')
# 指定经度（longitude）和纬度（latitude）维度指定为空间维度，以便在之后的空间操作中能够明确识别这些维度
ncDataset.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude", inplace=True)
# 添加空间参考坐标系（CRS），这里使用WGS 84坐标系 (EPSG:4326)
ncDataset.rio.write_crs("EPSG:4326", inplace=True)

# 加载shp文件
shp1 = gpd.read_file(shpFilePath1)
shp2 = gpd.read_file(shpFilePath2)
shp3 = gpd.read_file(shpFilePath3)
# 利用地理空间数据进行裁剪操作，保留在shapefile范围内的数据
ncDataset1 = ncDataset.rio.clip(shp1.geometry.apply(mapping), shp1.crs)
ncDataset2 = ncDataset.rio.clip(shp2.geometry.apply(mapping), shp2.crs)
ncDataset3 = ncDataset.rio.clip(shp3.geometry.apply(mapping), shp3.crs)
print(ncDataset1, ncDataset2, ncDataset3)

# 将处理后的文件保存输出
ncDataset1.to_netcdf(outputfilePath1)
ncDataset2.to_netcdf(outputfilePath2)
ncDataset3.to_netcdf(outputfilePath3)



