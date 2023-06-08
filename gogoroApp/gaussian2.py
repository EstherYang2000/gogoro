import numpy as NP
import rasterio as RAST
from tqdm import tqdm as TQDM
from rasterio.transform import from_origin
from utility import create_batchrange
import pandas as pd
from shapely.geometry import Point, LineString, Polygon
import geopandas as gpd
import shapely
import pyproj
import matplotlib.pyplot as plt


class Gaussian:
    """2D gaussian pars[6] = [x0, y0, amp, radius_x, radius_y, theta]

                                        2               2             
                         -[ a*(x - [0])^  + b*(y - [1])^  + c*(x - [0])*(y - [1]) ]
        y(x,y) = [2] * e^

        where a, b and c are

            cos([5])^2   sin([5])^2
        a = ---------- + ----------
             2*[3]^2      2*[4]^2

            sin([5])^2   cos([5])^2
        b = ---------- + ----------
             2*[3]^2      2*[4]^2

            - sin(2*[5])   sin(2*[5])
        c = ------------ + ----------
               4*[3]^2      4*[4]^2
        ref : https://en.wikipedia.org/wiki/Gaussian_function#Two-dimensional_Gaussian_function
    """

    def __init__(self, scooter_id='default', radius_x=2.5, radius_y=2.5, theta=0, debug=False):
        self.debug = debug
        self._size = 0
        self.update_params(radius_x, radius_y, theta)
        self.scooter_id = scooter_id

    def update_params(self, radius_x, radius_y, theta, scale=3):
        if len(NP.atleast_1d(radius_x)) != len(NP.atleast_1d(radius_y)):
            print('>> [ERROR] update_params: input different size')
            return
        else:
            self._radius_x = NP.atleast_1d(radius_x)/scale
            self._radius_y = NP.atleast_1d(radius_y)/scale
            self._theta = NP.atleast_1d(theta)
            self._size_params = len(self._theta)
# [lat,long,distance]

    def fit(self, X, y):  # X is lat and long ; y pass 1
        self._x0 = NP.atleast_2d(X)[:, 0]
        self._y0 = NP.atleast_2d(X)[:, 1]
        self._z0 = NP.atleast_1d(y)
        if len(NP.atleast_2d(X)[:, 0]) != len(NP.atleast_1d(y)):
            print('>> [ERROR] fit: input different size')
            return
        elif self._size_params > 1 and self._size_params != len(NP.atleast_1d(y)):
            print('>> [ERROR] fit: input different size with parameters')
            return
        else:
            self._x0 = NP.atleast_2d(X)[:, 0]
            self._y0 = NP.atleast_2d(X)[:, 1]
            self._z0 = NP.atleast_1d(y)
            self._size = len(self._z0)

        self._a = NP.cos(self._theta)**2/(2*self._radius_x**2) + \
            NP.sin(self._theta)**2/(2*self._radius_y**2)
        self._b = NP.sin(self._theta)**2/(2*self._radius_x**2) + \
            NP.cos(self._theta)**2/(2*self._radius_y**2)
        self._c = -NP.sin(2*self._theta)/(4*self._radius_x**2) + \
            NP.sin(2*self._theta)/(4*self._radius_y**2)

    def predict(self, X):
        x = NP.atleast_2d(X)[:, 0][:, NP.newaxis]
        y = NP.atleast_2d(X)[:, 1][:, NP.newaxis]
        size = len(x)
        x0 = NP.ones((size, self._size))*self._x0
        y0 = NP.ones((size, self._size))*self._y0
        z0 = NP.ones((size, self._size))*self._z0
        a = NP.ones((size, self._size))*self._a
        b = NP.ones((size, self._size))*self._b
        c = NP.ones((size, self._size))*self._c

        z = z0 * NP.exp(-1 * (a*(x-x0)**2 + b*(y-y0)**2 + 2*c*(x-x0)*(y-y0)))
        # p = z / len(z)
        return z.sum(axis=1)/len(z[0])

    def to_2D(self, xmin, xmax, ymin, ymax, cellsize=100, n_cell_per_job=1000, tqdm=False, crs='EPSG:3826', to_raster=None):
        """
        [DESCRIPTION]
            Output raster data of prediction w.r.t. given extent size
        [INPUT]
            xmin      : float,  minimum value of x-axis of extent (None) 
            xmax      : float,  maximum value of x-axis of extent (None) 
            ymin      : float,  minimum value of y-axis of extent (None)
            ymax      : float,  maximum value of y-axis of extent (None)
            cellsize  : flaot,  cell size of extent (50)
            n_cell_per_job : int, number of cell in processing batch jobs (1000)
            tqdm      : bool,   if show the tqdm processing bar (False)
            crs       : string, the projection code for output raster tif ('epsg:3826')
            to_raster : string, path to save raster tif. If None, the function return the values only (None) 
        [OUTPUT]
            predicted value,
            grid of extent
        """
        if self._size == 0:
            print(">> [ERROR] do fit() before calling to_2D()")
            return

        # Create 2D extent
        # Grid is from top-left to bottom-right
        X, Y = NP.meshgrid(NP.arange(xmin, xmax+cellsize, cellsize),
                           NP.arange(ymin, ymax+cellsize, cellsize))

        xy = NP.concatenate((X.flatten()[:, NP.newaxis] + cellsize/2,
                             Y.flatten()[:, NP.newaxis] - cellsize/2),
                            axis=1)
        print(xy)
        print(len(xy))
        print(int(len(xy)/n_cell_per_job))
        # Define variables
        xbins = X.shape[1]
        ybins = Y.shape[0]
        z = NP.array([])

        # Create batches

        batches = create_batchrange(len(xy), int(len(xy)/n_cell_per_job))
        if self.debug:
            print(">> [INFO] Interpolating %d pixels (%d, %d) to %d batches:" % (
                len(xy), xbins, ybins, len(batches)))
        if tqdm:
            batches = TQDM(batches)

        print("----------------Predict values-------------------")
        # Predict values
        for idx in batches:
            print(idx)
            if tqdm:
                batches.set_description(">> ")
            z_new = self.predict(xy[idx[0]:idx[1], :])

            z = NP.append(z, z_new)

        # create heatmap dataframe
        new_xy = self.change3826to4326(xy)
        result = pd.DataFrame(new_xy, columns=['Latitude', 'Longitude'])
        result['Probability'] = z
        result.to_feather(f'{self.scooter_id}.feather', index=False)

        # Set raster left-top's coordinates
        if to_raster is not None:
            # Set raster left-top's coordinates
            transform = from_origin(X.min(), Y.max(), cellsize, cellsize)
            print(to_raster)
            # Writing raster
            raster = RAST.open(to_raster,
                               'w',
                               driver='GTiff',
                               height=ybins,
                               width=xbins,
                               count=1,
                               dtype=z.dtype,
                               crs=crs,
                               transform=transform)
            raster.write(NP.flip(z.reshape(ybins, xbins), 0), 1)
            raster.close()
        return z, xy

    def create_longlat(self, long_list, lat_list):
        if not isinstance(long_list, (list, tuple)):
            long_list = [long_list]
        if not isinstance(lat_list, (list, tuple)):
            lat_list = [lat_list]
        return [(x, y) for x, y in zip(long_list, lat_list)]

    def change3826to4326(self, coordinates_list):

        # Define the original CRS (EPSG:3826)
        original_crs = pyproj.CRS.from_epsg(3826)

        # Define the target CRS (EPSG:4326)
        target_crs = pyproj.CRS.from_epsg(4326)

        # Create a transformer to convert coordinates
        transformer = pyproj.Transformer.from_crs(
            original_crs, target_crs, always_xy=True)

        # Transform the coordinates to EPSG:4326
        transformed_coordinates = []
        lon_list = []
        lat_list = []
        for x, y in coordinates_list:
            lon, lat = transformer.transform(x, y)
            transformed_coordinates.append((lon, lat))
            lon_list.append(lon)
            lat_list.append(lat)
        latlon4326_list = self.create_longlat(lat_list, lon_list)
        return latlon4326_list

    def swapinout(self, longlat_list, distance_list, output_geojsonfile, output_shpfile):

        # Create a list of Shapely Point objects from the coordinates
        points = [Point(lon, lat) for lon, lat in longlat_list]

        # Create a GeoDataFrame with the Point geometries
        gdf = gpd.GeoDataFrame(geometry=points)

        # Set the coordinate reference system (CRS) of the GeoDataFrame
        gdf.crs = 'EPSG:3826'  # WGS84 CRS

        # Create the buffers around the points
        buffered_series = gdf.buffer(distance_list)

        # Plot the buffers
        # buffered_series.plot()
        # print(buffered_series)
        # Show the plot
        plt.show()

        # Plot the buffer
        # buffered_series.to_file(output_geojsonfile, driver='GeoJSON')
        buffered_series.to_file(output_shpfile)

        return buffered_series

    def setboundingbox(self, latitude, longitude):
        xmin = float('inf')
        xmax = float('-inf')
        ymin = float('inf')
        ymax = float('-inf')
        latlong_list = self.create_longlat(latitude, longitude)

        center = (NP.mean(longitude), NP.mean(latitude))
        distances = [NP.sqrt((lon - center[0]) ** 2 + (lat - center[1]) ** 2)
                     for lon, lat in zip(longitude, latitude)]
        max_distance = max(distances)

        # 建立 grid 的上下左右
        xmin = center[0] - max_distance
        xmax = center[0] + max_distance
        ymin = center[1] - max_distance
        ymax = center[1] + max_distance

        print("xmin:", xmin)
        print("xmax:", xmax)
        print("ymin:", ymin)
        print("ymax:", ymax)
        return xmin, xmax, ymin, ymax

    def create_boundingBox(self, longitude, latitude):
        longlatlist = self.create_longlat(longitude, latitude)

        ps = [Point(points) for points in longlatlist]
        geo_s = gpd.GeoSeries(ps)

        print(geo_s.total_bounds)
        return geo_s.total_bounds

    def filterDataframe(self, scooter_id): #改成sandy的檔案
        data = pd.read_csv('./XYdis_10users_30times_max.csv')
        userdata = data[data['scooter_id'] == scooter_id]

        return userdata

    def execute(self):

        userdata = self.filterDataframe(self.scooter_id)
        # parameter setting
        # longitude_swapin
        longitude_swapin = userdata['swapIn_X'].values.tolist()
        # latitude_swapin
        latitude_swapin = userdata['swapIn_Y'].values.tolist()
        # longitude_swapout
        longitude_swapout = userdata['swapout_X'].values.tolist()
        # latitude_swapin
        latitude_swapout = userdata['swapout_Y'].values.tolist()
        trip_1_distance_list = userdata['trip_1_distance'].values.tolist()
        trip_1_distance_list = [x * 1000 for x in trip_1_distance_list]
        trip_last_distance_list = userdata['trip_last_distance'].values.tolist(
        )
        trip_last_distance_list = [x * 1000 for x in trip_last_distance_list]
        # latlongswapin_list = self.create_longlat(latitude_swapin, longitude_swapin)
        longlatswapin_list = self.create_longlat(
            longitude_swapin, latitude_swapin)
        longlatswapout_list = self.create_longlat(
            longitude_swapout, latitude_swapout)

        # setboundingbox_swapin
        xmin, ymin, xmax, ymax = self.create_boundingBox(
            latitude_swapin, longitude_swapin)
        # xmin,xmax,ymin,ymax = self.setboundingbox(latitude_swapin,longitude_swapin)
        theta_list = NP.zeros((30,), dtype=int)
        y = NP.ones((30,), dtype=int)
        scale = 3
        filepath_swapingeojson = f'/Users/yangyujie/Documents/NTU_DAC/Gogoro專案/Gaussian/output/{self.scooter_id}_swapin.geojson'
        print(filepath_swapingeojson)
        filepath_swapinshp = f'/Users/yangyujie/Documents/NTU_DAC/Gogoro專案/Gaussian/output/{self.scooter_id}_swapin.shp'
        filepath_swapoutgeojson = f'/Users/yangyujie/Documents/NTU_DAC/Gogoro專案/Gaussian/output/{self.scooter_id}_swapout.geojson'
        print(filepath_swapoutgeojson)
        filepath_swapoutshp = f'/Users/yangyujie/Documents/NTU_DAC/Gogoro專案/Gaussian/output/{self.scooter_id}_swapout.shp'

        filepath_raster = f'/Users/yangyujie/Documents/NTU_DAC/Gogoro專案/Gaussian/output/{self.scooter_id}_raster.tif'
        print(filepath_raster)
        self.swapinout(longlatswapin_list, trip_1_distance_list,
                       filepath_swapingeojson, filepath_swapinshp)  # swapin buffer
        self.swapinout(longlatswapout_list, trip_last_distance_list,
                       filepath_swapoutgeojson, filepath_swapoutshp)  # swapout buffer

        # swap in raster
        self.update_params(trip_1_distance_list,
                           trip_1_distance_list, theta_list, scale)
        self.fit(longlatswapin_list, y)
        zRaster, yRaster = self.to_2D(xmin, xmax, ymin, ymax, cellsize=100,
                                      n_cell_per_job=1000, tqdm=False, crs='EPSG:3826', to_raster=filepath_raster)
        print("-------------------swap in raster-----------------------")
        print(zRaster)
        # for i in zRaster:
        #     print(i,end=',')
        print(max(zRaster))
        print(yRaster)


def main():
    
    test = Gaussian(scooter_id='03a6729f-8045-4ada-9110-3ec91079c83c')
    test.execute()
    # set parameter
    # scooter_id = ''
    # data = pd.read_csv('./sample_data2.csv')
    # longitude = data['longitude'].values.tolist()
    # latitude = data['latitude'].values.tolist()
    # latlong_list = test.create_longlat(latitude, longitude)
    # longlat_list = test.create_longlat(longitude, latitude)
    # xmin,xmax,ymin,ymax = test.setboundinbox(latitude,longitude)
    # distance_list = data['cumulative_distance'].values.tolist()
    # theta_list =  NP.zeros((30,), dtype=int)
    # y = NP.ones((30,), dtype=int)
    # scale = 3

    # filepath = f'/data/workspace_g6/test/{scooter_id}_swapin.geojson'


    # test.swapin(longlat_list,distance_list, filepath)
    # test.update_params(distance_list, distance_list,theta_list,scale)
    # test.fit(longlat_list, y)
    # zRaster,yRaster = test.to_2D(xmin, xmax, ymin, ymax, cellsize=50, n_cell_per_job=1000, tqdm=False, crs='EPSG:3826', to_raster='/data/workspace_g6/test/test.tif')
    # print("-------------------raster-----------------------")
    # print(zRaster)
    # print(max(zRaster))
    # print(yRaster)
if __name__ == "__main__":
    main()


# comment
   # print(longlat_list)
    # x_max = max(longlat_list)[0]
    # print(x_max)
    # y_max = max(longlat_list)[0]
    # print(y_max)
    # xmin = min(sublist[0] for sublist in longlat_list)- x_max
    # print(xmin)
    # xmax = max(sublist[0] for sublist in longlat_list)+x_max
    # print(xmax)
    # ymin = min(sublist[1] for sublist in longlat_list)-y_max
    # print(ymin)
    # ymax = max(sublist[1] for sublist in longlat_list)+y_max
    # print(ymax)
    # Initialize bounding box variables
