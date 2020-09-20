import copy
import random

from attrdict import AttrDict
import numpy as np
import matplotlib.pyplot as mpl
import rasterio.features
import geopandas as gpd


def investigate1():
  filename = 'cb_2018_us_county_500k/cb_2018_us_county_500k.shp'
  data = gpd.read_file(filename)
  print(data)
  print(data.head())
  print(data.crs)
  print(data.columns)

  for col in data.columns:
    print('----------------------------------------------------------------------')
    print(col)
    print(data[col][0:3])

  # Just consider the continguous 48 states, selected by lon/lat
  data = data.cx[-126:-66, 24:50]
  print(data)

  data.plot()
  mpl.show()


def rasterize1():
  filename = 'cb_2018_us_county_500k/cb_2018_us_county_500k.shp'
  data = gpd.read_file(filename)
  data = data.cx[-126:-66, 24:50]
  print(data)

  west, south, east, north = data.total_bounds
  latitude = 0.5 * (north + south)

  width = 512
  width_degrees = east - west
  height_degrees = north - south
  degrees_per_pixel_lon = width_degrees/width
  degrees_per_pixel_lat = np.cos(latitude*np.pi/180) * degrees_per_pixel_lon
  height = int(width * (height_degrees / width_degrees) * (degrees_per_pixel_lon / degrees_per_pixel_lat))
  transform = rasterio.transform.from_origin(west,
                                             north,
                                             degrees_per_pixel_lon,
                                             degrees_per_pixel_lat)
  print(transform)

  counties = [(geom, i) for i, geom in enumerate(data['geometry'])]
  fill = len(counties)   # Fill with highest county index + 1
  image = rasterio.features.rasterize(counties,
                                      out_shape=(height,width),
                                      transform=transform,
                                      fill=fill)
  tag = 'contiguous48_%d' % width
  print(image.shape, image.dtype)
  np.save(tag+'.npy', image)
  mpl.imsave(tag+'.png', image)

  with open(tag+'_legend.txt', 'w') as f:
    for i, geoid, name in zip(range(len(counties)), data['GEOID'], data['NAME']):
      f.write('%d,%s,%s\n' % (i, geoid, name))

  # Make a list of counties that are too small to show up in image.
  with open(tag+'_small_counties.txt', 'w') as f:
    for i, geoid, name in zip(range(len(counties)), data['GEOID'], data['NAME']):
      g = np.where(image == i)[0]
      if len(g) == 0:
        data2 = data.iloc[i]
        geom = data2['geometry']
        image2 = rasterio.features.rasterize(((geom, 1)), all_touched=True,
            out_shape=(height,width), transform=transform, fill=0)
        print(i, geoid, name, np.sum(image2))
        image2 = image2.ravel()
        indices = np.where(image2 == 1)[0]
        f.write('%d,%s,%s,%s\n' % (i, geoid, name, ' '.join(['%d' % x for x in indices])))


def foo2():
  width = 512
  #width = 1024
  tag = 'contiguous48_%d' % width
  image = np.load(tag+'.npy')
  print(image.shape, image.dtype)
  with open(tag+'_legend.txt') as f:
    lines = f.readlines()
  triples = [line.strip().split(',') for line in lines]
  assert(list(range(len(triples))) == [int(t[0]) for t in triples])

  for i, geoid, name in triples:
    i = int(i)
    g = np.where(image == i)[0]
    if len(g) == 0:
      print(i, geoid, name)




if __name__ == "__main__":
  #investigate1()
  #rasterize1()
  foo2()


