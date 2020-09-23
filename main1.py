import copy
import random

import csv
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

  #width = 512
  width = 1024 
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
      indices = np.where(image == i)[0]
      # Check for counties covering zero pixels
      if len(indices) == 0:
        # Rasterize that county individually, touching more pixels
        geom = (data.iloc[i])['geometry']
        image2 = rasterio.features.rasterize(((geom, 1)), all_touched=True,
            out_shape=(height,width), transform=transform, fill=0)
        print(i, geoid, name, np.sum(image2))
        image2 = image2.ravel()
        indices = np.where(image2 == 1)[0]
        f.write('%d,%s,%s,%s\n' % (i, geoid, name, ' '.join(['%d' % x for x in indices])))


def get_weights(image):
  hist = np.bincount(image.ravel())
  weights = np.zeros_like(hist, dtype='float32')
  weights[hist > 0] = 1.0 / hist[hist > 0]
  return weights

def load_legend(tag):
  # 0,21007,Ballard
  with open(tag+'_legend.txt') as f:
    lines = f.readlines()
  triples = [line.strip().split(',') for line in lines]
  assert(list(range(len(triples))) == [int(t[0]) for t in triples])
  geoid_to_index = {int(t[1]): int(t[0]) for t in triples}
  return triples, geoid_to_index

def load_small_counties(tag):
  with open(tag+'_small_counties.txt') as f:
    lines = f.readlines()
  index_to_pixels = {}
  for line in lines:
    tokens = line.split(',')
    index = int(tokens[0])
    geoid = int(tokens[1])
    pixel_indices = [int(t) for t in tokens[3].split()]
    index_to_pixels[index] = pixel_indices
  return index_to_pixels


def animate1():
  #width = 512
  width = 1024
  tag = 'contiguous48_%d' % width

  triples, geoid_to_index = load_legend(tag)
  index_to_pixels = load_small_counties(tag)
  image = np.load(tag+'.npy')
  print(image.shape, image.dtype)
  print(np.amax(image))
  weights = get_weights(image)
  print(weights.shape, weights.dtype)
  print(weights)
  filename='COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
  index_to_row = {}
  with open(filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = reader.__next__()
    for row in reader:
      if row[0][0:3] != '840':   # US country code?
        continue
      geoid = int(row[0][3:])
      if geoid in geoid_to_index:
        index = geoid_to_index[geoid]
        index_to_row[index] = row
      else:
        print('  not found', row[10])

  keys = list(index_to_row.keys())
  keys.sort()
  print(len(keys))
  print(set(range(len(keys))) - set(keys))

  vmax = 25
  case_array = np.array([[int(x) for x in index_to_row[i][11:]] for i in range(len(keys))])
  # Put a column of zeros at the start of case_array
  case_array = np.hstack((np.zeros((len(keys),7)), case_array))
  diffs = (case_array[:,7:] - case_array[:,:-7]) / 7.0
  print(case_array.shape, case_array.dtype)
  print(diffs.shape, diffs.dtype)

  days = header[11:]
  print(days[0:10])
  print(len(days))

  orig_shape = image.shape
  for d, day in enumerate(days):
    # Add a zero at the end, for the filler index (the ocean)
    new_cases = np.hstack((diffs[:,d], np.zeros((1,))))
    new_cases[new_cases < 0] = 0
    new_cases = new_cases * weights
    new_cases[-1] = 0.5 * vmax # ocean
    case_image = new_cases[image]
    case_image = case_image.ravel()
    for key in index_to_pixels:
      indices = index_to_pixels[key]
      case_image[indices] += new_cases[key] / len(indices)
    case_image.shape = orig_shape
    print(d, day, np.amax(case_image))

    mpl.imsave('frames/frame%04d.png' % d, case_image, vmin=0, vmax=vmax)






if __name__ == "__main__":
  #investigate1()
  #rasterize1()
  animate1()


