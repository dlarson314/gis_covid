import copy
import random

import csv
from attrdict import AttrDict
import numpy as np
import matplotlib.pyplot as mpl
import rasterio.features
import geopandas as gpd
import pandas

import pixel as pix
import png16


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


def rasterize1(permute=True, seed=123456789):
  filename = 'cb_2018_us_county_500k/cb_2018_us_county_500k.shp'
  data = gpd.read_file(filename)
  data = data.cx[-126:-66, 24:50]
  print(data)

  west, south, east, north = data.total_bounds
  latitude = 0.5 * (north + south)

  #width = 512
  #width = 1024
  width = 2048
  #width = 3840  # 4K TV resolution?
  #width = 4096
  #width = 8192
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

  if permute:
    generator = np.random.default_rng(seed)
    new_indices = generator.permutation(len(data['geometry']))
  else:
    new_indices = np.arange(len(data['geometry']))

  tuples = list(zip(new_indices, data['GEOID'], data['NAME'], data['geometry']))
  tuples.sort()
  print(tuples[0:10])

  counties = list([(t[3], t[0]) for t in tuples])
  fill = len(counties)   # Fill with highest county index + 1
  image = rasterio.features.rasterize(counties,
                                      out_shape=(height,width),
                                      transform=transform,
                                      fill=fill)
  tag = 'contiguous48_%d' % width
  print(image.shape, image.dtype)
  print('saving numpy')
  np.save(tag+'.npy', image)
  print('saving png')
  mpl.imsave(tag+'.png', image)
  print('saving 16 bit grayscale png')
  png16.write16bit(tag+'_16bit.png', image)

  print('writing legend')
  with open(tag+'_legend.txt', 'w') as f:
    #for i, geoid, name in zip(range(len(counties)), data['GEOID'], data['NAME']):
    f.write('"pixel_value","geoid","county_name"\n')
    for t in tuples:
      f.write('%d,"%s","%s"\n' % (t[0], t[1], t[2]))

  # Make a list of counties that are too small to show up in image.
  print('checking for small counties')
  found = set(np.unique(image))
  desired = set(range(len(counties)))
  missing = list(desired - found)
  missing.sort()
  print(missing)

  with open(tag+'_small_counties.txt', 'w') as f:
    for m in missing:
      t = tuples[m]
      dummy, geoid, name, geom = t
      assert(dummy == m)
      indices = np.where(image == m)[0]
      # Check for counties covering zero pixels
      if len(indices) == 0:
        # Rasterize that county individually, touching more pixels
        geom = t[3]
        image2 = rasterio.features.rasterize(((geom, 1)), all_touched=True,
            out_shape=(height,width), transform=transform, fill=0)
        print(m, geoid, name, np.sum(image2))
        image2 = image2.ravel()
        indices = np.where(image2 == 1)[0]
        f.write('%d,%s,%s,%s\n' % (m, geoid, name, ' '.join(['%d' % x for x in indices])))

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

def graph_histogram(hist, height, color=(255,255,255), peak=None):
  width = len(hist)
  if peak is None:
    peak = np.amax(hist)
  if peak == 0:
    peak = 1
  def to_row(count):
    if count < 0:
      count = 0
    row = height - int(height * count / peak)
    return row

  image = np.zeros((height, width, 3), dtype='uint8')
  for col in range(len(hist)):
    row = to_row(hist[col])
    for c in range(3):
      image[row:,col,c] = color[c]
  return image

def animate2():
  #width = 3840
  width = 2048

  font = pix.load_6x4_font()
  font_scale = width // 256
  font = np.repeat(font, font_scale, axis=1)
  font = np.repeat(font, font_scale, axis=2)

  tag = 'contiguous48_%d' % width
  filename = tag+'_legend.txt'
  legend = pandas.read_csv(filename)
  filename='COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
  data = pandas.read_csv(filename)
  data['geoid'] = data['FIPS'].fillna(0).astype(np.int64)
  combined = legend.merge(data, on='geoid')
  dates = data.columns[11:-1]
  diff = combined['pixel_value'] - np.arange(len(combined['pixel_value']))
  assert(np.amax(np.abs(diff)) == 0)

  tag = 'contiguous48_%d' % width
  image = png16.read16bit(tag+'_16bit.png', auto_collapse=True)

  num_counties = len(combined['geoid'])
  weights = get_weights(image)[0:num_counties]

  colormap = mpl.cm.get_cmap(name='magma')

  ocean = np.where(image == num_counties)

  hist_height = 128

  history = [0]
  hist_expand = int((width + hist_height) / len(dates)) - 1
  for i, date in enumerate(dates):
      #if (i > 250):
      if (i > 7):
        diff = (combined[date] - combined[dates[i-7]]) / 7.0
      else:
        diff = combined[date] / 7.0
      total_cases = np.sum(diff)
      history.append(total_cases)
      values = np.zeros((num_counties + 1), dtype='float32')
      values[0:num_counties] = diff * weights
      peak = np.amax(values)
      # values = values / np.amax(values)
      case_image = values[image]
      top_hist = np.sum(case_image, axis=0)
      side_hist = np.sum(case_image, axis=1)
      #print(top_hist.shape, top_hist.dtype, np.amax(top_hist))
      #print(side_hist.shape, side_hist.dtype, np.amax(side_hist))

      color=(127,127,127)
      hist_image_top = graph_histogram(top_hist, hist_height, color=color)
      hist_image_side = graph_histogram(side_hist, hist_height, color=color)
      hist_image_side = np.transpose(hist_image_side, axes=(1,0,2))
      hist_image_side = hist_image_side[:,::-1,:]

      history_image = graph_histogram(history, hist_height, color=(127,0,0))
      history_image = np.repeat(history_image, hist_expand, axis=1)
      row0 = np.zeros((hist_height, width + hist_height, 3), dtype='uint8')
      row0[:,0:history_image.shape[1],:] = history_image

      corner = np.zeros((hist_height, hist_height, 3), dtype='uint8')

      case_image = case_image / peak
      rgba = colormap(case_image, bytes=True)
      print(i, date, peak, total_cases)
      rgba[ocean[0],ocean[1],0] = 0
      rgba[ocean[0],ocean[1],1] = 0
      rgba[ocean[0],ocean[1],2] = 127
      strings = [date, '7-day avg cases: %.1f' % total_cases]
      text = pix.stringlist_to_array(font, strings)
      rgb = pix.add_to_rgb_image(rgba[:,:,0:3],
                                 text,
                                 rgba.shape[0] - text.shape[0] - font_scale,
                                 font_scale,
                                 color=[255,255,255])

      row1 = np.hstack((hist_image_top, corner))
      row2 = np.hstack((rgb, hist_image_side))
      rgb = np.vstack((row0, row1, row2))

      mpl.imsave('frames%d/frame%04d.png' % (width, i), rgb)


if __name__ == "__main__":
  #investigate1()
  #rasterize1()
  animate2()


