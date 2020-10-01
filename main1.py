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
  #width = 2048
  width = 3840  # 4K TV resolution?
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


def foo():
  array = np.arange(256)
  array = np.tile([array], (256,1))
  print(array.shape)
  array = array.astype('uint8')
  print(array.shape, array.dtype)
  print(array)

  mpl.imsave('test.png', array)
  data = mpl.imread('test.png')
  print(data.shape, data.dtype)
  print(np.amin(data), np.amax(data))
  mpl.imsave('test2.png', data[:,:,0:3])


def make_numeric(x):
  try:
    x = int(x)
  except:
    try:
      x = float(x)
    except:
      pass
  return x


def csv_to_dataframe(csv_filename):
  with open(csv_filename, newline='') as csvfile:
    reader = csv.reader(csvfile)
    header = reader.__next__()
    rows = []
    for row in reader:
      row = [make_numeric(t) for t in row] 
      rows.append(row)
  data = pandas.DataFrame(rows, columns=header)
  return data


def foo2():
  filename='COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
  data = csv_to_dataframe(filename)
  print(data.columns)
  print(data.columns[11:])
  print(data)

  dates = data.columns[11:]
  data = data[dates]

  for c in data.columns[0:10]:
    print(c, data[c].sum())

  sums = data.sum(axis=0)
  print(sums)
  print(len(sums))


def foo3():
  filename='COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
  data = csv_to_dataframe(filename)
  dates = data.columns[11:]
  data = data[dates]
  sums = data.sum(axis=0)

  sums = sums.to_numpy()
  print(sums.shape, sums.dtype)
  sums7 = np.hstack((np.zeros(7, dtype='int'), sums))
  diffs = (sums7[7:] - sums7[:-7]) / 7.0
  print(diffs.shape, diffs.dtype)
  peak = np.amax(diffs)
  print(peak)

  width = 1024
  height = 128
  pixels_per_day = 4

  image = np.zeros((height, width), dtype='float')
  def to_row(count):
    if count < 0:
      count = 0
    row = height - int(height * count / peak)
    return row

  for i in range(len(diffs)):
    row = to_row(diffs[i])
    col0 = i * pixels_per_day
    col1 = (i+1) * pixels_per_day
    image[row:,col0:col1] = 0.5

  font = pix.load_6x4_font()
  print(font.shape)
  font = np.repeat(font, 4, axis=1)
  font = np.repeat(font, 4, axis=2)
  print(font.shape)
  for i in range(len(diffs)):
    print(i, dates[i])
    rasterized = pix.string_to_array(font, dates[i])
    rasterized[rasterized > 1] = 1
    h, w = rasterized.shape
    image[0:h, 0:w] = rasterized

    row = to_row(diffs[i])
    col0 = i * pixels_per_day
    col1 = (i+1) * pixels_per_day
    image[row:,col0:col1] = 1
    mpl.imsave('frames/graph%04d.png' % i, image)


def foo4():
  width = 3840
  tag = 'contiguous48_%d' % width
  filename = tag+'_legend.txt'
  legend = pandas.read_csv(filename)
  print(legend)

  filename='COVID-19/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_US.csv'
  data = pandas.read_csv(filename)
  data['geoid'] = data['FIPS'].fillna(0).astype(np.int64)
  print(data)
  combined = legend.merge(data, on='geoid')
  print(combined)
  dates = data.columns[11:-1]
  print(dates)
  diff = combined['pixel_value'] - np.arange(len(combined['pixel_value']))
  assert(np.amax(np.abs(diff)) == 0)

  tag = 'contiguous48_%d' % width
  image = png16.read16bit(tag+'_16bit.png', auto_collapse=True)
  print(np.amin(image), np.amax(image))

  num_counties = len(combined['geoid'])
  for date in dates:
    values = np.zeros((num_counties + 1), dtype='float32')
    values[0:num_counties] = combined[date]
    cases = values[image]
    print(date, cases.shape, cases.dtype)


def animate2():
  font = pix.load_6x4_font()
  font = np.repeat(font, 16, axis=1)
  font = np.repeat(font, 16, axis=2)

  width = 3840
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

  for i, date in enumerate(dates):
      #if (i > 200):
      if (i > 7):
        diff = (combined[date] - combined[dates[i-7]]) / 7.0
      else:
        diff = combined[date] / 7.0
      total_cases = np.sum(diff)
      values = np.zeros((num_counties + 1), dtype='float32')
      values[0:num_counties] = diff * weights
      peak = np.amax(values)
      values = values / np.amax(values)
      case_image = values[image]
      rgba = colormap(case_image, bytes=True)
      print(i, date, peak, total_cases)
      rgba[ocean[0],ocean[1],0] = 0
      rgba[ocean[0],ocean[1],1] = 0
      rgba[ocean[0],ocean[1],2] = 127
      strings = [date, '7-day avg cases: %.1f' % total_cases]
      text = pix.stringlist_to_array(font, strings)
      rgb = pix.add_to_rgb_image(rgba[:,:,0:3],
                                 text,
                                 rgba.shape[0] - text.shape[0] - 16,
                                 16,
                                 color=[255,255,255])

      mpl.imsave('frames%d/frame%04d.png' % (width, i), rgb)


if __name__ == "__main__":
  #investigate1()
  #rasterize1()
  #animate1()
  #foo()
  #foo2()
  #foo3()
  #foo4()
  animate2()


