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


def load_legend(tag):
  # 0,21007,Ballard
  with open(tag+'_legend.txt') as f:
    lines = f.readlines()
  triples = [line.strip().split(',') for line in lines]
  assert(list(range(len(triples))) == [int(t[0]) for t in triples])
  geoid_to_index = {int(t[1]): int(t[0]) for t in triples}
  return triples, geoid_to_index


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

