import png
import numpy as np

"""
Write a 16-bit png grayscale image of an array, scaled by some integer to fill
the 16-bit range of values, for visualization.
Set scale to some integer, or leave it as "None" for auto-scaling.
"""
def write16bit(array_in, filename='test16.png', scale=None):
  assert(len(array_in.shape) == 2)
  assert(array_in.dtype == 'uint16')
  if scale is None:
    scale = int(np.floor((2**16 - 1) / np.amax(array_in)))
  print('scale = ', scale)
  array = array_in * scale
  print(array.shape, array.dtype)
  num_rows = array.shape[0]
  num_columns = array.shape[1]
  png_writer = png.Writer(num_columns, num_rows, greyscale=True, bitdepth=16)
  with open(filename, 'wb') as f:
    png_writer.write(f, array)
  return array, scale

def read16bit(filename, auto_collapse=False, scale=None):
  png_reader = png.Reader(filename)
  num_columns, num_rows, png_data, properties = png_reader.read()
  print(properties)
  assert(properties['bitdepth'] == 16)
  assert(properties['greyscale'] == True)
  assert(properties['planes'] == 1)
  array = np.vstack(tuple(map(np.uint16, png_data)))
  assert(len(array.shape) == 2)
  assert(array.dtype == 'uint16')
  assert(num_rows == array.shape[0])
  assert(num_columns == array.shape[1])
  if scale is not None:
    array = array // scale
    print('scale = ', scale)
  elif auto_collapse:
    scale = 1
    unique = np.unique(array)
    if (len(unique) > 1):
      scale = unique[1]
    print('scale = ', scale)
    array = array // scale
  return array

def foo2():
  array = np.arange(256, dtype='uint16')
  print(array.shape, array.dtype)
  array = np.tile(array, (128,1))
  print(array.shape, array.dtype)
  print(array)
  new_array, scale = write16bit(array, filename='test16.png')
  print(new_array)
  print(new_array // scale)

  print(read16bit('test16.png', auto_collapse=True))
  print(read16bit('test16.png', scale=257))

if __name__ == "__main__":
  foo2()
