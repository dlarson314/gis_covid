import numpy as np
import matplotlib.pyplot as mpl

"""
https://robey.lag.net/2010/01/23/tiny-monospace-font.html
"""
def load_6x4_font(filename='tom-thumb-4x6_inverted.png'):
    font = np.zeros((256, 6, 4), dtype='uint8')
    data = mpl.imread(filename)
    data = data[:,:,0]

    # This copies in 128 characters.  Rest are blank. 
    for col in range(32):
        for row in range(4):
            index = col + row * 32
            font[index, :, :] = data[row*6:row*6+6, col*4:col*4+4]

    font[font > 0] = 255

    return font


def foo():
    print ("hello world")

    data = mpl.imread('tom-thumb-4x6.png')
    print (data.shape, data.dtype)
    print (np.amin(data), np.amax(data))
    data = data[:,:,0]
    print (data.shape, data.dtype)
    print (np.amin(data), np.amax(data))

    font = load_6x4_font()
    print (font.shape, font.dtype)


def string_to_array(font, string):
    #print len(string)

    height = font.shape[1]
    width = font.shape[2] 

    image = np.zeros((height, width * len(string)), dtype='uint8')
    for i, c in enumerate(string):
        image[:,i*width:(i+1)*width] = font[ord(c),:,:]

    return image


def stringlist_to_array(font, stringlist):
    height = font.shape[1]
    width = font.shape[2] 

    rows = len(stringlist)
    nchar = max(map(len, stringlist))

    image = np.zeros((height * rows, width * nchar), dtype='uint8')
    for r in range(len(stringlist)):
        for i, c in enumerate(stringlist[r]):
            image[r*height:(r+1)*height,i*width:(i+1)*width] = font[ord(c),:,:]

    return image


def sampler(font):
    height = font.shape[1]
    width = font.shape[2] 

    image = np.zeros((height * 8, width * 16), dtype='uint8')
    for r in range(8):
        for c in range(16):
            n = r * 16 + c
            image[r*height:(r+1)*height,c*width:(c+1)*width] = font[n,:,:]

    return image


def foo4():
    font = load_6x4_font()
    image = string_to_array(font, 'This is a test.')
    mpl.imsave('test.png', image)

    image = stringlist_to_array(font, ('This is a test.', "Here's a second row.", "1234567890!@#$%^&*()"))
    mpl.imsave('test2.png', image)


def foo5():
    font = load_6x4_font()
    image = sampler(font)
    mpl.imsave('font.png', image)


#def rebin_image(a, shape, strict2d=False):
def rebin_image(a, shape):
    """ 
    http://stackoverflow.com/questions/8090229/resize-with-averaging-or-rebin-a-numpy-2d-array
    """
    if len(shape) == 2:
        if a.shape[0] > shape[0]:
            assert(a.shape[1] >= shape[1])
            sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1]
            return a.reshape(sh).mean(3).mean(1)
        elif a.shape[0] < shape[0]:
            assert(a.shape[1] <= shape[1])
            return np.repeat(np.tile(a, shape[0]//a.shape[0]), shape[1]//a.shape[1]).reshape(shape)
        else:
            assert(a.shape[1] == shape[1])
            return a
    elif len(shape) == 1:
        if a.shape[0] >= shape[0]:
            sh = shape[0], a.shape[0]//shape[0]
            return a.reshape(sh).mean(1)
        else:
            raise NotImplementedError
    else:
        assert(len(shape) == 3)
        sh = shape[0], a.shape[0]//shape[0], shape[1], a.shape[1]//shape[1], a.shape[2]
        return a.reshape(sh).mean(3).mean(1)


def resize_font(font, width):
    h = font.shape[0]
    w = font.shape[1] 

    height = (h * width) / w
    # print h, w
    #print height, width

    font2 = rebin_image(font, (128, height, width)).astype('uint8')
    return font2


def overlap1d(len1, len2, row):
    r1range = np.arange(len1)
    r2range = np.arange(len2)
    g1 = (r1range >= row) * (r1range < row + len2)
    g2 = (r2range >= -row) * (r2range < -row + len1)

    gg1 = np.where(g1)
    gg2 = np.where(g2)
    #print gg1
    #print gg2

    if len(gg1[0]) > 0:
        assert(len(gg1[0]) == len(gg2[0]))
        r1a = np.amin(gg1[0])
        r1b = np.amax(gg1[0]) + 1
        r2a = np.amin(gg2[0])
        r2b = np.amax(gg2[0]) + 1
    else:
        assert(len(gg2[0]) == 0)
        r1a = 0
        r1b = 0
        r2a = 0
        r2b = 0

    #print len1, len2, row, '; ', r1a, r1b, r2a, r2b

    return r1a, r1b, r2a, r2b


def overlap(im1, im2, row, col):
    """
    place im2 in im1 at row,col, so that
    im1[row,col] = im2[0,0]
    Calculate overlap, so that 
    im1[r1a:r1b,c1a:c1b] = im2[r2a:r2b, c2a:c2b]
    """
    h1, w1 = im1.shape[0:2]
    height, width = im2.shape[0:2]

    r1a, r1b, r2a, r2b = overlap1d(h1, height, row)
    c1a, c1b, c2a, c2b = overlap1d(w1, width, col)

    return r1a, r1b, r2a, r2b, c1a, c1b, c2a, c2b


def add_to_rgb_image(image, text, row, col, color=[255, 255, 255], right_justify=False):
    height = text.shape[0] 
    width = text.shape[1] 

    if right_justify:
        col = col - width

    if len(image.shape) == 2:
        r1a, r1b, r2a, r2b, c1a, c1b, c2a, c2b = overlap(image, text, row, col)
        image[r1a:r1b, c1a:c1b] = \
            (text[r2a:r2b, c2a:c2b] / 255.0) * color[0] + \
            (1 - text[r2a:r2b, c2a:c2b] / 255.0) * image[r1a:r1b, c1a:c1b]
    else:
        assert(len(image.shape) == 3)
        r1a, r1b, r2a, r2b, c1a, c1b, c2a, c2b = overlap(image, text, row, col)
        #print (r1a, r1b, r2a, r2b, c1a, c1b, c2a, c2b)
        for i in range(3):
            image[r1a:r1b, c1a:c1b, i] = \
                (text[r2a:r2b, c2a:c2b] / 255.0) * color[i] + \
                (1 - text[r2a:r2b, c2a:c2b] / 255.0) * image[r1a:r1b, c1a:c1b, i]

    return image


def foo6():
    font = load_6x4_font()
    array = string_to_array(font, "This is a test.")
    print(array)

    rgb = np.zeros((128, 256, 3), dtype='uint8')
    add_to_rgb_image(rgb, array, 0, 0, color=[255,0,0])
    print(rgb)
    mpl.imsave("test6.png", rgb)


if __name__ == "__main__":
    #foo4()
    foo6()
