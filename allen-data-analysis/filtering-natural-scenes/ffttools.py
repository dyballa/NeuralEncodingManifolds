####### DISCLAIMER ############
# All the code below was adapted from https://github.com/Pella86/DenoiseAverage
# It contains helper functions for handling images, computing their Fourier transforms,
# and applying masks to them to implement the low-, high-, and band-pass filters.


# -*- coding: utf-8 -*-
"""
Created on Mon Jun 19 16:33:12 2017

@author: Mauro

This class manages gray scale images. The images are stored as mxn arrays and 
the class provide basic processing metods
"""

#==============================================================================
# # Imports
#==============================================================================

# numpy import
import numpy as np

# matplotlib import
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

# scipy import
from scipy import signal

# py imports
from copy import deepcopy

#==============================================================================
# # Image Handling class
#==============================================================================

class MyImage(object):
    ''' Main class, this will be a standard image object mxn matrix of values'''
    
    # ------ initialization functions ------
    
    def __init__(self, data = np.zeros((5,5))):  
        ''' The image can be initiated with any numpy array, default is a 5x5 
        zeroed matrix. The image can be intiated by tuple indicating its size
        (mxn). The image can be initiated by a path to an image.
        The data is stored in the self data folder.
        Usage:
            img = MyImg()
            img = MyImg(np.zeros((512, 512)))
            img = MyImg((512, 512))
            img = MyImg(path/to/picture.png)
        '''
        if type(data) == np.ndarray:
            self.data = data
        elif type(data) == tuple:
            if len(data) == 2:
                self.data = np.zeros(data)
        elif type(data) == str:
            # shall i check for path being an image?
            self.read_from_file(data)
        else:
            raise ValueError("data type not supported")
            
    
    # ------ debug options ------
    
    def inspect(self, output = True):
        ''' short function that returns the image values: mean,
        standard deviation, max, min and size of image
        if output is True, it prints to the console the string containing the 
        formatted value
        ''' 
        m = np.mean(self.data)
        s = np.std(self.data)
        u = np.max(self.data)
        l = np.min(self.data)
        d = self.data.shape
        
        if output:
            s  = "Mean: {0:.2f} | Std: {1:.2f} | Max: {2:.2f}|Min: {3:.2f} | \
                  Dim: {4[0]}x{4[1]}".format(m, s, u, l, d)
            print(s)
            return s
            
        return (m, s, u, l, d)   
    
    def show_image(self, cmap = "gray", ax=None, center_crop=None, posonly=False):
        ''' This prepares a image canvas for matlibplot visible in  the 
        IPython console.
        '''
        data = deepcopy(self.data)  # copy the data to not modify them
        if posonly:
            data[data < 0] = 0
        # limit the data between 0 and 1
        npic = (data - data.min())/float((data.max()-data.min()))
#         mi = np.min(data)
#         pospic = data + mi
#         m = np.max(pospic)
#         npic = pospic / float(m)
#         data = 1 - npic 
        
        if center_crop is not None:
            assert type(center_crop) == int
            ci, cj = data.shape[0]//2, data.shape[1]//2
            data = data[ci-center_crop:ci+center_crop, cj-center_crop:cj+center_crop]
        
        # show the image in greyscale
        if ax is None:
            plt.imshow(data, cmap=cmap)    
        else:
            ax.imshow(data, cmap=cmap)
    
    def get_size(self):
        return (self.data.shape[0], self.data.shape[1])
    
    def get_sizex(self):
        return self.get_size()[0]
    
    def get_sizey(self):
        return self.get_size()[1]
    
    # ------ I/O functions ------
    
    def read_from_file(self, filepathname):
        ''' import image from file using the mpimg utility of matplotlib'''
        # todo warnings about file existing ?
        self.data = mpimg.imread(filepathname)

    def save(self, filename):
        ''' saves an image using the pyplot method'''
        plt.imsave(filename, self.data)    


    # ------ operators overload  ------       
    def __add__(self, rhs):
        ''' sums two images px by px'''
        self.data = self.data + rhs.data
        return MyImage(self.data)
    
    def __truediv__(self, rhs):
        ''' divide image by scalar (myimg / number)'''
        rpic = deepcopy(self.data)
        for x in range(self.data.shape[0]):
            for y in range(self.data.shape[1]):
                rpic[x][y] = self.data[x][y] / rhs
        
        return MyImage(rpic)

    # ------ editing functions ------
    def create_composite_right(self, rhsimage):
        ''' concatenates 2 images on the right'''
        # todo multiple arugments
        # enlarge the array to fit the next image
        self.data = np.concatenate((self.data, rhsimage.data), axis = 1)
    
    def normalize(self):
        ''' normalize the picture values so that the resulting image will have
        mean = 0 and std = 1'''
        m = np.mean(self.data)
        s = np.std(self.data)
        self.data = (self.data - m) / s

    def convert2grayscale(self):
        ''' when importing an rgb image is necessary to calculate the
        luminosity and reduce the array from mxnx3 to mxn
        '''
        self.data = np.dot(self.data[...,:3], [0.299, 0.587, 0.114])
    
    def transpose(self):
        ''' transposes the picture from mxn to nxm'''
        self.data.transpose()
        
    def binning(self, n = 1):
        ''' Averages a matrix of 2x2 pixels to one value, effectively reducing
        size by two and averaging the value, giving less noise. n indicates
        how many time the procedure is done 
        512x512 bin 1 -> 256x256 bin 2 -> 128128 bin 3 -> ...
        '''
        for i in range(n):
            # initialize resulting image
            rimg = np.zeros( (int(self.data.shape[0] / 2) , int(self.data.shape[1] / 2)))
            
            # calculate for each pixel the corresponding
            # idx rimg = 2*idx srcimg
            for x in range(rimg.shape[0]):
                for y in range(rimg.shape[1]):
                    a = self.data[x*2]    [y*2]
                    b = self.data[x*2 + 1][y*2]
                    c = self.data[x*2]    [y*2 + 1]
                    d = self.data[x*2 + 1][y*2 + 1]
                    rimg[x,y] =  (a + b + c + d) / 4.0
                    
            
            self.data = rimg

    def move(self, dx, dy):
        ''' moves the picture by the dx or dy values. dx dy must be ints'''
        # correction to give dx a right movement if positive
        dx = -dx
        
        # initialize the image
        mpic = np.zeros(self.data.shape)
        
        # get image size
        sizex = mpic.shape[0]
        sizey = mpic.shape[1]
        
        for x in range(sizex):
            for y in range(sizey):
                xdx = x + dx
                ydy = y + dy
                if xdx >= 0 and xdx < sizex and ydy >= 0 and ydy < sizey:
                    mpic[x][y] = self.data[xdx][ydy]
        
        self.data = mpic
    
    def squareit(self, mode = "center"):
        ''' Squares the image. Two methods available 
        center: cuts a square in the center of the picture
        left side: cuts a square on top or on left side of the pic
        '''
        if mode == "center":
            lx = self.data.shape[0]
            ly = self.data.shape[1]
            
            if lx > ly:
                ix = int(lx / 2 - ly / 2)
                iy = int(lx / 2 + ly / 2)
                self.data = self.data[ ix : iy , 0 : ly]
            else:
                ix = int(ly / 2 - lx / 2)
                iy = int(ly / 2 + lx / 2)
                self.data = self.data[0 : lx, ix : iy ]            
        if mode == "left side":
            lx = self.data.shape[0]
            ly = self.data.shape[1]
            
            if lx > ly:
                self.data = self.data[0:ly,0:ly]
            else:
                self.data = self.data[0:lx,0:lx]
    
    def correlate(self, image):
        ''' scipy correlate function. veri slow, based on convolution'''
        corr = signal.correlate2d(image.data, self.data, boundary='symm', mode='same')
        return Corr(corr)

    def limit(self, valmax):
        ''' remaps the values from 0 to valmax'''
        # si potrebbe cambiare da minvalue a value
        mi = self.data.min()
        mi = np.abs(mi)
        pospic = self.data + mi
        m = np.max(pospic)
        npic = pospic / float(m)
        self.data = npic * valmax
    
    def apply_mask(self, mask):
        ''' apply a mask on the picture with a dot product '''
        self.data = self.data * mask.data
        
    def rotate(self, deg, center = (0,0)):
        ''' rotates the image by set degree'''
        #where c is the cosine of the angle, s is the sine of the angle and
        #x0, y0 are used to correctly translate the rotated image.
        
        # size of source image
        src_dimsx = self.data.shape[0]
        src_dimsy = self.data.shape[1]
        
        # get the radians and calculate sin and cos
        rad = np.deg2rad(deg)
        c = np.cos(rad)
        s = np.sin(rad)
        
        # calculate center of image
        cx = center[0] + src_dimsx/2
        cy = center[1] + src_dimsy/2
        
        # factor that moves the index to the center
        x0 = cx - c*cx - s*cx
        y0 = cy - c*cy + s*cy
        
        # initialize destination image
        dest = MyImage(self.data.shape)
        for y in range(src_dimsy):
            for x in range(src_dimsx):
                # get the source indexes
                src_x = int(c*x + s*y + x0)
                src_y = int(-s*x + c*y + y0)
                if src_y > 0 and src_y < src_dimsy and src_x > 0 and src_x < src_dimsx:
                    #paste the value in the destination image
                    dest.data[x][y] = self.data[src_x][src_y]
                    
        self.data = dest.data

    def flip_H(self):
        sizex = self.data.shape[0] - 1
        sizey = self.data.shape[1] - 1
        for x in range(int(sizex / 2)):
            for y in range(sizey):
                tmp = self.data[x][y]
                
                self.data[x][y] = self.data[sizex - x][y]
                self.data[sizex - x][y] = tmp

    def flip_V(self):
        sizex = self.data.shape[0] - 1
        sizey = self.data.shape[1] - 1
        for x in range(int(sizex)):
            for y in range(int(sizey / 2)):
                tmp = self.data[x][y]
                
                self.data[x][y] = self.data[x][sizey - y]
                self.data[x][sizey - y] = tmp

#==============================================================================
# # Cross correlation image Handling class
#==============================================================================

class Corr(MyImage):
    ''' This class provide additional methods in case the picture is a
    correlation picture.
    '''
    
    def find_peak(self, msize = 5):
        ''' finde the pixel with highest value in the image considers a matrix
        of msize x msize, for now works very good even if size is 1.
        returns in a tuple s, x, y. s is the corrrelation coefficient and
        x y are the pixel coordinate of the peak.
        '''
        #' consider a matrix of some pixels
        best = (0,0,0)
        for x in range(self.data.shape[0] - msize):
            for y in range(self.data.shape[1] - msize):
                # calculate mean of the matrix
                s = 0
                for i in range(msize):
                    for j in range(msize):
                        s += self.data[x + i][y + j]
                s =  s / float(msize)**2
                
                # assign the best value to best, the return tuple
                if s > best[0]:
                    best = (s, x, y)
        return best
    
    def find_translation(self, peak):
        ''' converts the peak into the translation needed to overlap completely
        the pictures
        '''
        if type(peak) == int:
            peak = self.find_peak(peak)
        
        #best = self.find_peak(msize)
        peakx = peak[1]
        peaky = peak[2]
        
        dx = -(self.data.shape[0]/2 - peakx)
        dy = self.data.shape[1]/2 - peaky
        
        return int(dx), int(dy)
    
    def show_translation(self, dx, dy):
        ''' prints on the image where the peak is
        usage:
            corr = Corr()
            best = corr.find_peak()
            dx, dy = corr.find_translation(best)
            corr.show_image()
            corr.show_translation(dx, dy)
            plt.show()
        '''
        ody = dx + self.data.shape[0]/2
        odx = self.data.shape[1]/2 - dy
        plt.scatter(odx, ody, s=40, alpha = .5)    
        return odx, ody

    

# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 19:26:28 2017

@author: Mauro
"""

# Image handling class

# Define Imports

# numpy, scipy, matlibplot imports
import numpy as np
from numpy import fft

# py imports
from copy import deepcopy

# my imports
# from MyImage_class import MyImage, Corr, Mask


#==============================================================================
# # Exceptions
#==============================================================================
class FFTnotInit(Exception):
    def __init__(self, value = 0):
        self.value = value
    def __str__(self):
        return "FFTerror: Fourier transform not initialized"

class FFTimagesize(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        s = "."
        try:
            s = "FFTerror: Image size not supported: {0} | {1}".format(self.value[0], self.value[1])
        except Exception as e:
            print(e)
            print(type(e))
          
        return s


#       try:
#           return "FFTerror: Image size not supported {0} | {1}".foramt(self.value[0], self.value[1])
#       except:
#           return "wtf dude??"

#==============================================================================
# # classes
#==============================================================================

class myFFT(object):
    def __init__(self, ft):
        self.ft = ft
    
    def ift(self):
        return MyImage(np.real(fft.ifft2(fft.fftshift(self.ft))))


def map_range(value, leftMin, leftMax, rightMin, rightMax):
    # Figure out how 'wide' each range is
    leftSpan = leftMax - leftMin
    rightSpan = rightMax - rightMin

    # Convert the left range into a 0-1 range (float)
    valueScaled = float(value - leftMin) / float(leftSpan)

    # Convert the 0-1 range into a value in the right range.
    return rightMin + (valueScaled * rightSpan)

class ImgFFT(object):
    
    
    # initialize functions
    #  the class has two behaviors one is the standard and lets the user decide
    #  when to apply specific functions
    #  the other is when a mask is given and the function will automatically 
    #  calculate the images
    def __init__(self, myimage, mask = False):
        self.img = myimage         # original image
        self.imgfft = None            # fourier transform
        self.imgifft = None   # inverted fourier transform 

        
        if mask:
            self.ft()
            self.power_spectrum()
            self.apply_mask(mask)
            self.ift()            
    
    # image editing functions
    def ft(self):
        im = self.img.data.copy()
#         im = (im - im.min())/float((im.max()-im.min()))
#         im -= im.mean() 

        w1 = np.asmatrix(np.cos(np.linspace(-np.pi/2, np.pi/2, im.shape[0])))
        w2 = np.asmatrix(np.cos(np.linspace(-np.pi/2, np.pi/2, im.shape[1])))
        w = np.dot(w1.T,w2)
        im = np.asarray(np.multiply(im,w))
        self.imgfft = fft.fftshift(fft.fft2(im))
    
    def ift(self):
        self.imgifft = MyImage(np.real(fft.ifft2(fft.fftshift(self.imgfft))))
        
    def power_spectrum(self, dolog=True):
        absolutefft = np.abs(self.imgfft)
        xlen = absolutefft.shape[0]
        ylen = absolutefft.shape[1]
        squareftt = deepcopy(absolutefft)
        if dolog:
            nonzero_mask = ~np.isclose(squareftt,0)
            squareftt[nonzero_mask] = np.log(squareftt[nonzero_mask])**2
#         for i in range(xlen):
#             for j in range(ylen):
#                 if squareftt[i][j] != 0:
#                     squareftt[i][j] = np.log(squareftt[i][j])**2
        realpart = np.real(squareftt)
        ps = MyImage(realpart)
        
        return ps
    
    def apply_mask(self, mask):
        self.imgfft = self.imgfft * mask.data
    
    def get_real_part(self):
        r = MyImage(np.real(self.imgfft))       
        r.limit(1)        
        return r

    def get_imag_part(self):
        r = MyImage(np.imag(self.imgfft))  
        r.limit(1)
        return r
    # Correlate functions
    
    def get_magnitude(self):
        sizeimg = np.real(self.imgfft).shape
        
        mag = np.sqrt(np.real(self.imgfft)**2 + np.imag(self.imgfft)**2)
#         mag = np.zeros(sizeimg)
#         for x in range(sizeimg[0]):
#             for y in range(sizeimg[1]):
#                 mag[x][y] = np.sqrt(np.real(self.imgfft[x][y])**2 + np.imag(self.imgfft[x][y])**2)
        rpic = MyImage(mag)
        rpic.limit(1)
        return rpic
    
    def get_phases(self):
        sizeimg = np.real(self.imgfft).shape
        mag = np.arctan2(np.real(self.imgfft), np.imag(self.imgfft))
#         mag = np.zeros(sizeimg)
#         for x in range(sizeimg[0]):
#             for y in range(sizeimg[1]):
#                 mag[x][y] = np.arctan2(np.real(self.imgfft[x][y]), np.imag(self.imgfft[x][y]))
        rpic = MyImage(mag)
        rpic.limit(1)
        return rpic    



    def get_polar_t(self):
        mag = self.get_magnitude()
        sizeimg = np.real(self.imgfft).shape
        
        pol = np.zeros(sizeimg)
        for x in range(sizeimg[0]):
            for y in range(sizeimg[1]):
                my = y - sizeimg[1] / 2
                mx = x - sizeimg[0] / 2
                if mx != 0:
                    phi = np.arctan(my / float(mx))
                else:
                    phi = 0
                r   = np.sqrt(mx**2 + my **2)
                
                ix = map_range(phi, -np.pi, np.pi, sizeimg[0], 0)
                iy = map_range(r, 0, sizeimg[0], 0, sizeimg[1])

                if ix >= 0 and ix < sizeimg[0] and iy >= 0 and iy < sizeimg[1]:
                    pol[x][y] =  mag.data[int(ix)][int(iy)]    
        pol = MyImage(pol)
        pol.limit(1)
        return pol
    
    def correlate(self, imgfft):
        #Very much related to the convolution theorem, the cross-correlation
        #theorem states that the Fourier transform of the cross-correlation of
        #two functions is equal to the product of the individual Fourier
        #transforms, where one of them has been complex conjugated:  
        
        
        if self.imgfft != 0 or imgfft.imgfft != 0:
            imgcj = np.conjugate(self.imgfft)
            imgft = imgfft.imgfft
            
            prod = deepcopy(imgcj)
            for x in range(imgcj.shape[0]):
                for y in range(imgcj.shape[0]):
                    prod[x][y] = imgcj[x][y] * imgft[x][y]
            
            cc = Corr( np.real(fft.ifft2(fft.fftshift(prod)))) # real image of the correlation
            
            # adjust to center
            cc.data = np.roll(cc.data, int(cc.data.shape[0] / 2), axis = 0)
            cc.data = np.roll(cc.data, int(cc.data.shape[1] / 2), axis = 1)
        else:
            raise FFTnotInit()
        return cc
    
    def resize_image(self, sizex, sizey):
        imsizex = self.img.data.shape[0]
        imsizey = self.img.data.shape[1]
        
        if sizex > imsizex or sizey > imsizey:
            raise FFTimagesize((imsizex, imsizey))
        else:
            l2x = imsizex / 2
            l2y = imsizex / 2
            
            if self.imgfft is None:
                raise FFTnotInit()
            else:
                xl = int(l2x - sizex / 2)
                xu = int(l2x + sizex / 2)
                yl = int(l2y - sizey / 2)
                yu = int(l2y + sizey / 2)
                imgfft = np.array(self.imgfft[xl : xu, yl : yu])
                fftresized = myFFT(imgfft)
                
                return fftresized.ift()

#==============================================================================
# # Mask image Handling class
#==============================================================================

class Mask(MyImage):
    ''' This class manages the creation of masks
    '''
    
    def create_circle_mask(self, radius, smooth, high_pass=False):
        ''' creates a smoothed circle with value 1 in the center and zero
        outside radius + smooth, uses a linear interpolation from 0 to 1 in 
        r +- smooth.
        '''
        # initialize data array
        dims = self.data.shape
        mask = np.ones(dims)*0.5
        center = (dims[0]/2.0, dims[1]/2.0)
        for i in range(dims[0]):
            for j in range(dims[1]):
                # if distance from center > r + s = 0, < r - s = 1 else 
                # smooth interpolate
                dcenter = np.sqrt( (i - center[0])**2 + (j - center[1])**2)
                if dcenter >= (radius + smooth):
                    mask[i][j] = 0
                elif dcenter <= (radius - smooth):
                    mask[i][j] = 1
                else:
                    y = -1*(dcenter - (radius + smooth))/radius
                    mask[i][j] = y
        self.data = mask
        
        # normalize the picture from 0 to 1
        self.limit(1) 
        
        if high_pass:
            self.invert()
        
        return self.data
    
    def invert(self):
        self.data = 1 - self.data 
    
    def bandpass(self, rin, sin, rout, sout):
        ''' To create a band pass two circle images are created, one inverted
        and pasted into dthe other'''
        
        # if radius zero dont create the inner circle
        if rin != 0:
            self.create_circle_mask(rin, sin)
        else:
            self.data = np.zeros(self.data.shape)
        
        # create the outer circle
        bigcircle = deepcopy(self)
        bigcircle.create_circle_mask(rout, sout)
        bigcircle.invert() 
        
        # sum the two pictures
        m = (self + bigcircle)
        
        # limit fro 0 to 1 and invert 
        m.limit(1)
        m.invert()  
        
        self.data = m.data
    
    def __add__(self, rhs):
        ''' overload of the + operator why is not inherited from MyImage?'''
        
        self.data = self.data + rhs.data
        return Mask(self.data)
    