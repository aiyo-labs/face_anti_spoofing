#!/usr/bin/env python
# encoding: utf-8

import os, sys
import numpy
import collections

#import bob.ip.base
#import bob.ip.facedetect
'''
def scale_image(image, width, height):
  """scales an image.

  Parameters
  ----------
  image: numpy.ndarray
    The image to be scaled.
  width: int
    The new image width.
  height: int
    The new image height

  Returns
  -------
  result: numpy.ndarray
    The scaled image
  
  """
  assert len(image.shape) == 3, "This is meant to work with color images (3 channels)"
  result = numpy.zeros((3, width, height))
  bob.ip.base.scale(image, result)
  return result


def crop_face(image, bbx, facewidth):
  """crops a face from an image.
  
  Parameters
  ----------
  image: numpy.ndarray
    The image to be scaled.
  bbx: :py:class:`bob.ip.facedetect.BoundingBox`
    The bounding box of the face.
  facewidth: int
    The width of the face after cropping.

  Returns
  -------
  face: numpy.ndarray
    The face image.
  """
  face = image[:, bbx.topleft[0]:(bbx.topleft[0] + bbx.size[0]), bbx.topleft[1]:(bbx.topleft[1] + bbx.size[1])]
  aspect_ratio = bbx.size_f[0] / bbx.size_f[1] # height/width
  faceheight = int(facewidth * aspect_ratio)
  face = scale_image(face, faceheight, facewidth)
  face = face.astype('uint8')
  return face
'''

def build_bandpass_filter(fs, order, min_freq=0.7, max_freq=4.0, plot=False):
  """builds a butterworth bandpass filter.
  
  Parameters
  ----------
  fs: float
    sampling frequency of the signal (i.e. framerate).
  order: int
    The order of the filter (the higher, the sharper).
  min_freq: int
    The order of the filter (the higher, the sharper).
  order: int
    The order of the filter (the higher, the sharper).
  plot: bool
    Plots the frequency response of the filter.
  
  Returns
  -------
  b: numpy.ndarray
    The coefficients of the FIR filter.
  
  """
  # frequency range in Hertz, corresponds to plausible heart-rate values, i.e. [42-240] beats per minute
  min_freq = 0.7 
  max_freq = 4.0 

  from scipy.signal import firwin 
  nyq = fs / 2.0
  numtaps = order + 1
  b = firwin(numtaps, [min_freq/nyq, max_freq/nyq], pass_zero=False)

  # show the frequency response of the filter
  if plot:
    from matplotlib import pyplot
    from scipy.signal import freqz
    w, h = freqz(b)
    fig = pyplot.figure()
    pyplot.title('Bandpass filter frequency response')
    pyplot.plot(w * fs / (2 * numpy.pi), 20 * numpy.log10(abs(h)), 'b')
    pyplot.axvline(x=min_freq, color="red")
    pyplot.axvline(x=max_freq, color="red")
    pyplot.ylabel('Amplitude [dB]', color='b')
    pyplot.xlabel('Frequency [Hz]')
    pyplot.show()

  return b

'''
def get_parameter(args, configuration, keyword, default):
  """ Get the right value for a parameter

  The parameters are either defined in a separate configuration file
  or given directly via command-line. Note that the command-line
  has priority over the configuration file.

  As a convention, parameters made with more than one word 
  are provided with an underscore in the configuration file, and with an
  hyphen in the command-line:

    - configuration:  batch_size=64
    - command line:   --batch-size=64

  Parameters
  ----------
  args: dictionary
    The arguments as parsed from the command line.
  configuration: object
    The arguments given by the configuration file.
  keyword: string
    the keyword for the parameter to process (in the "configuration" style)
  default: 
    The default value of the parameter

  Returns
  -------
  arg:
    The right value for the given keyword argument

  """
  
  # get the keyword in a "docopt" friendly format
  args_kw = '--' + keyword.replace('_', '-')
  
  # get the type of this argument
  _type = type(default)

  # get the default value 
  arg_default = default 

  # get the argument in the configuration file 
  if hasattr(configuration, keyword):
    arg_config = getattr(configuration, keyword)
  else:
    arg_config = None

  # get the argument from the command-line
  if default is not None:
    arg_command = _type(args[args_kw])
  else:
    arg_command = default 

  # if the argument was not specified in the config file
  if not hasattr(configuration, keyword):
    return arg_command
  else:
    if (arg_command == arg_default):
      return arg_config
    else:
      return arg_command

'''

