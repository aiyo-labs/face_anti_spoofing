#!/usr/bin/env python
# encoding: utf-8

import numpy
#import bob.ip.base
import cv2

def compute_mean_rgb(image, mask=None):
  """computes the mean R, G and B of an image.
  
  Note that a mask could be provided to tell which pixels should
  be taken into account when computing the mean.
  
  Parameters
  ----------
  image: numpy.ndarray 
    The image to process
  mask: numpy.ndarray
    Mask of the size of the image, telling which pixels
    should be considered

  Returns
  -------
  mean_r: float
    The mean red value
  mean_g: float
    The mean green value
  mean_b: float
    The mean blue value
  
  """
  assert len(image.shape) == 3, "This is meant to work with color images (3 channels)"
  mean_r = numpy.mean(image[0, mask])
  mean_g = numpy.mean(image[1, mask])
  mean_b = numpy.mean(image[2, mask])
  return mean_r, mean_g, mean_b


def compute_gray_diff(previous, current):
  """computes the difference in intensity between two images.
  
  Parameters
  ----------
  previous: numpy.ndarray  
    The previous frame.
  current: numpy.ndarray 
    The current frame.
 
  Returns
  -------
  float: 
    The sum of the absolute difference in pixel intensity between two frames
  
  """
  #from bob.ip.color import rgb_to_gray
  prevg = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
  currg = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
  #prevg = rgb_to_gray(previous)
  #currg = rgb_to_gray(current)
  return numpy.sum(numpy.absolute(prevg - currg))


def select_stable_frames(diff, n):
  """selects a stable subset of consecutive frames
 
  The selection is made by considering the grayscale difference between frames.
  The subset is chosen as the one for which the sum of difference is minimized

  Parameters
  ----------
  diff: numpy.ndarray 
    The sum of absolute pixel intensity differences between 
    consecutive frames, across the whole sequence.
  n: int
    The number of consecutive frames you want to select.

  Returns
  -------
  index: int
    The frame index at which the stable segment begins.
  
  """
  current_min = float("inf")
  current_index = 0
  for i in range(0, diff.shape[0]-n, 1):
    current_sum = sum(diff[i: i+n])
    if current_sum < current_min:
      current_index = i
      current_min = current_sum
  return current_index


def project_chrominance(r, g, b):
  """Projects rgb values onto the x and y chrominance space
  
  See equation (9) of [dehaan-tbe-2013]_.

  Parameters
  ----------
  r: float
    The red value
  g: float
    The green value
  b: float
    The blue value

  Returns
  -------
  x: float
    The x value
  y: float
    The y value
  """
  x = (3.0 * r) - (2.0 * g)
  y = (1.5 * r) + g - (1.5 * b)
  return x, y
