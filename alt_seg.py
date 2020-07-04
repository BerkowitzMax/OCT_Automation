#!/usr/bin/python
# executable for automated segmentation of mouse OCT
# FIXME-- looping variable names are inaccurate
import cv2
import numpy as np
import math
import scipy.signal as sig
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

CUR_IMAGE_PATH = '/home/maxberko/seg_automation/example_stack.jpg'
original = cv2.imread(CUR_IMAGE_PATH)

def collect_coordinates(row_start=0):
	x_coord_arr = [] # array of x coordinate sets
	for j in range(0, cols):

		# temp set for each column
		row_set = []
		for i in range(row_start, rows-1):
			# black to white
			if binary[i, j] == 0 and binary[i+1, j] == 255:
				row_set.append(i)

		x_coord_arr.append(row_set)
	return x_coord_arr

# data must be numpy array
# stdev default
def reject_outliers(data, m=1):
	y = [y[1] for y in data]

	elements = np.array(y)
	mean = np.mean(elements, axis=0)
	sd = np.std(elements, axis=0)

	# gathers list of data points to remove
	iter_list = []
	for i in range(len(y)):
		if (y[i] < mean - m * sd) or (y[i] > mean + m * sd):
			iter_list.append(data[i])

	# remove points from data set
	# that were collected in iter_list
	for e in iter_list:
		if e in data:
			data.remove(e)

	# remove middle
	l = len(data)/2
	data = data[:l-30] + data[l+30:]

	return data

# takes in a list of pairs
def draw(arr, d_img):
	for i in range(len(arr)-1):
		p1 = arr[i]
		p2 = arr[i+1]
		cv2.line(d_img, (p1[0], p1[1]), (p2[0], p2[1]), (0,255,0), 2)

print('running threshold')

############################################
# make binary copy which will not be saved
# but is used to determine where to 
# draw lines on the original image
### VALUES: 0-black, 255-white
############################################
img_bw = cv2.imread(CUR_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
rows, cols = img_bw.shape

(thresh, binary) = cv2.threshold(img_bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
## additional manipulation to remove any outliers
binary = cv2.medianBlur(binary, 5)

print('finding starting row')
x_coord_arr = collect_coordinates()
top = [(i, x[0]) for i, x in enumerate(x_coord_arr)]

print('removing outliers top')
top = reject_outliers(top, m=2)

## TODO--- lookup fitting a line to nth order polynomial
#####################################################################################

# read in and save a jpeg
grayscaled = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)

## adaptive gaussian thresholdng
# second to last value: focuses on how specific to filter color (higher value = sloppy thresholding)
# last value: switches between black (neg) and white (pos) -- higher values increase intensity
threshold = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 37, -2)
#37 -2
# remove salt and pepper noise- convert to BGR colour channelis
median_reduction = cv2.medianBlur(threshold, 5)
img_thresh = cv2.medianBlur(median_reduction, 5)
#img = cv2.cvtColor(median_reduction, cv2.COLOR_GRAY2BGR)

################################
# uses thresholded image and top arr
# of coordinate points to mark the lower
# strip of the upper bound
################################
bot = [] # array of coordinates
for pair in top:
	for y in range(pair[1], rows-1): 
		x = pair[0]
		# white to black
		if img_thresh[y, x] == 255 and img_thresh[y+1, x] == 0:
			bot.append((x, y+1))
			break	

print('removing outliers top bottom')
bot = reject_outliers(bot, m=2)

# takes in 2 lists of x and y coordinates
# fits data to 12th degree polynomial
# plots original data with fitted data and returns
# best fit
def curve_fit(x, y):
	plt.plot(x, y, 'bo-')

	x = np.array(x)
	y = np.array(y)

	x = x.reshape(-1, 1)
	poly = PolynomialFeatures(degree=12)
	X_poly = poly.fit_transform(x)
	poly.fit(X_poly, y)

	linreg = LinearRegression()
	linreg.fit(X_poly, y)

	y_pred = linreg.predict(X_poly)

	# fitted polynomial
	plt.plot(x, y_pred, color='red')
	plt.savefig('0_plot.png')

	return y_pred

# TODO-- take a cut across the row instead of a column
r = 280

x, y = [], []
for c in range(0, cols/2, 10):
	x.append(c)
	y.append(sum(original[r, c])/3)

fit_y = curve_fit(x, y)

# writing to image
for i in range(len(fit_y)):
	if y[i] > fit_y[i]:
		original[r, x[i]] = [0,255, 0]

'''
#for c in range(0, cols/2-40, 10):
c = 70
# TEMPORARILY JUST FOR COLUMN 70
last = (0, 0)
for r in range(rows-1, 0, -1):
	if img_thresh[r, c] == 0 and img_thresh[r-1, c] == 255:
		last = (c, r)
		break

# using c = 70 as a test example
# curve fitting 
x = []
y = []
for r in range(top[c][1], last[1]):
	x.append(r)
	y.append(sum(original[r, c])/3)

fit_y = curve_fit(x, y)

# writing to image
for i in range(len(fit_y)):
	if y[i] > fit_y[i]:
		original[x[i], c] = [0,255, 0]
'''

# writing modified image files
NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_modified.jpg'
cv2.imwrite(NEW_IMAGE_PATH, original)

# shows top and bottom clearly but nothing else
#cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_binary.jpg', binary)

cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_despeck.jpg', img_thresh)

# gaussian peaks
# fourier transform
# maximum entropy

# comparing dilation to original image
