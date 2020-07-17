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
	data = data[:l-60] + data[l+60:]

	return data

# takes in a list of pairs (x, y)
def draw(arr, d_img, color=(0, 255, 0)):
	for i in range(len(arr)-1):
		p1 = arr[i]
		p2 = arr[i+1]
		cv2.line(d_img, (p1[0], p1[1]), (p2[0], p2[1]), color, 1)

# takes in 2 lists of x and y coordinates
# fits data to 12th degree polynomial
# plots original data with fitted data and returns
# best fit
def curve_fit(x, y, dgr=12):
	plt.plot(x, y, 'bo-')

	x = np.array(x)
	y = np.array(y)

	x = x.reshape(-1, 1)
	poly = PolynomialFeatures(degree=dgr)
	X_poly = poly.fit_transform(x)
	poly.fit(X_poly, y)

	linreg = LinearRegression()
	linreg.fit(X_poly, y)

	y_pred = linreg.predict(X_poly)

	# fitted polynomial
	#plt.plot(x, y_pred, color='red')
	#plt.savefig('0_plot.png')

	return y_pred

# takes in and produces a single array of points
# calls curve_fit to smoothen out an existing array
# fits to 9th degree polynomial
def smoothen(arr):
	x, y = [], []
	for p in range(len(arr)/2):
		x.append(arr[p][0])
		y.append(arr[p][1])
	pred_y = curve_fit(x, y, dgr=9)

	combined = []
	for i in range(len(pred_y)):
		combined.append((x[i], int(pred_y[i])))
	return combined

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
top_lower = [] # array of coordinates
for pair in top:
	for y in range(pair[1], rows-1): 
		x = pair[0]
		# white to black
		if img_thresh[y, x] == 255 and img_thresh[y+1, x] == 0:
			top_lower.append((x, y+1))
			break	
top_lower = reject_outliers(top_lower, m=2)


print('retina bottom')
# use thresholded image to find bottom of retina
bot = []
for c in range(cols):
	for r in range(rows-1, 0, -1):
		if img_thresh[r, c] == 0 and img_thresh[r-1, c] == 255:
			bot.append((c, r))
			break

bot = reject_outliers(bot, m=1)

# Create a 'frame' to adjust final outline to, ensures the correct number
# of lines are in place

## TODO -- give every pixel a weighting, iterate through every 
## 10 rows-- place a point in the center (5th) and create a diameter
## of 5 all the way around, select next highest weighting thats also the rightmost (vertical) point

# combining thresholded img with original to highlight
# areas of interest
bgr_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
original = cv2.addWeighted(original, 0.7, bgr_thresh, 0.3, 0)

print('drawing top, top-bottom, and bottom segments')
# smoothes out curves
smooth_top = smoothen(top)
draw(smooth_top, original)

smooth_toplower = smoothen(top_lower)
draw(smooth_toplower, original)

smooth_bot = smoothen(bot)
draw(smooth_bot, original)

#####################################################################################
# averages top and bottom lines defining the retina
med = []
for c in range(cols):
	green = []
	for r in range(rows-1):
		if list(original[r, c]) != [0, 255, 0] and list(original[r+1, c]) == [0, 255, 0]:
			green.append((c, r))
	if len(green) == 3:
		avg = (green[0][1] + green[1][1])/2
		original[avg, c] = [0, 255, 0]
		med.append((c, avg))

# find otsu's threshold value with OpenCV function
cur_img = cv2.imread(CUR_IMAGE_PATH,0)
blur = cv2.GaussianBlur(cur_img,(5,5),0)
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
tval = ret

# line: (c, r)
#####################################################################
# takes in a list of coordinates and produces a weighting based on 
# otsu's binarization possible weight measures: 
# 		* most of the lines has to be above otsu thresh value
#		* take avg grayscale value
# decisions based on img_thresh and otsu's binarization
#####################################################################
def give_weight(line, thresh):
	above = 0
	for i in line:
		if img_thresh[i[1], i[0]] > thresh:
			above += 1
	return above > len(line)/2

# (c, r)
# line iterations-- going downwards
shift = 0
for r in range(med[0][1], bot[0][1]):
	cur_line = []
	shift += 1

	for p in med:
		cur_line.append((p[0], p[1]+shift))

	if give_weight(cur_line, tval):
		draw(cur_line, original)

# writing modified image files
NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_modified.jpg'
cv2.imwrite(NEW_IMAGE_PATH, original)

# shows top and bottom clearly but nothing else
#cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_binary.jpg', binary)

cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_despeck.jpg', img_thresh)



# ADDITIONAL NOTES #########################
# gaussian peaks
# fourier transform
# maximum entropy

# comparing dilation to original image

# TODO-- try straightening the image out in imagej
# then do the ctrl+k filter thing to find peaks
# edit -> options -> draw a line (500 thickness) across one of the curves
# edit -> selection -> straighten
