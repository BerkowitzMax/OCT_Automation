#!/usr/bin/python
# executable for automated segmentation of mouse OCT
# FIXME-- looping variable names are inaccurate
import cv2
import numpy as np
import statistics as stat
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

# connect all points and mark up original image
#draw(top, original)
#draw(bot, original)

## Static threshold
'''
_coordinates = []
mag = []
for pair in bot:
	column = pair[0]
	row = pair[1]

	for r in range(row, rows-1):
		_coordinates.append((r, column))
		mag.append(sum(original[r, column])/3)

peaks = []
for i in range(len(mag)):
	if mag[i] > 112:
		r = _coordinates[i][0]
		c = _coordinates[i][1]
		original[r, c] = [0, 255, 0]
'''

###TODO: if this works it'll be more efficient to work in a matrix
# Adaptive threshold
# create a series of subimages (blocks-- block size)
# apply a threshold to each of these subimages
# can try applying threshold based on the mean
print('adaptive threshold')
def thresh_algorithm(main_img, blocksize, coord):
	mean = 0

	# collect magnitudes per pixel
	_coordinates = []
	mag = []
	for c in range(coord[1], coord[1]+blocksize):
		for r in range(coord[0], coord[0]+blocksize):
			s = sum(main_img[r, c])/3
			mean += s

			_coordinates.append((r, c))
			mag.append(s)

	mean /= (blocksize * blocksize)
	for i in range(len(mag)):
		if mag[i] > mean:
			r = _coordinates[i][0]
			c = _coordinates[i][1]
			main_img[r, c] = [0, 255, 0]

blocksize = 20
for c in range(0, cols, blocksize):
	for r in range(0, rows, blocksize):
		thresh_algorithm(original, blocksize, (r,c))
original = cv2.medianBlur(original, 11)

# writing modified image files
NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_modified.jpg'
cv2.imwrite(NEW_IMAGE_PATH, original)

# shows top and bottom clearly but nothing else
#cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_binary.jpg', binary)

cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_despeck.jpg', img_thresh)