#!/usr/bin/python
# executable for automated segmentation of mouse OCT
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
median_reduction = cv2.medianBlur(binary, 5)

print('finding starting row')
x_coord_arr = collect_coordinates()
top = [(i, x[0]) for i, x in enumerate(x_coord_arr)]

print('removing outliers')
top = reject_outliers(top, m=2)

# remove middle
l = len(top)/2
top = top[:l-30] + top[l+30:]

## TODO--- lookup fitting a line to nth order polynomial
## use 'feeler' to look several pixels ahead/up/down to find next brightest spot


#####################################################################################

# read in and save a jpeg
img = cv2.imread(CUR_IMAGE_PATH)
grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## adaptive gaussian thresholding
# second to last value: focuses on how specific to filter color (higher value = sloppy thresholding)
# last value: switches between black (neg) and white (pos) -- higher values increase intensity
threshold = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 37, -2)

# remove salt and pepper noise- convert to BGR colour channels
median_reduction = cv2.medianBlur(threshold, 5)
img = cv2.medianBlur(median_reduction, 5)
#img = cv2.cvtColor(median_reduction, cv2.COLOR_GRAY2BGR)

## Draw all bottom points connecting top points using despeck
## ROW START is the correlating top[i] row coordinate
bot = [] # array of coordinates
for pair in top:
	for y in range(pair[1], rows-1): 
		x = pair[0]
		# white to black
		if img[y, x] == 255 and img[y+1, x] == 0:
			bot.append((x, y+1))
			break	

print('removing outliers')
bot = reject_outliers(bot, m=2)

# remove middle
l = len(bot)/2
bot = bot[:l-30] + bot[l+30:]


## TODO: use binary img to find the absolute
## bottom then use the despeck image (img) to find the top of that

#brightness of a pixel using BGR/RGB:
# sum([R, G, B]) / 3 

# connect all points
draw(top, original)
draw(bot, original)


# writing modified image files

NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_modified.jpg'
cv2.imwrite(NEW_IMAGE_PATH, original)

# shows top and bottom clearly but nothing else
#cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_binary.jpg', binary)

cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_despeck.jpg', img)