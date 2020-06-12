#!/usr/bin/python
# executable for automated segmentation of mouse OCT
import cv2
import numpy as np
import statistics as stat
import os

CUR_IMAGE_PATH = '/home/maxberko/seg_automation/D.jpg'
img = cv2.imread(CUR_IMAGE_PATH)

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

# connects across all top points
for i in range(len(top)-1):
	p1 = top[i]
	p2 = top[i+1]
	cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,255,0), 2)

## TODO--- skip point if it's a suddent change in elevation 
## INSTEAD of removing outliers, REPLACE these points with neighboring points
## after this, create and use the gaussian picture where the lines are cleanly separated
## TODO--- lookup fitting a line to nth order polynomial
## use 'feeler' to look several pixels ahead/up/down to find next brightest spot

cv2.imwrite('/home/maxberko/seg_automation/despeck_changed.jpg', img)