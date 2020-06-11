#!/usr/bin/python
# executable for automated segmentation of mouse OCT
import cv2
import numpy as np
import statistics as stat
import os

CUR_IMAGE_PATH = '/home/maxberko/seg_automation/D.jpg'

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
	elements = np.array(data)
	mean = np.mean(elements, axis=0)
	sd = np.std(elements, axis=0)

	# gathers list of data points to remove
	iter_list = []
	for i in range(len(data)):
		if (data[i] < mean - m * sd) or (data[i] > mean + m * sd):
			iter_list.append(data[i])

	# remove points from data set
	# that were collected in iter_list
	for e in iter_list:
		if e in data:
			data.remove(e)

	return data

img = cv2.imread(CUR_IMAGE_PATH)

print('running threshold')
# read in and save a jpeg
img_bw = cv2.imread(CUR_IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
rows, cols = img_bw.shape

(thresh, binary) = cv2.threshold(img_bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
## additional manipulation to remove any outliers
median_reduction = cv2.medianBlur(binary, 5)


print('finding starting row...')
x_coord_arr = collect_coordinates()
top = [x[0] for x in x_coord_arr]

#print('pruning outliers...')
#top = reject_outliers(top)

# connects across all top points
for i in range(len(top)-1):
	p1 = top[i]
	p2 = top[i+1]
	cv2.line(img, (i, p1), (i+1, p2), (0,255,0), 2)

## TODO--- skip point if it's a suddent change in elevation 
## INSTEAD of removing outliers, REPLACE these points with neighboring points

cv2.imwrite('/home/maxberko/seg_automation/despeck_changed.jpg', img)