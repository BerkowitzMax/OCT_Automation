#!/usr/bin/python
# executable for automated segmentation of mouse OCT
import cv2
import numpy as np
import statistics as stat
import os

print("testing")
print(os.getcwd())

# read in and save a jpeg
img = cv2.imread('/home/maxberko/seg_automation/example_stack.jpg')

grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

## adaptive gaussian thresholding
# second to last value: focuses on how specific to filter color (higher value = sloppy thresholding)
# last value: switches between black (neg) and white (pos) -- higher values increase intensity
threshold = cv2.adaptiveThreshold(grayscaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 37, -2.5)

cv2.imwrite('/home/maxberko/seg_automation/modified.jpg', threshold)

print('loading modified image')
#NEXT STEP:
# Process -> Noise -> Remove outliers [radius 2.0, brightness 50]
# Process -> Noise -> Despeckle


# locate vertical color divisions
# opencv uses BGR
img = cv2.imread('/home/maxberko/seg_automation/B_despeck.jpg')

# make binary copy
## this binary will not be saved but will
## be used to determine where to draw lines on the
## grayscaled version
## VALUES: 0-black, 255-white
img_bw = cv2.imread('/home/maxberko/seg_automation/B_despeck.jpg', cv2.IMREAD_GRAYSCALE)
(thresh, binary) = cv2.threshold(img_bw, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

#rows, cols = img.shape
rows = 500
cols = 1000

#################################
# generates a list of sets where 
# black changes to white
# **every index is the column coordinate
# **(y * 10)
#################################
def collect_coordinates(row_start=0):
	x_coord_arr = [] # array of x coordinate sets
	for j in range(0, cols):

		# temp set for each column
		row_set = set()
		for i in range(row_start, rows-1):
			# black to white
			if binary[i, j] == 0 and binary[i+1, j] == 255:
				row_set.add(i)

		x_coord_arr.append(row_set)
	return x_coord_arr

###############################################
# takes in two sets and finds the intersection 
# within a +-5 range
# returns [min, max] vertical range
###############################################
def intersect(a, b):
	set_inter = set()
	for ai in a:
		for bi in b:
			# verify range of numbers
			for r in range(-5, 5):
				if ai == bi + r:
					set_inter.add(ai)

	return [min(set_inter), max(set_inter)]

# [x1, y1] [x2, y2]
def slope(a, b):
	return (b[1] - a[1]) / (b[0] - a[0])

##################################
# Goes through and marks segments 
# in first and last half of image
##################################
def calc_left(x_coord_arr, top):
	# left side of img
	for j in range(0, cols/2-100, 10):
		inter = intersect(x_coord_arr[j], x_coord_arr[j+5])
		#cv2.line(img, (j, inter[0]), (j, inter[1]), (0,255,0), 2)
		pair = [j, inter[0]]
		top.append(pair)
	return top

def calc_right(x_coord_arr):
	# right side of img
	for j in range(cols/2+100, cols, 10):
		inter = intersect(x_coord_arr[j], x_coord_arr[j+5])
		cv2.line(img, (j, inter[0]), (j, inter[1]), (0,255,0), 2)
		pair = [j, inter[0]]
		#top.append(pair)

# data must be numpy array
def reject_outliers(data):
	y = [y[1] for y in data]

	elements = np.array(y)
	mean = np.mean(elements, axis=0)
	sd = np.std(elements, axis=0)

	# gathers list of data points to remove
	iter_list = []
	for i in range(len(y)):
		if (y[i] < mean - 1 * sd) or (y[i] > mean + 1 * sd):
			iter_list.append(data[i])

	# remove points from data set
	# that were collected in iter_list
	for e in iter_list:
		if e in data:
			data.remove(e)

	return data


print('Calculating intersections')
x_coord_arr = collect_coordinates()

top = []
top = calc_left(x_coord_arr, top)

print('pruning outliers')
top = reject_outliers(top)
#top = reject_outliers(top)
#top = reject_outliers(top)
## TODO-- repeat
## restart the process of drawing lines downwards 
# and taking the averages -- calling reject_outliers
# ***start drawing lines on row by sorting by highest coordinate in top[] lst
# *** might need to discard image and re-open

## TODO-- Check angle of point A -> B and B-> C and compare to A -> C

# connects across all top points
for i in range(len(top)-1):
	p1 = top[i]
	p2 = top[i+1]
	cv2.line(img, (p1[0], p1[1]), (p2[0], p2[1]), (0,255,0), 2)

###############################
## Highlights all white areas
###############################
'''
for j in range(0, cols):
	for i in range(0, rows-1):
		# top
		if binary[i, j] == 0 and binary[i+1, j] == 255:
			img[i, j] = [0,255,0]

		# bottom
		if binary[i, j] == 255 and binary[i+1, j] == 0:
			img[i, j] = [0,255,0]
'''
#######################
## coordinates testing
#######################
# Find the center retina-- <5 incidents
'''
center = 0
arr_center = []
for j in range(0, cols):
	for i in range(rows-1):

		if binary[i, j] == 0 and binary[i+1, j] == 255:
			arr_center.append('\0')

	if len(arr_center) < 4:
		center = j
		img[:, center] = [0,255,0]
		print(center)
	arr_center = []
'''


cv2.imwrite('/home/maxberko/seg_automation/despeck_changed.jpg', img)