#!/usr/bin/python
# executable for automated segmentation of mouse OCT
import cv2
import numpy as np
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

### TODO-- potentially repeat several times with the starting row incremented
### Draw lines across the top (array) line of points
###
#################################
# generates a list of sets where 
# black changes to white
# **every index is the column coordinate
# **(y * 10)
#################################
x_coord_arr = [] # array of x coordinate sets
for j in range(0, cols):

	# temp set for each column
	row_set = set()
	for i in range(rows-1):
		# black to white
		if binary[i, j] == 0 and binary[i+1, j] == 255:
			row_set.add(i)

	x_coord_arr.append(row_set)

# takes in two sets and finds the intersection 
# within a +-5 range
# returns [min, max] vertical range
def intersect(a, b):
	set_inter = set()
	for ai in a:
		for bi in b:
			# verify range of numbers
			for r in range(-5, 5):
				if ai == bi + r:
					set_inter.add(ai)

	return [min(set_inter), max(set_inter)]

print('Calculating intersections')
##################################
# TODO: iterate through mins and maxes and eliminate outliers (>15%?)
# Goes through and marks segments 
# in first and last half of image
##################################

top = []
# first half
for j in range(0, cols/2-100, 10):
	inter = intersect(x_coord_arr[j], x_coord_arr[j+5])
	cv2.line(img, (j, inter[0]), (j, inter[1]), (0,255,0), 2)
	pair = [j, inter[0]]
	top.append(pair)

# prune outliers- percent change
# top_dif = float(abs(top[0] - top[i]))/top[i] * 100

# back half
for j in range(cols/2+100, cols, 10):
	inter = intersect(x_coord_arr[j], x_coord_arr[j+5])
	cv2.line(img, (j, inter[0]), (j, inter[1]), (0,255,0), 2)
	
#######################################

### TODO handle y coordinates when some x coordinates get removed
print('pruning outliers')
# data must be numpy array
def reject_outliers(data, m=1):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

test = []
for i in top:
	test.append(i[1])
print(test)
t = reject_outliers(np.array(test))
print(t)

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