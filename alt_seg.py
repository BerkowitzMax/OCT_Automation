#!/usr/bin/python
# executable for automated segmentation of mouse OCT

import cv2
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tkinter import *
from PIL import ImageTk, Image

import copy
import os

CUR_IMAGE_PATH = '/home/maxberko/seg_automation/example_stack.tif'
original = cv2.imread(CUR_IMAGE_PATH)
c_original = copy.deepcopy(original)

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
		cv2.line(d_img, p1, p2, color, 1)

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
	for p in range(len(arr)):
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

# combining thresholded img with original to highlight
# areas of interest
bgr_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_GRAY2BGR)
print('drawing top, top-bottom, and bottom segments')

# mark up copy of original with all lines marked
copy = copy.deepcopy(original)
copy = cv2.addWeighted(original, 0.7, bgr_thresh, 0.3, 0)

draw(bot, copy)

## split up into smaller segments
l = len(top)
smooth_top = smoothen(top[:l/8])
draw(smooth_top, copy)

smooth_top = smoothen(top[l/8 : l/4])
draw(smooth_top, copy)

smooth_top = smoothen(top[l/4 : l/2])
draw(smooth_top, copy)


l = len(top_lower)
smooth_toplower = smoothen(top_lower[:l/8])
draw(smooth_toplower, copy)

smooth_toplower = smoothen(top_lower[l/8 : l/4])
draw(smooth_toplower, copy)

smooth_toplower = smoothen(top_lower[l/4 : l/2])
draw(smooth_toplower, copy)

#####################################################################################
# shift curve approximations to best fits
# averages top and bottom lines defining the retina
med = []
for c in range(cols):
	green = []
	for r in range(rows-1):
		if list(copy[r, c]) != [0, 255, 0] and list(copy[r+1, c]) == [0, 255, 0]:
			green.append((c, r))
	if len(green) == 3:
		avg = (green[0][1] + green[1][1])/2
		copy[avg, c] = [0, 255, 0]
		med.append((c, avg))

#####################################################################################
# TOP OF RETINA
# preserve x coordinate, shift y coordinate by a const value 
# to align with top of retina
# TODO just uses first coordinate for now
L_approx = []
const_shift = abs(top[0][1] - med[0][1]) 
for p in med:
	L_approx.append((p[0], p[1] - const_shift))
draw(L_approx, original)

# contains the shift values from the top of the retina
L_shift_values = []

c = 120
# c is currently set, arbitrarily, to 120
# draw a vertical line where shift occurs to mark it up
for r in range(rows):
	c_original[r, c] = [0, 255, 0]

NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_copy.tif'
cv2.imwrite(NEW_IMAGE_PATH, c_original)

### tkinter experimenting
root = Tk()

# image converter to show image using tkinter
tk_img = ImageTk.PhotoImage(Image.open(NEW_IMAGE_PATH))
img_label = Label(image=tk_img)
img_label.pack()


# print current mouse coordinates whenever button 1 is clicked
def Lclick(e):
	lbl.config(text=str(e.x) + ', ' + str(e.y))

	L_shift_values.append(e.y)


lbl = Label(root, text='')
lbl.pack()

root.bind('<Button>', Lclick)


# quit-out button
quit_button = Button(root, text='Exit', command=root.destroy).pack()
# event loop
root.mainloop()


# zeroing values and shifting
for i in range(1, len(L_shift_values)):
	L_shift_values[i] = abs(L_shift_values[i] - L_shift_values[0])
L_shift_values[0] = 0

# TEMP MANUALLY DELETE LAST?
L_shift_values.remove(L_shift_values[-1])
print(L_shift_values)

# FIXME-- raise error if incorrect number

# writing final image file
FINAL_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_modified.tif'
cv2.imwrite(FINAL_IMAGE_PATH, original)


# PLAN
'''
1. Have the original, unaltered image up [done]
2. Have the guessed curve for the right half saved [done?]
3. Draw a vertical line on image [done]
4. Have user click on the vertical lines that intersect with the points of interest [done]
5. Plot based on the shift [done]

^^ repeat for left hand side
ensure that it's the same number of both sides then link them

AFTER:
	-mark on image where user clicked
	-option to hand-draw initial curve
	-option to hand-mark right and left sides?
'''





# alternative copies of image analysis in-progress
#NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_copy.tif'
#cv2.imwrite(NEW_IMAGE_PATH, copy)

#NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_markup.tif'
#cv2.imwrite(NEW_IMAGE_PATH, markup)

# shows top and bottom clearly but nothing else
#cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_binary.tif', binary)

#cv2.imwrite(CUR_IMAGE_PATH.split('.')[0] + '_despeck.tif', img_thresh)







########################################################################################
# COPY starting from ~line 192
original = cv2.imread(CUR_IMAGE_PATH) # start fresh, same process
copy = cv2.addWeighted(original, 0.7, bgr_thresh, 0.3, 0)

draw(bot, copy)

## split up into smaller segments
half = len(top)/2 + 60
l = len(top)
smooth_top = smoothen(top[half : half + l/8])
draw(smooth_top, copy)

smooth_top = smoothen(top[half + l/8 : half + l/4])
draw(smooth_top, copy)

smooth_top = smoothen(top[half + l/4 : half + l/2])
draw(smooth_top, copy)

half = len(top_lower)/2 + 60
l = len(top_lower)
smooth_toplower = smoothen(top_lower[half : half + l/8])
draw(smooth_toplower, copy)

smooth_toplower = smoothen(top_lower[half + l/8 : half + l/4])
draw(smooth_toplower, copy)

smooth_toplower = smoothen(top_lower[half + l/4 : half + l/2])
draw(smooth_toplower, copy)

#####################################################################################
# Creates averaged line
med = []
for c in range(cols):
	green = []
	for r in range(rows-1):
		if list(copy[r, c]) != [0, 255, 0] and list(copy[r+1, c]) == [0, 255, 0]:
			green.append((c, r))
	if len(green) == 3:
		avg = (green[0][1] + green[1][1])/2
		copy[avg, c] = [0, 255, 0]
		med.append((c, avg))

#####################################################################################
# TOP OF RETINA
# preserve x coordinate, shift y coordinate by a const value 
# to align with top of retina
# TODO just uses last coordinate for now
top = top[len(top)/2 + 60 :]
R_approx = []
const_shift = abs(top[-1][1] - med[-1][1]) 
for p in med:
	R_approx.append((p[0], p[1] - const_shift))
draw(R_approx, original)


# contains the shift values from the top of the retina
R_shift_values = []

c = 950
# c is currently set, arbitrarily, to 850
# draw a vertical line where shift occurs to mark it up
for r in range(rows):
	c_original[r, c] = [0, 255, 0]

NEW_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_copy.tif'
cv2.imwrite(NEW_IMAGE_PATH, c_original)

### tkinter experimenting
root = Tk()

# image converter to show image using tkinter
tk_img = ImageTk.PhotoImage(Image.open(NEW_IMAGE_PATH))
img_label = Label(image=tk_img)
img_label.pack()


# print current mouse coordinates whenever button 1 is clicked
def Rclick(e):
	lbl.config(text=str(e.x) + ', ' + str(e.y))
	R_shift_values.append(e.y)

lbl = Label(root, text='')
lbl.pack()

root.bind('<Button>', Rclick)


# quit-out button
quit_button = Button(root, text='Exit', command=root.destroy).pack()
# event loop
root.mainloop()


# zeroing values and shifting
for i in range(1, len(R_shift_values)):
	R_shift_values[i] = abs(R_shift_values[i] - R_shift_values[0])
R_shift_values[0] = 0

# TEMP MANUALLY DELETE LAST?
R_shift_values.remove(R_shift_values[-1])
print(R_shift_values)

# FIXME-- raise error if incorrect number


# takes 'approx' list, which is the zero'd avg line and
# shifts baseline by certain amount
def shift(idx):
	color = (0, 255, 0)
	if idx == 0:
		color = (249, 249, 249)
	elif idx == 1:
		color = (250, 250, 250)
	elif idx == 2:
		color = (252, 252, 252)
	elif idx == 3:
		color = (253, 253, 253)
	elif idx == 4:
		color = (254, 254, 254)
	elif idx == 5:
		color = (255, 255, 255)
	else:
		print('Incorrect number of lines')
		exit()

	L_shft = L_shift_values[idx]
	line = []
	for p in L_approx:
		line.append((p[0], p[1] + L_shft))
	draw(line, original, color=color)


	R_shft = R_shift_values[idx]
	line = []
	for p in R_approx:
		line.append((p[0], p[1] + R_shft))
	draw(line, original, color=color)

	# connect left and right side
	p1 = (L_approx[-1][0], L_approx[-1][1] + L_shft)
	p2 = (R_approx[0][0], R_approx[0][1] + R_shft)
	cv2.line(original, p1, p2, color, 1)


	# clean up edges
	# (left)
	edge = (0, R_approx[0][1] + L_shft)
	p = (L_approx[0][0], R_approx[0][1] + L_shft)
	cv2.line(original, edge, p, color, 1)

	# (right)
	edge = (cols, R_approx[-1][1] + R_shft)
	p = (R_approx[-1][0], R_approx[-1][1] + R_shft)
	cv2.line(original, edge, p, color, 1)


for val in range(len(R_shift_values)):
	shift(val)

# TODO-- PLACE center vertical line (with correct colour) here
# eventually create two lines 20+- 500 (halfway) so person can place line
# within these bounds
t = (500, 0)
b = (500, rows)
cv2.line(original, t, b, (243, 243, 243), 1)

# writing final image file
FINAL_IMAGE_PATH = CUR_IMAGE_PATH.split('.')[0] + '_modified.tif'
cv2.imwrite(FINAL_IMAGE_PATH, original)
