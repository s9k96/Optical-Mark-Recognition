import cv2
import numpy as np
import matplotlib.pyplot as pyplot
import imutils
from transform import order_points, four_point_transform, sort_contours, draw_contour
"""
1. Detect the rectangle of the image.
2. Perspective transform to get the sheet rectangle.
"""

NUMBER_OF_OPTIONS =5

ANSWER_KEY = {0: 1, 1: 4, 2: 0, 3: 3, 4: 1}

img = cv2.imread('test-img.png', 1)
cv2.imshow("Original-Image", img)
blurred = cv2.GaussianBlur(img, (5, 5), 0)
edged = cv2.Canny(blurred, 75, 200)


# cv2.imshow("Edged", edged)


cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL,
	cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
docCnt = None


if len(cnts)>0:
	cnts =  sorted(cnts, key= cv2.contourArea, reverse=True)
	for c in cnts:
		peri = cv2.arcLength(c, True)                      # The perimeter of each contour, sorted by area
		approx = cv2.approxPolyDP(c, 0.02 * peri, True)
		print(len(approx))
		if len(approx) == 4:
			print("Found the sheet")
			docCnt = approx
			break

# cv2.drawContours(img, [docCnt], -1, (0, 0 , 200), 2)
# cv2.imshow("Outline", img)

# print(docCnt)
paper = four_point_transform(img, docCnt.reshape(4,2))

warped = four_point_transform(img, docCnt.reshape(4,2))

# cv2.imshow("Perspective Transform", warped)
warped_gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(warped_gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

# cv2.imshow("thresh", thresh)

cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
questionCnts = []

for c in cnts:
	x, y, w, h = cv2.boundingRect(c)

	aspect_ratio = w/float(h)

	if w>=20 and h>=20 and aspect_ratio>=0.9 and aspect_ratio<=1.1:
		questionCnts.append(c)



# cv2.drawContours(warped, questionCnts, -1, (0, 0 , 200), 2)
# cv2.imshow("Questions contours", warped)

questionCnts = sort_contours(questionCnts, method = 'top-to-bottom')[0]

print("Questions found : ", len(questionCnts)/NUMBER_OF_OPTIONS)


correct = 0
for (q, i) in enumerate(np.arange(0, len(questionCnts), NUMBER_OF_OPTIONS)):
	cnts = sort_contours(questionCnts[i:i + 5])[0]
	bubbled = None
	for j, c in enumerate(cnts):
		# draw_contour(warped, c, j)

		mask = np.zeros(thresh.shape, dtype = 'uint8')
		cv2.drawContours(mask, [c], -1, 255, -1 )

		mask = cv2.bitwise_and(thresh, thresh, mask=mask)
		total = cv2.countNonZero(mask)

		if bubbled is None or total > bubbled[0]:
			bubbled = (total, j)
			# print(total)

	color = (0,0,255)
	k = ANSWER_KEY[q]

	if k == bubbled[1]:
		color=(0,255,0)
		correct+=1
	cv2.drawContours(warped, [cnts[k]], -1, color, 3)

print("Correct answers : ",correct)
cv2.imshow("Questions contours", warped)

cv2.waitKey(0)