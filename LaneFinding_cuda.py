import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import glob
import pickle
from moviepy.editor import VideoFileClip
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

	
""" 
Function to calibrate the camera feed. 
Attempts to read saved calibration coefficients, 
and if no coefficients have been saved previously then new ones are created and saved.
"""
def get_camera_calibration_coefficients():
	try:
		cal_coeffs = pickle.load(open('calibration_coefficients.p','rb'))
		print('Camera calibration coefficients loaded successfully.')
		return cal_coeffs[0], cal_coeffs[1], cal_coeffs[2]
	except (OSError,IOError):
		print('Calibrating camera feed...', end='')
		image_files = glob.glob('camera_cal/calibration*.jpg')
		nx = 9
		ny = 6
		# Create a mesh grid with dimensions (nx, ny)
		objp = np.zeros((nx*ny,3),np.float32)
		objp[:,:2] = np.mgrid[0:nx, 0:ny].T.reshape(-1,2)
		
		imgpoints = []
		objpoints = []
		
		image = mpimg.imread(image_files[0])
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# For each distorted image, locate the chessboard corners
		for image_file in image_files:
			image = cv2.imread(image_file)
			gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
			ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
			if ret:
				imgpoints.append(corners)
				objpoints.append(objp)
		
		ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
		
		src = np.float32([[275,660],[540,484],[740,484],[1010,660]])
		dst = np.float32([[275, 660],[275,275],[1010,275],[1010,660]])
		M = cv2.getPerspectiveTransform(src, dst) 

		print('Done')
		cal_coeffs = [mtx, dist, M]
		pickle.dump(cal_coeffs, open('calibration_coefficients.p','wb'))
		print('Camera calibration coefficients saved')
		return mtx, dist, M

"""
Returns the undistorted image.
"""
def return_undist_image(image,mtx,dist):
	undist = cv2.undistort(image, mtx, dist, None, mtx)
	return undist

"""
Returns the warped image, which is now in a top-view perspective.
"""
def return_warped_image(image,M):
	img_size = (image.shape[1], image.shape[0])
	warped = cv2.warpPerspective(image, M, img_size)
	return warped

"""
Returns the thresholded image using Sobelx, hue, and value thresholding. Output is a binary image.
"""
def return_thresholded_image(image, v_thresh=(0,255), h_thresh=(0, 255), sx_thresh=(30, 100)):
	ksize = 9
	hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS).astype(np.float)
	hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV).astype(np.float)
	v_channel = hsv[:,:,2]
	h_channel = hls[:,:,0]
	l_channel = hls[:,:,1]
	s_channel = hls[:,:,2]
	
	# Sobel x
	sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=ksize) # Take the derivative in x
	abs_sobelx = np.absolute(sobelx)
	scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
	
	# Threshold x gradient	
	sxbinary = np.zeros_like(scaled_sobel)
	sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
	# plt.imshow(sxbinary,cmap='gray')
	# plt.show()
	
	# Threshold v channel
	v_channel = hsv[:,:,2]
	
	v_binary = np.zeros_like(v_channel)
	v_binary[(v_channel >= v_thresh[0]) & (v_channel <= v_thresh[1])] = 1
	
	# Threshold h channel
	h_binary = np.zeros_like(h_channel)
	h_binary[(h_channel >= h_thresh[0]) &(h_channel <= h_thresh[1])] = 1
	# plt.imshow(h_binary,cmap='gray')
	# plt.show()
	
	
	gray_binary = np.zeros_like(v_binary, dtype=np.uint8)
	gray_binary[(v_binary==1) | (sxbinary==1) | (h_binary==1)] = 1
	# plt.imshow(gray_binary,cmap='gray')
	# plt.show()

	return gray_binary

def window_mask(width, height, img_ref, center,level):
	output = np.zeros_like(img_ref)
	output[int(img_ref.shape[0]-(level+1)*height):int(img_ref.shape[0]-level*height),max(0,int(center-width/2)):min(int(center+width/2),img_ref.shape[1])] = 1
	return output

"""
Returns the polynomial fit for the left and right lanes.
"""
def return_polyfit(warped, window_width, window_height, last_leftx, last_rightx, margin, should_use_last_points):
	window_centroids = [] # Store the (left,right) window centroid positions per level
	window = np.ones(window_width) # Create our window template that we will use for convolutions
	
	nonzero = warped.nonzero()
	nonzeroy = np.array(nonzero[0])
	nonzerox = np.array(nonzero[1])
	
	# First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
	# and then np.convolve the vertical image slice with the window template 
	# If we have l_center and r_center values from previous frame, use these as our initial reference. Otherwise, use a window search. 
	if should_use_last_points:
		l_center = last_leftx
		r_center = last_rightx
	else:
		# Reset bad frames counter and do a sliding window search
		l_sum = np.sum(warped[int(3*warped.shape[0]/4):, :400], axis=0)
		l_center = np.argmax(np.convolve(window,l_sum))-window_width/2
		r_sum = np.sum(warped[int(3*warped.shape[0]/4):, 850:], axis=0)
		# if (np.sum(r_sum)) == 0:
			# r_center = l_center + 735
		# else:
			# r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+850
		r_center = np.argmax(np.convolve(window,r_sum))-window_width/2+850
			
	# Add what we found for the first layer
	window_centroids.append((l_center,r_center))
	
	# Go through each layer looking for max pixel locations
	left_lane_inds = []
	right_lane_inds = []
	
	delta_l_center = 0
	old_l_center = l_center
	
	delta_r_center = 0
	old_r_center = r_center
	
	
	for level in range(1,(int)(warped.shape[0]/window_height)):
		win_y_low = warped.shape[0] - (level+1)*window_height
		win_y_high = warped.shape[0] - level*window_height
		win_xleft_low = l_center - margin
		win_xleft_high = l_center + margin
		win_xright_low = r_center - margin
		win_xright_high = r_center + margin
		# convolve the window into the vertical slice of the image
		image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height),:], axis=0)
		conv_signal = np.convolve(window, image_layer)
		# Find the best left centroid by using past left center as a reference
		# Use window_width/2 as offset because convolution signal reference is at right side of window, not center of window
		offset = window_width/2
		l_min_index = int(max(l_center+offset-margin,0))
		l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
		
		# If convolution equals zero, use linear approxiation of the next center point
		if np.sum(conv_signal[l_min_index:l_max_index]) != 0:
			l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index - offset
			delta_l_center = l_center - old_l_center
		else:
			l_center += delta_l_center
		# Find the best right centroid by using past right center as a reference
		r_min_index = int(max(r_center+offset-margin,0))
		r_max_index = int(min(r_center+offset+margin,warped.shape[1]))
		if np.sum(conv_signal[r_min_index:r_max_index]) != 0:
			r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index - offset
			delta_r_center = r_center - old_r_center
		else:
			r_center += delta_r_center
		# Add what we found for that layer
		
		window_centroids.append((l_center,r_center))
		old_l_center = l_center
		old_r_center = r_center
		
		
		# Identify the nonzero pixels in x and y within the window
		good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
		(nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
		good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & \
		(nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
		# Append these indices to the lists
		left_lane_inds.append(good_left_inds)
		right_lane_inds.append(good_right_inds)
			
			
	# Concatenate the arrays of indices
	left_lane_inds = np.concatenate(left_lane_inds)
	right_lane_inds = np.concatenate(right_lane_inds)

	# Extract left and right line pixel positions
	leftx = nonzerox[left_lane_inds]
	lefty = nonzeroy[left_lane_inds] 
	rightx = nonzerox[right_lane_inds]
	righty = nonzeroy[right_lane_inds] 
	

	# Fit a second order polynomial to each
	#	If there are no pixels detected, return None for each fit
	if ((leftx.size == 0) | (lefty.size == 0) | (rightx.size == 0) | (righty.size == 0)):
		left_fit = None
		right_fit = None
	else:
		left_fit = np.polyfit(lefty, leftx, 2)
		right_fit = np.polyfit(righty, rightx, 2)

	return window_centroids, left_fit, right_fit

"""
Function to process the image. Pipeline for incoming image:
	1) Threshold the raw image
	2) Undistort and warp the image to top-view perspective
	3) Call the return_polyfit function to get the best polynomial fit for left and right lanes
	4) Check polynomial fits for usability
	5) Find the radii of curvature and offset of vehicle from the center of the lane
	6) If fits are usable, do a sanity check on the separation between the left and right lanes. 
	7) If data is good, add to averaging array. Averaging array contains n_samples with first-in-first-out structure
	8) If fits are not usable, continue using the previous average values
	9) Apply a mask of the lane identification over the original video feed
"""
def process_image(image):
	global mtx
	global dist
	global M
	global last_leftx
	global last_rightx
	global should_use_last_points
	global coefficient_avgs
	global num_of_successive_bad_frames
	
	# Define conversions in x and y from pixels space to meters
	ym_per_pix = 30/720 # meters per pixel in y dimension
	xm_per_pix = 3.7/700 # meters per pixel in x dimension
	
	# Threshold, undistort, and warp the image
	thresholded_image = return_thresholded_image(image, v_thresh=(220,250), h_thresh=(30,90), sx_thresh=(30,100))
	undist = return_undist_image(thresholded_image, mtx, dist)
	warped = return_warped_image(undist, M)
	undist_color = return_undist_image(image,mtx,dist)
	# plt.imshow(warped,cmap='gray')
	# plt.show()
	
	# Define sliding window search settings
	window_width = 50 
	window_height = 90 # Break image into 9 vertical layers since image height is 720
	margin = 110 # How much to slide left and right for searching
	
	
	# Define the center of the vehicle, assuming camera is mounted in the middle of hte windshield
	center_of_vehicle = warped.shape[1]*xm_per_pix/2.0
	
	window_centroids, left_fit, right_fit = return_polyfit(warped, window_width, window_height, last_leftx, last_rightx, margin, should_use_last_points)
	# If we found any window centers
	if len(window_centroids) > 0:
		
		ploty = np.linspace(0, warped.shape[0]-1, warped.shape[0])
		y_eval = np.max(ploty)
		
		coeffs = np.zeros((2,3))
		if ((left_fit != None) & (right_fit != None)):
			coeffs[0] = left_fit
			coeffs[1] = right_fit
			
			# Generate x and y values for plotting
			left_fitx_current = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
			right_fitx_current = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
			
			# Sample the left and right fit lines to see if parallel avg separation is as expected
			left_fitx_sample = left_fitx_current[0::10]
			right_fitx_sample = right_fitx_current[0::10]
			n_samples = len(left_fitx_sample)
			
			avg_separation = 0
			for (left,right) in zip(left_fitx_sample,right_fitx_sample):
				avg_separation += right - left
			avg_separation /= n_samples
			
			# Store the top most points of each lane to be used as the center for future lane identification

			
			# Fit new polynomials to x,y in world space
			left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx_current*xm_per_pix, 2)
			right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx_current*xm_per_pix, 2)
			# Calculate the new radii of curvature
			left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
			right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
			
			if (avg_separation >= 675) & (avg_separation <= 800) & (np.abs(left_curverad) > 250) & (np.abs(right_curverad) > 250):
				should_use_last_points = True
				for i in range(3): # i indexes through A, B, C
					for j in range(2): # j indexes through left, right
						# indexing example: (0, 1) refers to array of A coefficients for the right lane
						coefficient_avgs[i][j] = np.roll(coefficient_avgs[i][j], -1)
						np.put(coefficient_avgs[i][j], -1, coeffs[j][i])
			else:
				# Got a fit, but the fit is unreasonable -> bad frame
				num_of_successive_bad_frames += 1
		else: 
			# Bad frame, no fit available
			num_of_successive_bad_frames += 1
		
		
		if num_of_successive_bad_frames == 5:
			should_use_last_points = False
			num_of_successive_bad_frames = 0
		
		# Calculate average for each polynomial coefficient for left and right sides
		for i in range(3):
			for j in range(2):
				arr = coefficient_avgs[i][j]
				avg = np.average(arr[arr.nonzero()])
				coeffs[j][i] = avg
		left_fit = coeffs[0]
		right_fit = coeffs[1]
		
		# Generate x and y values for plotting
		left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
		right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
		
		if should_use_last_points:
			last_leftx = left_fitx[0]
			last_rightx = right_fitx[0]

		# Find base of fit lines
		left_base = left_fitx[-1]
		right_base = right_fitx[-1]
		
		# Find the offset of the vehicle from the center of the lane
		center_of_lane = (((right_base - left_base)/2.)+left_base)*xm_per_pix
		offset_of_vehicle = center_of_vehicle - center_of_lane
		
		# Fit new polynomials to x,y in world space
		left_fit_cr = np.polyfit(ploty*ym_per_pix, left_fitx*xm_per_pix, 2)
		right_fit_cr = np.polyfit(ploty*ym_per_pix, right_fitx*xm_per_pix, 2)
		
		# Calculate the new radii of curvature
		left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
		right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
		
		# Create an image to draw the lines on
		warp_zero = np.zeros_like(warped).astype(np.uint8)
		color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

		# Recast the x and y points into usable format for cv2.fillPoly()
		pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
		pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
		pts = np.hstack((pts_left, pts_right))

		# Draw the lane onto the warped blank image
		cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

		# Warp the blank back to original image space using inverse perspective matrix (Minv)
		newwarp = cv2.warpPerspective(color_warp, np.linalg.inv(M), (warped.shape[1], warped.shape[0])) 
		# Combine the result with the original image
		result = cv2.addWeighted(undist_color, 1, newwarp, 0.3, 0)
		output = result
		cv2.putText(output, 'Left lane curvature: {:03.3f} m'.format(left_curverad), (100,100), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
		cv2.putText(output, 'Right lane curvature: {:03.3f} m'.format(right_curverad), (100,125), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
		cv2.putText(output, 'Offset from center of lane: {:03.3f} m'.format(offset_of_vehicle), (100,150), cv2.FONT_HERSHEY_SIMPLEX, .8, (255,255,255), 2)
		# plt.imshow(output)
		# plt.show()
	 
	# If no window centers found, just display orginal road image
	else:
		output = np.array(cv2.merge((warped,warped,warped)),np.uint8)
	return output
	

global mtx
global dist
global M
global last_leftx
global last_rightx
global should_use_last_points
global coefficient_avgs
global num_of_successive_bad_frames

num_of_successive_bad_frames = 0
	
# Calibrate the camera
mtx, dist, M = get_camera_calibration_coefficients()

# 'should_use_last_points' refers to whether lane positions from previous frame should be utilized in finding lanes in next frame.
# Initialize to false.
should_use_last_points = False 

last_leftx = 0
last_rightx = 0

# n_samples is the number of samples used when averaging the fit coefficients
n_samples = 10
coefficient_avgs = np.zeros((3,2,n_samples))


# Process the test images
test_image_files = glob.glob('test_images/*.jpg')
images = []
for test_image_file in test_image_files:
	images.append(mpimg.imread(test_image_file))

# for image in images:
	# process_image(image)
# process_image(images[0])


project_video_annotated = 'annotated_videos/project_video_annotated.mp4'
challenge_video_annotated = 'annotated_videos/challenge_video_annotated.mp4'

project_video = VideoFileClip('project_video.mp4')
clip = project_video.fl_image(process_image)
clip.write_videofile(project_video_annotated, audio=False)

# challenge_video = VideoFileClip('challenge_video.mp4')
# clip = challenge_video.fl_image(process_image)
# clip.write_videofile(challenge_video_annotated, audio=False)


	