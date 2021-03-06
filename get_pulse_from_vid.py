from lib.processor_loadvid import findFaceGetPulse
from lib.interface import plotXY, imshow, waitKey, destroyWindow
from cv2 import moveWindow
import argparse
import numpy as np 
import datetime
import socket
import sys
import cv2

class getPulseApp(object):

	"""
	Python applicatoin that finds a face from an uploaded video, then isolates 
	the forehead. Then the average green-light intesnity in the forehead 
	region is gathered over time, and the detected persons' pulse is estimated. 
	"""

	def __init__(self, args, filename):
		serial = args.serial 
		baud = args.baud
		self.send_serial = False
		self.send_udp = False
		if serial:
			self.send_serial = True
			if not baud:
				baud = 9600
			else:
				baud = int(baud)
			self.serial = Serial(port=serial, baudrate=baud)

		udp = args.udp
		if udp:
			self.send_udp = True
			if ":" not in udp:
				ip = udp
				port = 5005
			else:
				ip, port = udp.split(":")
				port = int(port)
			self.udp = (ip, port)
			self.sock = socket.socket(socket.AF_INET, # Internet
				 socket.SOCK_DGRAM) # UDP

		self.vidcap = cv2.VideoCapture(filename)
		# if (vidcap.isOpened() == False)
		# 	print("Error opening video")
		# Containerized analysis of recieved image frames (an openMDAO assembly)
		# is defined next.

		# This assembly is designed to handle all image & signal analysis,
		# such as face detection, forehead isolation, time series collection,
		# heart-beat detection, etc.

		# Basically, everything that isn't communication
		# to the camera device or part of the GUI
		self.processor = findFaceGetPulse(bpm_limits=[50, 160],
										  data_spike_limit=2500.,
										  face_detector_smoothness=10.)

		# Init parameters for the cardiac data plot
		self.bpm_plot = True
		self.plot_title = "Data display - raw signal (top) and PSD (bottom)"

		# Maps keystrokes to specified methods
		#(A GUI window must have focus for these to work)
		self.key_controls = {"s": self.toggle_search,
							 "d": self.toggle_display_plot,
							 "x": self.toggle_color_ch,
							 "z": self.toggle_measuring_area,
							 "f": self.write_csv}

	def write_csv(self):
		"""
		Writes current data to a csv file
		"""
		fn = "Webcam-pulse" + str(datetime.datetime.now())
		fn = fn.replace(":", "_").replace(".", "_")
		data = np.vstack((self.processor.times, self.processor.samples)).T
		np.savetxt(fn + ".csv", data, delimiter=',')
		print("Writing csv")

	def toggle_search(self):
		"""
		Toggles a motion lock on the processor's face detection component.
		Locking the forehead location in place significantly improves
		data quality, once a forehead has been sucessfully isolated.
		"""
		state = self.processor.find_faces_toggle()
		print("face detection lock =", not state)

	def toggle_display_plot(self):
		"""
		Toggles the data display.
		"""
		if self.bpm_plot:
			print("bpm plot disabled")
			self.bpm_plot = False
			destroyWindow(self.plot_title)
		else:
			print("bpm plot enabled")
			if self.processor.find_faces:
				self.toggle_search()
			self.bpm_plot = True
			self.make_bpm_plot()
			moveWindow(self.plot_title, self.w, 0)

	def toggle_color_ch(self):
		"""
		Toggles what color channel is used to calculate BPM.
		"""
		color_ch = self.processor.color_ch_toggle()   
		print("Color channel used = " + color_ch)    

	def toggle_measuring_area(self):
		"""
		Toggles what measuring area is used to calculate BPM and magnify.
		"""
		measuring_area = self.processor.measuring_area_toggle()   
		print("Measuring_area used = " + measuring_area)      

	def make_bpm_plot(self):
		"""
		Creates and/or updates the data display
		"""
		plotXY([[self.processor.times,
				 self.processor.samples],
				[self.processor.freqs,
				 self.processor.fft]],
			   labels=[False, True],
			   showmax=[False, "bpm"],
			   label_ndigits=[0, 0],
			   showmax_digits=[0, 1],
			   skip=[3, 3],
			   name=self.plot_title,
			   bg=self.processor.slices[0])

	def key_handler(self):
		"""
		Handle keystrokes, as set at the bottom of __init__()
		A plotting or camera frame window must have focus for keypresses to be
		detected.
		"""

		self.pressed = waitKey(10) & 255  # wait for keypress for 10 ms
		if self.pressed == 27:  # exit program on 'esc'
			print("Exiting")
			for cam in self.cameras:
				cam.cam.release()
			if self.send_serial:
				self.serial.close()
			sys.exit()

		for key in self.key_controls.keys():
			if chr(self.pressed) == key:
				self.key_controls[key]()

	def main_loop(self):
		"""
		Single iteration of the application's main loop.
		"""
		# Get current image frame from the video 
		success, frame = self.vidcap.read()
		#ret, jpeg = cv2.imencode('.jpg', image)
		#frame = jpeg.tobytes()
		self.h, self.w, _c = frame.shape

		# set current image frame to the processor's input
		self.processor.frame_in = frame
		# process the image frame to perform all needed analysis
		self.processor.run()
		# collect the output frame for display
		output_frame = self.processor.frame_out

		# show the processed/annotated output frame
		imshow("Processed", output_frame)

		# create and/or update the raw data display if needed
		if self.bpm_plot:
			self.make_bpm_plot()

		if self.send_serial:
			self.serial.write(str(self.processor.bpm) + "\r\n")

		if self.send_udp:
			self.sock.sendto(str(self.processor.bpm), self.udp)

		# handle any key presses
		self.key_handler()


if __name__ == "__main__":
	filename = input("Video file for hearbeat extraction (Make sure this file is in the same directory as this program: ");
	parser = argparse.ArgumentParser(description='Webcam pulse detector.')
	parser.add_argument('--serial', default=None,
						help='serial port destination for bpm data')
	parser.add_argument('--baud', default=None,
						help='Baud rate for serial transmission')
	parser.add_argument('--udp', default=None,
						help='udp address:port destination for bpm data')

	args = parser.parse_args()
	App = getPulseApp(args, filename)
	while True:
		App.main_loop()