import numpy as np
import time
import cv2
import pylab
import os
import sys
from scipy import signal


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


class findFaceGetPulse(object):

    def __init__(self, bpm_limits=[], data_spike_limit=250,
                 face_detector_smoothness=10):

        self.height = 720
        self.width = 1280
        self.frame_in = np.zeros((self.height, self.width))
        self.frame_out = np.zeros((self.height, self.width))
        self.fps = 0
        self.buffer_size = 250
        self.mean_buffer = []
        self.subframe_buffer = []
        self.times = []
        self.t0 = time.time()
        self.samples = []
        self.freqs = []
        self.fft = []
        self.slices = [[0]]
        self.bpm = 0
        dpath = resource_path("haarcascade_frontalface_alt.xml")
        if not os.path.exists(dpath):
            print("Cascade file not present!")
        self.face_cascade = cv2.CascadeClassifier(dpath)
        self.face_rect = [1, 1, 2, 2]
        self.forehead_coords = (0.5, 0.17, 0.30, 0.18) # (right, down, wide, tall)
        self.last_center = np.array([0, 0])
        self.output_dim = 13
        self.find_faces = True 

        self.method = 0

        # for gaussian pyramid
        self.levels = 3

    def method_toggle(self):
        if self.method == 0:
            self.method = 1
            method = "Eulerian"
        else: 
            self.method = 0
            method = "Naive"
        return method

    def get_method(self):
        if self.method == 0:
            method = "Naive"
        else: 
            method = "Eulerian"
        return method

    def find_faces_toggle(self):
        self.find_faces = not self.find_faces
        return self.find_faces

    def shift(self, detected):
        x, y, w, h = detected
        center = np.array([x + 0.5 * w, y + 0.5 * h])
        shift = np.linalg.norm(center - self.last_center)
        self.last_center = center
        return shift

    def draw_rect(self, rect, col=(41, 37, 204)):
        x, y, w, h = rect
        cv2.rectangle(self.frame_out, (x, y), (x + w, y + h), col, 1)

    def get_subface_coord(self, fh_x, fh_y, fh_w, fh_h):
        x, y, w, h = self.face_rect
        return [int(x + w * fh_x - (w * fh_w / 2.0)),
                int(y + h * fh_y - (h * fh_h / 2.0)),
                int(w * fh_w),
                int(h * fh_h)]

    def get_subface_vals(self, coord):
        x, y, w, h = coord
        subframe = self.frame_in[y:y + h, x:x + w, :]
        v1 = np.mean(subframe[:, :, 0])
        v2 = np.mean(subframe[:, :, 1])
        v3 = np.mean(subframe[:, :, 2])
        filtered_subframe = self.buildGpyr(subframe, self.levels+1)[self.levels]
        return filtered_subframe, (v1 + v2 + v3) / 3.

    # Helper Methods for gaussian pyramids
    def buildGpyr(self, subframe, levels):
        pyr = [subframe]
        for level in range(levels):
            subframe = cv2.pyrDown(subframe)
            pyr.append(subframe)
        return pyr
    def rebuildSubframe(self, pyr, levels):
        filtered_subframe = pyr[-1]
        for level in range(levels):
            filtered_subframe = cv2.pyrUp(filtered_subframe)
        _, _, w, h = self.get_subface_coord(*(self.forehead_coords))
        filtered_subframe = filtered_subframe[:h, :w]
        return filtered_subframe    

    def run(self, cam):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        if self.find_faces:
            cv2.putText(
                self.frame_out, "Press 'C' to change camera (current: %s)" % str(cam),
                (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'S' to lock face and begin",
                       (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'X' to change signal processing method (current: %s)" % self.get_method(),
                        (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                       (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
            self.mean_buffer, self.subframe_buffer, self.times = [], [], []
            detected = list(self.face_cascade.detectMultiScale(self.gray,
                                                               scaleFactor=1.3,
                                                               minNeighbors=4,
                                                               minSize=(
                                                                   50, 50),
                                                               flags=cv2.CASCADE_SCALE_IMAGE))

            if len(detected) > 0:
                detected.sort(key=lambda a: a[-1] * a[-2])

                if self.shift(detected[-1]) > 6:
                    self.face_rect = detected[-1]
            forehead1 = self.get_subface_coord(*(self.forehead_coords))
            self.draw_rect(self.face_rect, col=(96, 94, 211))
            x, y, w, h = self.face_rect
            col2 = (0, 94, 184)
            cv2.putText(self.frame_out, "Face",
                       (x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)
            self.draw_rect(forehead1)
            x, y, w, h = forehead1
            cv2.putText(self.frame_out, "Forehead",
                       (x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)
            return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        cv2.putText(
            self.frame_out, "Press 'C' to change camera (current: %s)" % str(
                cam),
            (10, 25), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
        cv2.putText(self.frame_out, "Press 'S' to restart",
                   (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, col)
        cv2.putText(self.frame_out, "Press 'X' to change signal processing method (current: %s)" % self.get_method(),
                   (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)           
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                   (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, col)

        forehead1 = self.get_subface_coord(*(self.forehead_coords))
        self.draw_rect(forehead1)

        # Grab subframe and subframe's average values 
        subframe, mean = self.get_subface_vals(forehead1)

        self.mean_buffer.append(mean)
        self.subframe_buffer.append(subframe)
        L = len(self.mean_buffer)
        if L > self.buffer_size:
            self.mean_buffer = self.mean_buffer[-self.buffer_size:]
            self.subframe_buffer = self.subframe_buffer[-self.buffer_size:]
            self.times = self.times[-self.buffer_size:]
            L = self.buffer_size

        self.samples = np.array(self.mean_buffer)
        subframes = np.array(self.subframe_buffer, copy = True)
        if L > 10:
            self.output_dim = self.samples.shape[0]
            
            self.fps = float(L) / (self.times[-1] - self.times[0])
            self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)

            # if method == 0:
            # Interpolate raw data
            even_times = np.linspace(self.times[0], self.times[-1], L)
            interpolated = np.interp(even_times, self.times, self.samples)
            interpolated = np.hamming(L) * interpolated
            interpolated = interpolated - np.mean(interpolated)

            # Calculate absolute FT and phase of buffer
            raw = np.fft.rfft(interpolated)
            phase = np.angle(raw)
            self.fft = np.abs(raw)

            # Pay attention to frequencies only in feasible heart rate range
            freqs = 60. * self.freqs
            idx = np.where((freqs > 50) & (freqs < 180))

            # Select highest frequency as heart rate
            self.freqs = freqs[idx]
            phase = phase[idx]
            self.fft = self.fft[idx]
            idx2 = np.argmax(self.fft)

            self.bpm = self.freqs[idx2]

            if self.method == 0:
                # Calculate new forehead box frame
                t = (np.sin(phase[idx2]) + 1.) / 2.
                t = 0.9 * t + 0.1
                alpha = t
                beta = 1 - t
                x, y, w, h = self.get_subface_coord(*(self.forehead_coords))
                r = alpha * self.frame_in[y:y + h, x:x + w, 0]
                g = alpha * self.frame_in[y:y + h, x:x + w, 1] + \
                    beta * self.gray[y:y + h, x:x + w]
                b = alpha * self.frame_in[y:y + h, x:x + w, 2]
                frame_out = cv2.merge([r,g,b])

            else:
                fs = 60 
                sos = signal.butter(3, (1, 3.0), 'bp', fs=60, output='sos')
                filtered = signal.sosfilt(sos, subframes, axis=0)
                alpha = 80
                filtered = filtered * alpha
                # Reconstruct Resulting Frame
                filtered = self.rebuildSubframe(filtered, self.levels)
                x, y, w, h = self.get_subface_coord(*(self.forehead_coords))
                frame_out = self.frame_in[y:y + h, x:x + w, :] + filtered


            # Insert new forehead colors
            self.frame_out[y:y + h, x:x + w] = frame_out
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]

            col = (100, 255, 100)
            gap = (self.buffer_size - L) / self.fps
            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)
            cv2.putText(self.frame_out, text,
                       (int(x - w / 2), int(y)), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, col)
