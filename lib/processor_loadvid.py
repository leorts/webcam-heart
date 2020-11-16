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
        self.g_pyr_forehead_buffer = []
        self.g_pyr_l_cheek_buffer = []
        self.g_pyr_r_cheek_buffer = []
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
        self.forehead_coords = (0.5, 0.17, 0.3, 0.18) # (right, down, wide, tall)
        self.l_cheek_coords = (0.75, 0.6, 0.15, 0.2) 
        self.r_cheek_coords = (0.25, 0.6, 0.15, 0.2) 
        self.last_center = np.array([0, 0])
        self.find_faces = True 

        self.used_color_ch = 0 # default color channel used is g
        self.used_color_chs = ["g","r","b","rgb"]
        self.measuring_area = 0

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

    def color_ch_toggle(self):
        self.used_color_ch+=1
        self.used_color_ch = 0 if self.used_color_ch == 4 else self.used_color_ch
        used_color_ch = self.used_color_chs[self.used_color_ch]
        return used_color_ch

    def get_color_ch(self):
        used_color_ch = self.used_color_chs[self.used_color_ch]  
        return used_color_ch

    def measuring_area_toggle(self):
        if self.measuring_area == 0:
            self.measuring_area = 1
            measuring_area = "Cheeks"
        elif self.measuring_area == 1:
            self.measuring_area = 2
            measuring_area = "Forehead and cheeks"
        else: # measuring_area == 2
            self.measuring_area = 0
            measuring_area = "Forehead"     
        return measuring_area

    def get_measuring_area(self):
        if self.measuring_area == 0:
            measuring_area = "Forehead"     
        elif self.measuring_area == 1:
            measuring_area = "Cheeks"
        else: # measuring_area == 2
            measuring_area = "Forehead and Cheeks"
        return measuring_area


    #-------------------------------------------------------------#     
    # BPM AND EULERIAN MAGNIFICATION METHODS
    #-------------------------------------------------------------#  
    
    # Helper Methods for get_bpm()
    def extract_color(self, frame):
        if self.used_color_ch == 0: # g
            return np.mean(frame[:,:,1])
        elif self.used_color_ch == 1: # r  
            return np.mean(frame[:,:,0])
        elif self.used_color_ch == 2: # b
            return np.mean(frame[:,:,2])
        else: # rgb
            return np.mean(frame)

    def get_bpm(self, L):
        # Interpolate raw data
        even_times = np.linspace(self.times[0], self.times[-1], L)
        self.samples = signal.detrend(self.samples) # Detrend mitigates light change interference 
        interpolated = np.interp(even_times, self.times, self.samples)
        interpolated = np.hamming(L) * interpolated
        interpolated = interpolated - np.mean(interpolated)

        # Calculate absolute FT of buffer
        raw = np.fft.rfft(interpolated)
        self.fft = np.abs(raw)

        # Pay attention to frequencies only in feasible heart rate range
        self.freqs = float(self.fps) / L * np.arange(L / 2 + 1)
        freqs = 60. * self.freqs
        idx = np.where((freqs > 50) & (freqs < 180))

        # Select highest frequency as heart rate
        self.freqs = freqs[idx]
        self.fft = self.fft[idx]
        idx2 = np.argmax(self.fft)

        return self.freqs[idx2]

    # Helper Methods for eulerian_magnify()
    def build_g_pyr(self, subframe, levels=3):
        pyr = [subframe]
        for level in range(levels):
            subframe = cv2.pyrDown(subframe)
            pyr.append(subframe)
        return pyr

    def temporal_ideal_filter(self, pyr_vid):
        fft = np.fft.fft(pyr_vid, axis=0)
        freqs = np.fft.fftfreq(pyr_vid.shape[0], d=(1.0 / self.fps)) * 60
        idx_low = (np.abs(freqs - 50)).argmin()
        idx_high = (np.abs(freqs - 180)).argmin()
        fft[:idx_low ] = 0
        fft[idx_high:-idx_high] = 0
        fft[-idx_low:] = 0
        return np.abs(np.fft.ifft(fft, axis=0))

    def rebuild_subframe(self, pyr, w, h, levels=3):
        for level in range(levels):
            pyr = cv2.pyrUp(pyr)
        subframe = pyr[:h, :w] # Account for possible pyrDown and pyrUp rounding
        return subframe    

    def eulerian_magnify(self, pyr_vid, w, h, amplification=50):
        t_filt_pyr_vid = self.temporal_ideal_filter(pyr_vid)  
        t_filt_pyr = t_filt_pyr_vid[-1] # Take last frame of filtered video
        t_filt_pyr = t_filt_pyr * amplification
        subframe_out = self.rebuild_subframe(t_filt_pyr, w, h)
        return subframe_out

    def add_magnification(self, m_subframe, x, y, w, h):
        self.frame_out[y:y + h, x:x + w] = self.frame_in[y:y + h, x:x + w, :] + m_subframe   

    #-------------------------------------------------------------#      
    # MAIN LOOP
    #-------------------------------------------------------------#    

    def run(self):
        self.times.append(time.time() - self.t0)
        self.frame_out = self.frame_in
        self.gray = cv2.equalizeHist(cv2.cvtColor(self.frame_in, cv2.COLOR_BGR2GRAY))
        col = (100, 255, 100)
        if self.find_faces:
            cv2.putText(
                self.frame_out, "Press 'S' to lock face and begin",
                       (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'X' to change used color channel(s) (current: %s)" % self.get_color_ch(),
                        (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
            cv2.putText(
                self.frame_out, "Press 'Z' to change measuring area (current: %s)" % self.get_measuring_area(),
                        (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)            
            cv2.putText(self.frame_out, "Press 'Esc' to quit",
                       (10, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)
            self.mean_buffer, self.g_pyr_buffer, self.times = [], [], []
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
            self.draw_rect(self.face_rect, col=(96, 94, 211))
            x, y, w, h = self.face_rect
            col2 = (0, 94, 184)
            cv2.putText(self.frame_out, "Face",
                       (x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)

            if self.measuring_area == 0: # forehead
                forehead = self.get_subface_coord(*(self.forehead_coords))           
                self.draw_rect(forehead)
                x, y, w, h = forehead
                cv2.putText(self.frame_out, "Forehead",(x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)
            elif self.measuring_area == 1: # cheeks
                l_cheek = self.get_subface_coord(*(self.l_cheek_coords))           
                self.draw_rect(l_cheek)
                x, y, w, h = l_cheek
                cv2.putText(self.frame_out, "L. Cheek",(x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)

                r_cheek = self.get_subface_coord(*(self.r_cheek_coords))           
                self.draw_rect(r_cheek)
                x, y, w, h = r_cheek
                cv2.putText(self.frame_out, "R. Cheek",(x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)
            else: # forehead and cheeks
                forehead = self.get_subface_coord(*(self.forehead_coords))           
                self.draw_rect(forehead)
                x, y, w, h = forehead
                cv2.putText(self.frame_out, "Forehead",(x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)
                
                l_cheek = self.get_subface_coord(*(self.l_cheek_coords))           
                self.draw_rect(l_cheek)
                x, y, w, h = l_cheek
                cv2.putText(self.frame_out, "L. Cheek",(x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)

                r_cheek = self.get_subface_coord(*(self.r_cheek_coords))           
                self.draw_rect(r_cheek)
                x, y, w, h = r_cheek
                cv2.putText(self.frame_out, "R. Cheek",(x, y), cv2.FONT_HERSHEY_PLAIN , 1.5, col2)

            return
        if set(self.face_rect) == set([1, 1, 2, 2]):
            return
        cv2.putText(self.frame_out, "Press 'S' to restart",
                   (10, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, col)
        cv2.putText(self.frame_out, "Press 'X' to change used color channel(s) (current: %s)" % self.get_color_ch(),
                   (10, 75), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.25, col)           
        cv2.putText(self.frame_out, "Press 'D' to toggle data plot",
                   (10, 100), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, col)
        cv2.putText(self.frame_out, "Press 'Esc' to quit",
                   (10, 125), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1.5, col)

        forehead = self.get_subface_coord(*(self.forehead_coords))
        xf, yf, wf, hf = forehead
        forehead_subframe = self.frame_in[yf:yf + hf, xf:xf + wf, :]

        l_cheek = self.get_subface_coord(*(self.l_cheek_coords))
        xl, yl, wl, hl = l_cheek
        l_cheek_subframe = self.frame_in[yl:yl + hl, xl:xl + wl, :]

        r_cheek = self.get_subface_coord(*(self.r_cheek_coords))
        xr, yr, wr, hr = r_cheek
        r_cheek_subframe = self.frame_in[yr:yr + hr, xr:xr + wr, :]

        forehead_mean = self.extract_color(forehead_subframe)
        l_cheek_mean = self.extract_color(l_cheek_subframe)
        r_cheek_mean = self.extract_color(r_cheek_subframe)

        if self.measuring_area == 0: # forehead
            self.draw_rect(forehead)
            mean = forehead_mean
            g_pyr_forehead = self.build_g_pyr(forehead_subframe)
            self.g_pyr_forehead_buffer.append(g_pyr_forehead[-1])
        elif self.measuring_area == 1: # cheeks
            self.draw_rect(l_cheek)
            self.draw_rect(r_cheek)
            mean = (l_cheek_mean + r_cheek_mean) / 2
            g_pyr_l_cheek = self.build_g_pyr(l_cheek_subframe)
            g_pyr_r_cheek = self.build_g_pyr(r_cheek_subframe)
            self.g_pyr_l_cheek_buffer.append(g_pyr_l_cheek[-1])
            self.g_pyr_r_cheek_buffer.append(g_pyr_r_cheek[-1])
        else: # forehead and cheeks
            self.draw_rect(forehead)
            self.draw_rect(l_cheek)
            self.draw_rect(r_cheek)
            mean = (forehead_mean + l_cheek_mean + r_cheek_mean) / 3 
            g_pyr_forehead = self.build_g_pyr(forehead_subframe)
            g_pyr_l_cheek = self.build_g_pyr(l_cheek_subframe)
            g_pyr_r_cheek = self.build_g_pyr(r_cheek_subframe)
            self.g_pyr_forehead_buffer.append(g_pyr_forehead[-1])
            self.g_pyr_l_cheek_buffer.append(g_pyr_l_cheek[-1])
            self.g_pyr_r_cheek_buffer.append(g_pyr_r_cheek[-1])

        self.mean_buffer.append(mean)
        
        L = len(self.mean_buffer)
        if L > self.buffer_size:
            self.times = self.times[-self.buffer_size:]
            self.mean_buffer = self.mean_buffer[-self.buffer_size:]
            self.g_pyr_forehead_buffer = self.g_pyr_forehead_buffer[-self.buffer_size:]
            self.g_pyr_l_cheek_buffer = self.g_pyr_l_cheek_buffer[-self.buffer_size:]
            self.g_pyr_r_cheek_buffer = self.g_pyr_r_cheek_buffer[-self.buffer_size:]
            L = self.buffer_size

        self.samples = np.array(self.mean_buffer)
        pyr_forehead_vid = np.array(self.g_pyr_forehead_buffer, copy = True)
        pyr_l_cheek_vid = np.array(self.g_pyr_l_cheek_buffer, copy = True)
        pyr_r_cheek_vid = np.array(self.g_pyr_r_cheek_buffer, copy = True)
        if L > 50: 
            self.fps = float(L) / (self.times[-1] - self.times[0])
            self.bpm = self.get_bpm(L)
            if self.measuring_area == 0: # forehead
                m_forehead = self.eulerian_magnify(pyr_forehead_vid, wf, hf)
                self.add_magnification(m_forehead, xf, yf, wf, hf)
            elif self.measuring_area == 1: # cheeks
                m_l_cheek = self.eulerian_magnify(pyr_l_cheek_vid, wl, hl)
                m_r_cheek = self.eulerian_magnify(pyr_r_cheek_vid, wr, hr)
                self.add_magnification(m_l_cheek, xl, yl, wl, hl)
                self.add_magnification(m_r_cheek, xr, yr, wr, hr)
            else: # forehead and cheeks
                m_forehead = self.eulerian_magnify(pyr_forehead_vid, wf, hf)
                m_l_cheek = self.eulerian_magnify(pyr_l_cheek_vid, wl, hl)
                m_r_cheek = self.eulerian_magnify(pyr_r_cheek_vid, wr, hr)
                self.add_magnification(m_forehead, xf, yf, wf, hf)
                self.add_magnification(m_l_cheek, xl, yl, wl, hl)
                self.add_magnification(m_r_cheek, xr, yr, wr, hr)
            
            x1, y1, w1, h1 = self.face_rect
            self.slices = [np.copy(self.frame_out[y1:y1 + h1, x1:x1 + w1, 1])]

            col = (100, 255, 100)
            gap = (self.buffer_size - L) / self.fps
            if gap:
                text = "(estimate: %0.1f bpm, wait %0.0f s)" % (self.bpm, gap)
            else:
                text = "(estimate: %0.1f bpm)" % (self.bpm)
            cv2.putText(self.frame_out, text,
                       (10, 150), cv2.FONT_HERSHEY_COMPLEX_SMALL , 1, col)

        

