![Alt text](http://i.imgur.com/2ngZopS.jpg "Screenshot")

webcam-heart
-----------------------

This code is adapted from https://github.com/thearn/webcam-pulse-detector, (we replaced the Mayer Wave visualization with Eulerian magnification visualization, added cheeks as a possible measuring area, and implemented some user customization of bpm calculation)

This code can be run from source by following the instructions below.

---------------------------------------

A python code that detects the heart-rate of an individual using a common webcam or network IP camera. 
Tested on OSX, Ubuntu, and Windows.

How it works:
-----------------
This application uses [OpenCV](http://opencv.org/) to find the location of the user's face, then isolate the forehead or cheeks. Data is collected
from this location over time to estimate the user's heart rate. This is done by measuring average optical
intensity in the forehead location, in the subimage's green channel alone (a better color mixing ratio may exist, but the 
blue channel tends to be very noisy). Physiological data can be estimated this way thanks to the optical absorption 
characteristics of (oxy-) haemoglobin (see http://www.opticsinfobase.org/oe/abstract.cfm?uri=oe-16-26-21434). 

With good lighting and minimal noise due to motion, a stable heartbeat should be 
isolated in about 15 seconds. 

Once the user's heart rate has been estimated, real-time phase variation associated with this 
frequency is also computed. This allows for the heartbeat to be exaggerated in the post-process frame rendering, 
causing the highlighted forehead location to pulse in sync with the user's own heartbeat.

Support for detection on multiple simultaneous individuals in a single camera's 
image stream is definitely possible, but at the moment only the information from one face 
is extracted for analysis.

The overall dataflow/execution order for the real-time signal processing looks like:

![Alt text](http://i.imgur.com/xS7O8U3.png "Signal processing")


Requirements:
---------------

- [Python v2.7 or v3.5+)](http://python.org/)
- [OpenCV v2+](http://opencv.org/)
- Numpy, Scipy

Quickstart:
------------

- run get_pulse.py to start the application

```
python get_pulse.py
```

- To run on an IP camera, set the `url`, `user`, and `password` strings on line 134 of `get_pulse_ipcam.py`, then run:

```
python get_pulse_ipcam.py
```
This was tested on a Wowwee Rovio.

- If there is an error, try running `test_webcam.py` in the same directory to check if your openCV installation and webcam can be made to work
with this application.

Usage notes:
----------
- When run, a window will open showing a stream from your computer's webcam
- When a forehead location has been isolated, the user should press "S" on their 
keyboard to lock this location, and remain as still as possible (the camera 
stream window must have focus for the click to register). This freezes the acquisition location in place. This lock can
be released by pressing "S" again.
- To view a stream of the measured data as it is gathered, press "D". To hide this display, press "D" again.
- The data display shows three data traces, from top to bottom: 
   1. raw optical intensity
   2. extracted heartbeat signal
   3. Power spectral density, with local maxima indicating the heartrate (in beats per minute).
- To change the color channel used to calculate BPM, press "X". The possible options are:
   1. Green channel only
   2. Red channel only
   3. Blue channel only
   4. Red, Blue, and Green channels
- To change the measuring area for BPM calculation and blood flow visualization, press "Z". The possible options are:
   1. Forehead
   2. Left and right cheeks
   3. Forehead and both cheeks
- With consistent lighting and minimal head motion, a stable heartbeat should be 
isolated in about 15 to 20 seconds. A count-down is shown in the image frame.
- If a large spike in optical intensity is measured in the data (due to motion 
noise, sudden change in lighting, etc) the data collection process is reset and 
started over. The sensitivity of this feature can be tweaked by changing `data_spike_limit` on line 31 of [get_pulse.py](get_pulse.py).
Other mutable parameters of the analysis can be changed here as well.
