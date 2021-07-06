# pyglFaceBlur

OpenGL, Mediapipe, opevcv video/webcam face writing
## Installation

Tested running on win10 with python 3.8 x64

```shell
$ pip install git+https://github.com/philipNoonan/pyglFaceBlur.git
```

## Using pyglFaceBlur

If using a system with only one webcam, select cam : 0 from the dropdown list.
If using videos, you need to create a data directory

```
mkdir data
```

Place any video you want to play with into the /data folder.

```
$ pyglFaceBlur
```

This opens a black window, with a menu allowing you to choose between Use Camera, and Select Video File. Either option prompts a new drop down list of camera IDs (click to choose 0 for default) or available video files found in the /data directory.
The sliders change which frame is rendered in either the R,G,B channels in the middle window. The 'run filter' option runs a simple gradient compute shader over the latest frame from the chosen video source.
