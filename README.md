README
==========================

Usage
----------------------
```
usage: liveview.py [-h] [-m [MODEL]] [-l [LABEL]] [-d DEVICE] [-f FILE] [--FPS [FPS]] [--debug]

optional arguments:
  -h, --help            show this help message and exit
  -m [MODEL], --model [MODEL]
                        Model used to do inference. Default=yolov3-tiny.onnx
  -l [LABEL], --label [LABEL]
                        Label file used by the model.(one line one class, w/o index number) Default=label.txt
  -d DEVICE, --device DEVICE
                        Device ID for liveview
  -f FILE, --file FILE  Video source to run the inference
  --FPS [FPS]           Frame per second. Default=30
  --debug               Run program in debug mode
```
Note that at least one of DEVICE and FILE should be assigned.  


Model Supported
----------------------
**ONNX engine**
 + [yolov3](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/yolov3)
 + [yolov3-tiny](https://github.com/onnx/models/tree/master/vision/object_detection_segmentation/tiny-yolov3)

**open VINO engine**
You can download models through openVINO Toolkit's model downloader  
 + All SSD models (ssd-mobilenet-v2, face-detection-retail-0005)


Dependencies
----------------------
 + pillow
 + cv2
 + (for ONNX models) onnx, onnxruntime
 + (for openVINO IR models) openvino
 + (for linux) sox


Trouble shooting
----------------------
1. `Unable to open video source` or `Fail to get frame`:  
   Please check if your device or filepath is correct  
2. `No module named 'openvino'`:  
   Install [openVINO toolkit](https://docs.openvinotoolkit.org/latest/index.html), and make sure you've set up environment variables.(`$VINO_DIR\bin\setupvars.bat`)  
3. `File is not found. Engine is not loaded.`:  
   Check model or label file path. Onnx model should be something like `*.onnx`, and openVINO model should be like `*.xml` (with `*.bin` in the same folder)  