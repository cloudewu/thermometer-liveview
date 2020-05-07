import tkinter
import PIL.Image, PIL.ImageTk
import os
import winsound #only for windows beeping usage
#for linux, please install sox beforehand (sudo apt-get install sox)
import argparse
import cv2

from WebCamera import WebCamera

class App:
    class User_Interface:
        window = None
        canvas = None,
        info = None,
        info_label = None
    
    ui = User_Interface()
    cap = None
    engine = None
    delay = 0
    
    def __init__(self, window, title, model_path, label_path, FPS = 30, source = 0, logging = True, debug = False):
        self.logging = logging
        self.debug = debug
        
        # get camera
        self.cap = WebCamera(source=source, logging=logging)
        if self.logging:
            print("Get camera.")
        
        # get the inference engine (newly add part)
        file_type = os.path.splitext(model_path)[1]
        if file_type == '.xml':
            # openvino's IR model
            import VINOengine
            self.engine = VINOengine.Infer_engine(model_path, label_path, debug=debug)
        elif file_type == '.onnx':
            # ONNX model
            import ONNXengine
            self.engine = ONNXengine.Infer_engine(model_path, label_path, debug=debug)
        else:
            print("[Error] Model not supported")
            return

        # set program
        self.UI_setup(window, title)
        self.delay = 1000 // FPS
        
        if self.logging:
            print("\nProgram Setting")
            print("\tProgram title: " + title)
            print("\tCanvas size: " + str(self.cap.get_size()))
            print("\tFPS: {} / update time: {} ms".format(FPS, self.delay))
        
    def UI_setup(self, window, title):
        self.ui.window = window
        self.ui.window.title(title)
        self.ui.canvas = tkinter.Canvas(self.ui.window, width=self.cap.width, height=self.cap.height)
        self.ui.canvas.pack()
        self.ui.info = tkinter.StringVar()
        self.ui.info.set("Empty")
        self.ui.info_label = tkinter.Label(self.ui.window, textvariable=self.ui.info)
        self.ui.info_label.pack(side="bottom", anchor=tkinter.CENTER)
    
    def run(self):
        if self.ui.window is None:
            print("Window is not created!")
            return
        if self.cap is None:
            print("Web camera is not set!")
            return
        
        # run
        self.update()
        self.ui.window.mainloop()
    
    def update(self):
        ret, frame = self.cap.get_frame()
        human_count = 0
        if ret:
            # inference result
            boxes, scores, classes = self.engine.inference_result(frame)
            im = self.engine.draw_bounding_box(frame, frame.shape[:2], boxes, scores, classes)
            
            '''  a test for beeping when something(people for example) was detected
                person's label =0(in coco_label_2018.txt), 1(in label.txt)'''
            human_count = classes.count(1)
            if 1 in classes:  
                # winsound.Beep(2750, 100) #for windows
                # os.system("play -n synth 0.1 sine 880 vol 0.5 >/dev/null 2>&1")#for linux
                pass

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(im))
            self.ui.canvas.create_image(0, 0, image=self.photo, anchor=tkinter.NW)

            self.ui.info.set("#Human detect: {:3d}".format(human_count))
            self.ui.window.after(self.delay, self.update)
        else:
            print(" [Warning] Fail to get frame") 
            self.videoWriter.release()
        

def parse_args():
    parser = argparse.ArgumentParser(description='This is a program to liveview inference result')
    parser.add_argument('-m', '--model', nargs='?', default='yolov3-tiny.onnx', help='Model used to do inference. Default=yolov3-tiny.onnx')
    parser.add_argument('-l', '--label', nargs='?', default='label.txt', help='Label file used by the model.(one line one class, w/o index number) Default=label.txt')
    parser.add_argument('-d', '--device', nargs=1, type=int, help='Device ID for liveview')
    parser.add_argument('-f', '--file', nargs=1, type=str, help='Video source to run the inference')
    parser.add_argument('--FPS', nargs='?', default=30, type=int, help='Frame per second. Default=30')
    parser.add_argument('--debug', action='store_true', help='Run program in debug mode')
    
    return parser.parse_args()

def main():
    args = parse_args()
    model_name = os.path.basename(args.model).split('.')[0]
    source = args.file if args.device is None else args.device
    if source is None:
        print('Please assign at least one of input device or input file.')
        return
    
    app = App(tkinter.Tk(), "Inference liveview ({})".format(model_name), args.model, args.label, FPS=args.FPS, source=source[0], debug=args.debug)
    app.run()

    ### Sample command to run this program ###
    """ test yolo-v3 (ONNX) """
    # app = App(tkinter.Tk(), "Inference liveview (yolov3)", 'yolov3.onnx', 'label.txt', FPS=10)
    """ test tiny yolo-v3 (ONNX) """
    # app = App(tkinter.Tk(), "Inference liveview (tiny yolov3)", 'yolov3-tiny.onnx', 'label.txt', FPS=30)
    """ test ssd-mobilenet-v2 (openVINO) """
    # app = App(tkinter.Tk(), "Inference liveview (ssd-mobilenet)", 'ssd_mobilenet_v2_coco/frozen_inference_graph.xml', 'coco_label_2018.txt', FPS=30)
    """ test face-detection-retail-0005 (openVINO) """
    # app = App(tkinter.Tk(), "Inference liveview (retail-0005)", 'face-detection-retail-0005/FP16/face-detection-retail-0005.xml', 'coco_label_2018.txt', FPS=30, source=0)
    

if __name__=='__main__':
    main()
