import tkinter
import PIL.Image, PIL.ImageTk
import os
import winsound #only for windows beeping usage
#for linux, please install sox beforehand (sudo apt-get install sox)
from WebCamera import WebCamera

class App:
    app = {
        'window': None,
        'canvas': None,
        'info': None,
        'info_label': None
    }
    cap = None
    engine = None
    delay = 0
    
    def __init__(self, window, title, model_path, label_path, FPS = 30, device = 0, logging = True):
        self.logging = logging

        # get camera
        self.cap = WebCamera(device=device, logging=logging)
        if self.logging:
            print("Get camera.")
        
        # get the inference engine (newly add part)
        file_type = os.path.splitext(model_path)[1]
        if file_type == '.xml':
            # openvino's IR model
            import VINOengine
            self.engine = VINOengine.Infer_engine(model_path, label_path, debug=False)
        elif file_type == '.onnx':
            # ONNX model
            import ONNXengine
            self.engine = ONNXengine.Infer_engine(model_path, label_path, debug=False)
        else:
            print("[Error] Model not supported")
            return

        # set program
        self.app['window'] = window
        self.app['window'].title(title)
        self.app['canvas'] = tkinter.Canvas(self.app['window'], width=self.cap.width, height=self.cap.height)
        self.app['canvas'].pack()
        self.app['info'] = tkinter.StringVar()
        self.app['info'].set("Empty")
        self.app['info_label'] = tkinter.Label(self.app['window'], textvariable=self.app['info'])
        self.app['info_label'].pack(side="bottom", anchor=tkinter.CENTER)

        self.delay = 1000 // FPS
        
        if self.logging:
            print("\nProgram Setting")
            print("\tProgram title: " + title)
            print("\tCanvas size: " + str(self.cap.get_size()))
            print("\tFPS: {} / update time: {} ms".format(FPS, self.delay))
    
    def run(self):
        if self.app['window'] is None:
            print("Window is not created!")
            return
        if self.cap is None:
            print("Web camera is not set!")
            return
        
        # run
        self.update()
        self.app['window'].mainloop()
    
    def update(self):
        ret, frame = self.cap.get_frame()
        if ret:
            # inference result
            boxes, scores, classes = self.engine.inference_result(frame)
            im = self.engine.draw_bounding_box(frame, frame.shape[:2], boxes, scores, classes)
            
            '''  a test for beeping when something(people for example) was detected
                person's label =0(in coco_label_2018.txt), 1(in label.txt)'''
            human_count = classes.count(0)
            if 0 in classes:  
                # winsound.Beep(2750, 100) #for windows
                #os.system("play -n synth 0.1 sine 880 vol 0.5 >/dev/null 2>&1")#for linux
                pass

            self.photo = PIL.ImageTk.PhotoImage(image=PIL.Image.fromarray(im))
            self.app['canvas'].create_image(0, 0, image=self.photo, anchor=tkinter.NW)
        else:
            print(" [Warning] Fail to get frame") 

        self.app['info'].set("#Human detect: {:3d}".format(human_count))
        self.app['window'].after(self.delay, self.update)


def main():
    """ test yolo-v3 (ONNX) """
    # app = App(tkinter.Tk(), "Inference liveview (yolov3)", 'yolov3.onnx', 'label.txt', FPS=10)
    """ test tiny yolo-v3 (ONNX) """
    app = App(tkinter.Tk(), "Inference liveview (tiny yolov3)", 'yolov3-tiny.onnx', 'label.txt', FPS=30)
    """ test ssd-mobilenet-v2 (openVINO) """
    # app = App(tkinter.Tk(), "Inference liveview (ssd-mobilenet)", 'ssd_mobilenet_v2_coco/frozen_inference_graph.xml', 'coco_label_2018.txt', FPS=30)
    """ test face-detection-retail-0005 (openVINO) """
    # app = App(tkinter.Tk(), "Inference liveview (retail-0005)", 'face-detection-retail-0005/FP16/face-detection-retail-0005.xml', 'coco_label_2018.txt', FPS=10)
    
    app.run()

if __name__=='__main__':
    main()
