from PIL import Image
from openvino.inference_engine import IECore
import numpy as np
import cv2
import os

class Infer_engine():
    debug = False
    network = None
    session = None
    input_names = []
    output_names = []
    def __init__(self, model_path, label_file, debug=False):
        print("Initializing inference engine...")

        self.debug = debug
        weight_path = model_path.split('.')[0] + '.bin'
        print("\tModel: {}\n\tWeights: {}\n\tLabel file: {}".format(model_path, weight_path, label_file))
        if not (os.path.isfile(model_path) or os.path.isfile(weight_path) or os.path.isfile(label_file)):
            print(" [Error] File is not found. Engine is not loaded.")
            return
        
        engine = IECore()
        self.network = engine.read_network(model=model_path, weights=weight_path)
        self.input_names = list(self.network.inputs.keys())
        self.output_names = list(self.network.outputs.keys())
        self.session = engine.load_network(network=self.network, device_name='CPU')
        print("\nNetwork loaded.")
        print("Input layers:")
        for name in self.input_names:
            print("\t"+name)
        print("Output layers:")
        for name in self.output_names:
            print("\t"+name)

        with open(label_file) as f:
            self.labels = f.read().splitlines()

    def pre_process(self, input_data, input_size=(300, 300)):
        image_data = cv2.resize(input_data, input_size)
        image_data = np.expand_dims(image_data, 0)
        return image_data

    def inference(self, image):
        image_data = self.pre_process(image, input_size=(300, 300))
        input_data = {
            self.input_names[0]: image_data
        }
        return self.session.infer(inputs=input_data)

    def inference_result(self, img):
        result = self.inference(img) 
        return self.post_process(result[self.output_names[0]])       

    def draw_bounding_box(self, image, imsize, boxes, scores, classes):
        return image

    def post_process(self, result):
        out_boxes, out_scores, out_classes = [], [], []
        return out_boxes, out_scores, out_classes
