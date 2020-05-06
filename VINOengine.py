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

    ''' resize image with aspect ratio and add padding to square '''
    def resize_padding(self, image, target_width, color=(255, 255, 255)):
        new_image = Image.new('RGB', (target_width, target_width), color)

        w, h = image.size
        padding = [0, 0]
        if w > h:
            ratio = target_width / w
            new_h = int(h * ratio)
            small_image = image.resize((target_width, new_h))
            padding[1] = (target_width - new_h) // 2
        else:
            ratio = target_width / h
            new_w = int(w * ratio)
            small_image = image.resize((new_w, target_width))
            padding[0] = (target_width - new_w) // 2
        new_image.paste(small_image, tuple(padding))
        return new_image

    def pre_process(self, input_data, input_size=(416, 416)):
        image_data = cv2.cvtColor(input_data, cv2.COLOR_RGB2BGR)
        image_data = cv2.resize(image_data, input_size)
        image_data = np.transpose(image_data, [2, 0, 1])   # input format [channel, height, width]
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
        ih, iw = imsize
        number = len(classes)
        for i in range(number):
            label = classes[i]
            score = scores[i]
            color = (0, 180, 116)
            xmin = int(iw * boxes[i][0])
            ymin = int(ih * boxes[i][1])
            xmax = int(iw * boxes[i][2])
            ymax = int(ih * boxes[i][3])
            x, y = xmin, ymin-5
            if y <=5: x, y = xmin+2, ymin+15
            if label ==1: #should be changed to the real comparison criteron
                color = (255, 31, 37)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image,self.labels[label]+' {:.4f}'.format(score), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
        return image

    def post_process(self, result):
        result = result[0][0]
        out_boxes, out_scores, out_classes = [], [], []

        for idx, data in enumerate(result):
            if data[2] > 0.5:
                out_classes.append(np.int(data[1]))
                out_scores.append(data[2])
                out_boxes.append(data[3:7])
        
        if self.debug:
            print("\nDetect {} objects.".format(len(out_classes)))
            for i in range(len(out_classes)):
                print("#{}: {} ({})".format(i, self.labels[out_classes[i]], out_scores[i]))
                print("\t" + str(out_boxes[i]))

        return out_boxes, out_scores, out_classes
