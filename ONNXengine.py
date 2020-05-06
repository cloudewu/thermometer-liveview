from PIL import Image
import onnx
import onnxruntime
import numpy as np
import cv2

class Infer_engine():
    debug = False
    isTiny = False
    def __init__(self, model_path, label_file, debug=False):
        print("Initializing inference engine...")
        print("\tModel: {}\n\tLabel file: {}".format(model_path, label_file))

        self.debug = debug
        self.isTiny = True if (model_path.lower().find('tiny') >= 0) else False
        self.session = onnxruntime.InferenceSession(model_path, None)
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
        image_data = np.array(self.resize_padding(input_data, input_size[0], (128, 128, 128)), dtype='float32')
        image_data /= 255    # normalize
        image_data = np.transpose(image_data, [2, 0, 1])
        image_data = np.expand_dims(image_data, 0)
        return image_data

    def inference(self, image):
        image_data = self.pre_process(Image.fromarray(image))
        img_size = np.array([image.shape[0], image.shape[1]], dtype=np.float32).reshape(1, 2)
        input_data = {
            'input_1': image_data,
            'image_shape': img_size
        }
        return self.session.run(None, input_data)

    def inference_result(self, img):
        boxes, scores, indice = self.inference(img) 
        if self.isTiny:
            indice = indice[0]
        return self.post_process( boxes, scores, indice)       

    def draw_bounding_box(self, image, imsize, boxes, scores, classes):
        # iw, ih = imsize
        number = len(classes)
        for i in range(number):
            label = classes[i]
            score = scores[i]
            color = (0, 180, 116)
            # xmin = int(iw * boxes[i][0])
            # ymin = int(ih * boxes[i][1])
            # xmax = int(iw * boxes[i][2])
            # ymax = int(ih * boxes[i][3])
            ymin = int(boxes[i][0])
            xmin = int(boxes[i][1])
            ymax = int(boxes[i][2])
            xmax = int(boxes[i][3])
            if label ==0: #should be changed to the real comparison criteron
                color = (255, 31, 37)
            x, y = xmin, ymin-5
            if y <=5: x, y = xmin+2, ymin+15
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image,self.labels[label]+' {:.4f}'.format(score), (x,y), cv2.FONT_HERSHEY_PLAIN, 1, color, 1, cv2.LINE_AA)
        return image

    def post_process(self, boxes, scores, indice):
        out_boxes, out_scores, out_classes = [], [], []
        
        for batch_id, class_id, box_id in indice:
            out_classes.append(class_id)
            out_scores.append(scores[batch_id][class_id][box_id])
            out_boxes.append(boxes[batch_id][box_id])
        
        if self.debug:
            print("\nDetect {} objects.".format(indice.shape[0]))
            for i in range(indice.shape[0]):
                print("#{}: {} ({})".format(i, self.labels[out_classes[i]], out_scores[i]))
                print("\t" + str(out_boxes[i]))
                
        return out_boxes, out_scores, out_classes
