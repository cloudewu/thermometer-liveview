import cv2

class WebCamera:
    cap = None
    width = 0
    height = 0
    def __init__(self, source=0, logging=True):
        if logging:
            print("Initializing web camera...")
        
        self.cap = cv2.VideoCapture(source)
        if not self.cap.isOpened():
            raise ValueError("Unable to open video source", source)
        self.width, self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def get_size(self):
        return (self.width, self.height)
    
    def get_frame(self):
        ret = True
        frame = None
        if not self.cap.isOpened():
            ret = False
        else:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        return (ret, frame)
    
    def __del__(self):
        if self.cap.isOpened():
            self.cap.release()