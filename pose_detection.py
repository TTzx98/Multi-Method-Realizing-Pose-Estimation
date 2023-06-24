import onnxruntime as ort
import cv2
import numpy as np
import argparse

class PoseDet():
    def __init__(self, model):

        # self.classes = list(map(lambda x: x.strip(), open('label.txt', 'r').readlines()))

        if model.startswith("DeepPose"):
            self.BlazeFlag = False
            self.inpWidth = 196
            self.inpHeight = 196
        elif model.startswith("ResNet"):
            self.BlazeFlag = False
            if model.endswith("128.onnx"):
                self.inpWidth = 128
                self.inpHeight = 128
            elif model.endswith("196.onnx"):
                self.inpWidth = 196
                self.inpHeight = 196
        else:
            self.BlazeFlag = True
            self.inpWidth = 256
            self.inpHeight = 256

        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.net = ort.InferenceSession(model, so, providers=['CPUExecutionProvider']) # CUDAExecutionProvider
        self.keep_ratio = False
        self.swaprgb = False

        self.COLORS = np.random.randint(0, 255, size=(200, 3), dtype="uint8")
        self.line_width = 1

    def resize_image(self, srcimg):
        top, left, newh, neww = 0, 0, self.inpHeight, self.inpWidth
        if self.keep_ratio and srcimg.shape[0] != srcimg.shape[1]:
            hw_scale = srcimg.shape[0] / srcimg.shape[1]
            if hw_scale > 1:
                newh, neww = self.inpHeight, int(self.inpWidth / hw_scale)
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                left = int((self.inpWidth - neww) * 0.5)
                img = cv2.copyMakeBorder(img, 0, 0, left, self.inpWidth - neww - left, cv2.BORDER_CONSTANT,
                                         value=0)  # add border
            else:
                newh, neww = int(self.inpHeight * hw_scale), self.inpWidth
                img = cv2.resize(srcimg, (neww, newh), interpolation=cv2.INTER_AREA)
                top = int((self.inpHeight - newh) * 0.5)
                img = cv2.copyMakeBorder(img, top, self.inpHeight - newh - top, 0, 0, cv2.BORDER_CONSTANT, value=0)
        else:
            img = cv2.resize(srcimg, (self.inpWidth, self.inpHeight), interpolation=cv2.INTER_AREA)
        return img, newh, neww, top, left

    def detect(self, frame):
        srcimg = frame.copy()
        if self.swaprgb:
            srcimg = cv2.cvtColor(srcimg, cv2.COLOR_BGR2RGB)
        img, newh, neww, top, left = self.resize_image(srcimg)
        # print(img.shape)
        if self.BlazeFlag:
            blob = np.expand_dims(np.transpose(img, (0, 1, 2)), axis=0).astype(np.float32) / 255.0
        else:
            blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0).astype(np.float32) / 255.0
        # print(blob.shape)
        outs = self.net.run(None, {self.net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

        srcHeight, srcWidth, _ = srcimg.shape
        if self.BlazeFlag:
            for j in (0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28): # 33
                point = (int(outs[5*j] / self.inpWidth * srcWidth), int(outs[5*j + 1] / self.inpHeight * srcHeight))
                # print(point)
                cv2.circle(frame, point, radius=0, color=(0, 0, 255), thickness=2)
            for j in ((11, 13), (13, 15), (12, 14), (14, 16), (11, 12), (23, 25), (25, 27), (24, 26), (26, 28), (11, 23), (23, 24), (12, 24)): # 11 17
                point1 = (int(outs[5*j[0]] / self.inpWidth * srcWidth), int(outs[5*j[0] + 1] / self.inpHeight * srcHeight))
                point2 = (int(outs[5*j[1]] / self.inpWidth * srcWidth), int(outs[5*j[1] + 1] / self.inpHeight * srcHeight))
                color = [int(c) for c in self.COLORS[j[1] % len(self.COLORS)]]
                cv2.line(frame, point1, point2, color, thickness=self.line_width)
            neckpoint = (int((outs[5*11] + outs[5*12]) / 2 / self.inpWidth * srcWidth), int((outs[5*11+1] + outs[5*12+1]) / 2 / self.inpHeight * srcHeight))
            nosepoint = (int(outs[5*0] / self.inpWidth * srcWidth), int(outs[5*0 + 1] / self.inpHeight * srcHeight))
            cv2.circle(frame, neckpoint, radius=0, color=(0, 0, 255), thickness=2)
            cv2.line(frame, neckpoint, nosepoint, color, thickness=self.line_width)

        else:
            for j in range(14):
                point = (int(outs[2*j] * srcWidth), int(outs[2*j + 1] * srcHeight))
                cv2.circle(frame, point, radius=0, color=(0, 0, 255), thickness=2)
            for j in ((13, 12), (12, 8), (12, 9), (8, 7), (7, 6), (9, 10), (10, 11), (2, 3), (2, 1), (1, 0), (3, 4), (4, 5), (2, 8), (3, 9)):
                point1 = (int(outs[2*j[0]] * srcWidth), int(outs[2*j[0] + 1] * srcHeight))
                point2 = (int(outs[2*j[1]] * srcWidth), int(outs[2*j[1] + 1] * srcHeight))
                color = [int(c) for c in self.COLORS[j[1] % len(self.COLORS)]]
                cv2.line(frame, point1, point2, color, thickness=self.line_width)

        return frame

    def change_line_width(self, height, width):
        if height < 256 and width < 256:
            self.line_width = 1
        elif height < 512 and width < 512:
            self.line_width = 2
        elif height < 1024 and width < 1024:
            self.line_width = 3
        else:
            self.line_width = 4


if __name__=='__main__':
    net = PoseDet('pose_landmark_lite.onnx')
    srcimg = cv2.imread('dataset\lsp\images\im0002.jpg')
    srcimg = net.detect(srcimg)

    winName = 'pose detection in ONNXRuntime'
    cv2.namedWindow(winName, cv2.WINDOW_NORMAL)
    cv2.imwrite("out.jpg", srcimg)
    cv2.imshow(winName, srcimg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()