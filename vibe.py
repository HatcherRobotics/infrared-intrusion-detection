import cv2 as cv
import numpy as np
import random
num_sam, min_match, r, rand_sam = 20, 2, 20, 16
c_y_off = c_x_off = c_off = [-1, 0, 1, -1, 1, -1, 0, 1, 0]


class ViBe:
    def init(self, img):
        self.samples = np.zeros([img.shape[0], img.shape[1], num_sam + 1])
        self.FGModel = np.zeros(img.shape)

    def process_first_frame(self, img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                for k in range(num_sam):
                    random_num = int(random.uniform(0, 9))
                    row = i + c_x_off[random_num]
                    random_num = int(random.uniform(0, 9))
                    col = j + c_y_off[random_num]
                    if row < 0:
                        row = 0
                    if row >= img.shape[0]:
                        row = img.shape[0] - 1
                    if col < 0:
                        col = 0
                    if col >= img.shape[1]:
                        col = img.shape[1] - 1
                    self.samples[i][j][k] = img[row, col]

    def run(self, img):
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                k = matches = 0
                while k < num_sam and matches < min_match:
                    k = k + 1
                    dist = abs(self.samples[i][j][k] - img[i, j])
                    if dist < r:
                        matches = matches + 1
                if matches >= min_match:
                    self.samples[i][j][num_sam] = 0
                    self.FGModel[i][j] = 0
                else:
                    self.samples[i][j][num_sam] = self.samples[i][j][num_sam] + 1
                    self.FGModel[i][j] = 255
                    if self.samples[i][j][num_sam] > 50:
                        random_num = int(random.uniform(0, num_sam))
                        self.samples[i][j][random_num] = img[i, j]
                if matches >= min_match:
                    random_num = int(random.uniform(0, rand_sam))
                    if random_num == 0:
                        random_num = int(random.uniform(0, num_sam))
                        self.samples[i][j][random_num] = img[i, j]
                    random_num = int(random.uniform(0, rand_sam))
                    if random_num == 0:
                        random_num = int(random.uniform(0, 9))
                        row = i + c_x_off[random_num]
                        random_num = int(random.uniform(0, 9))
                        col = j + c_y_off[random_num]
                        if row < 0:
                            row = 0
                        if row >= img.shape[0]:
                            row = img.shape[0] - 1
                        if col < 0:
                            col = 0
                        if col >= img.shape[1]:
                            col = img.shape[1] - 1
                        random_num = int(random.uniform(0, num_sam))
                        self.samples[row][col][random_num] = img[i, j]

    def get_FGModel(self):
        return self.FGModel

'''
def get_close_filter(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv.dilate(img, kernel)
    img = cv.erode(img, kernel)
    return img


def get_open_filter(img):
    kernel = np.ones((3, 3), np.uint8)
    img = cv.erode(img, kernel)
    img = cv.dilate(img, kernel)
    return img


def draw_bounding_rect(img, original_pic):
    # contours, hierarchy = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    contours = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    for i in range(len(contours)):
        if (cv.arcLength(contours[i], True) < 100) or (cv.arcLength(contours[i], True) > 10000):
            del contours[i]
        for j in range(len(contours)):
            x, y, w, h = cv.boundingRect(contours[i])
            cv.rectangle(original_pic, (x, y), (x + w, y + h), (255, 255, 255), 2)
            cv.circle(original_pic, (x + w / 2, y + h), 2, (255, 255, 255))
    return original_pic
'''

if __name__ == "__main__":
    '''
    url = "rtsp://admin:Admin12345@192.168.0.74/Streaming/Channels/101"
    cap = cv.VideoCapture(url)
    '''
    cap = cv.VideoCapture("night1.mp4")
    if bool(1 - cap.isOpened()):
        print("No camera or video input!")
    vibe = ViBe()
    count = True
    while True:
        ret, frame = cap.read()
        gray = cv.cvtColor(frame, cv.COLOR_RGB2GRAY)
        if count:
            vibe.init(gray)
            vibe.process_first_frame(gray)
            print("Training ViBe Success")
            count = False
        else:
            vibe.run(gray)
            FGModel = vibe.get_FGModel()
            cv.imshow("FGModel", FGModel)
            #close_filter = get_close_filter(get_open_filter(FGModel))
            #cv.imshow("close_filter", close_filter)
            #frame = draw_bounding_rect(get_close_filter, frame)
            cv.imshow("frame", frame)
        if cv.waitKey(1) == 27:
            break
