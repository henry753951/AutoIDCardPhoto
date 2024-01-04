import time
from PIL import Image
from PIL import ImageDraw
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
import cv2
import os
import numpy
import threading
import torch


from utils import find_upper_bound

device = None
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

global imageRemoveBackground
imageRemoveBackground = None


class Window:
    def __init__(self):
        cv2.namedWindow("AutoIDPhotoTools", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AutoIDPhotoTools", 500, 600)
        self.image = numpy.zeros((800, 600, 3), dtype=numpy.uint8)
        self.title = ""
        self.changed = True

    def show(self, image, title=""):
        if isinstance(image, Image.Image):
            image = numpy.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        elif isinstance(image, numpy.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        else:
            raise TypeError("image must be a PIL.Image.Image or numpy.ndarray")
        self.image = image
        self.title = title
        self.changed = True

    def update(self):
        image = self.image.copy()
        windowImage = numpy.zeros((800, 600, 3), dtype=numpy.uint8)
        # Resize image to fit window but keep aspect ratio
        fit = windowImage.shape[1] / windowImage.shape[0]
        if image.shape[1] / image.shape[0] > fit:
            # width is longer
            scale = windowImage.shape[1] / image.shape[1]
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)
        else:
            # height is longer
            scale = windowImage.shape[0] / image.shape[0]
            image = cv2.resize(image, (0, 0), fx=scale, fy=scale)

        # psate on windowImage align center
        x = (windowImage.shape[1] - image.shape[1]) // 2
        y = (windowImage.shape[0] - image.shape[0]) // 2
        windowImage[y : y + image.shape[0], x : x + image.shape[1]] = image
        cv2.putText(
            windowImage,
            self.title,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 0, 255),
            2,
        )
        cv2.imshow("AutoIDPhotoTools", windowImage)
        self.waitKey(1)

    def waitKey(self, delay=0):
        key = cv2.waitKey(delay) & 0xFF
        if key == ord("q"):
            exit()

    def run(self):
        while True:
            self.update()
            if self.changed:
                self.update()
                self.waitKey(1000)
                self.changed = False


class ImageRemoveBackground:
    def __init__(self):
        self.seg_net = TracerUniversalB7(device=device, batch_size=1)
        self.fba = FBAMatting(device=device, input_tensor_size=1024, batch_size=1)
        self.trimap = TrimapGenerator(
            prob_threshold=231, kernel_size=30, erosion_iters=2
        )
        self.preprocessing = PreprocessingStub()
        self.postprocessing = MattingMethod(
            matting_module=self.fba, trimap_generator=self.trimap, device=device
        )
        self.interface = Interface(
            pre_pipe=self.preprocessing,
            post_pipe=self.postprocessing,
            seg_pipe=self.seg_net,
        )


def removeBackground(image: Image.Image, bg="WHITE") -> Image.Image:
    global imageRemoveBackground
    if imageRemoveBackground is None:
        imageRemoveBackground = ImageRemoveBackground()
    image_ = imageRemoveBackground.interface([image])[0]
    upperBound = find_upper_bound(image_)

    # Change the background to white
    new_image = Image.new("RGBA", image_.size, bg)
    new_image.paste(image_, (0, 0), image_)
    return new_image, upperBound


def centerAvatar(
    image: Image.Image, cropSize=(400, 500), upperBound=0, hScale=1.8, wScale=1.2
) -> Image.Image:
    width, height = image.size
    # using cv2 recognize face and points
    image = numpy.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )
    faces = face_cascade.detectMultiScale(image, 1.1, 4)
    if len(faces) == 0:
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image
    x, y, w, h = faces[0]
    # Show Bounding Box
    cvImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(cvImage, (x, y), (x + w, y + h), (255, 0, 0), 4)

    x = x + w // 2
    y = y + h // 2
    h = int(h * hScale)
    w = int(w * wScale)

    if y - h // 2 < upperBound:
        h = y * 2 - upperBound

    # Show upperBound
    cv2.line(cvImage, (0, upperBound), (cvImage.shape[0], upperBound), (0, 0, 255), 4)
    # Show Bounding Box and Center
    cv2.rectangle(
        cvImage, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 4
    )
    cv2.circle(cvImage, (x, y), 2, (0, 255, 0), 10)
    window.show(cvImage, "Face Detected!")
    time.sleep(2)

    # fit h and w to cropSize
    fit = cropSize[0] / cropSize[1]
    if w / h > fit:
        # width is longer
        h = int(w / fit)
    else:
        # height is longer
        w = int(h * fit)

    # crop image
    image = image[
        max(0, y - h // 2) : min(height, y + h // 2),
        max(0, x - w // 2) : min(width, x + w // 2),
    ]

    # Resize image
    image = cv2.resize(image, cropSize)
    # PIL image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    return image


directory = "./Photos"
output_dir = "./Converted"
extension = ".jpg"
window = Window()


def main():
    time.sleep(1)
    for root, dirs, files in os.walk(directory):
        for file in files:
            image = Image.open(os.path.join(root, file))
            window.show(image, "Background Removing...")
            # Remove the background and white background
            image, upperBound = removeBackground(image, "WHITE")
            window.show(image, "Background Removed!")

            # Center the avatar
            window.show(image, "Centering Avatar")
            image = centerAvatar(image, upperBound=upperBound * 0.9)
            window.show(image, "Avatar Centered!")
            time.sleep(0.5)

            # Save the image
            output_file = os.path.join(output_dir, root.split("\\")[-1], file)
            temp = output_file.split(".")
            temp[-1] = extension
            output_file = ".".join(temp)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            image.convert("RGB").save(output_file)


threading.Thread(target=main).start()
window.run()
