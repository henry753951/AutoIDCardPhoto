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


class Window:
    def __init__(self):
        cv2.namedWindow("AutoIDPhotoTools", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("AutoIDPhotoTools", 500, 600)
        self.image = numpy.zeros((800, 600, 3), dtype=numpy.uint8)
        self.title = ""
        self.pause = False

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
        self.update()

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
            if self.pause:
                cv2.waitKey(0)
                self.pause = False


class ImageRemoveBackground:
    def __init__(self):
        self.seg_net = TracerUniversalB7(device=device, batch_size=1)
        self.fba = FBAMatting(device=device, input_tensor_size=1024, batch_size=1)
        self.trimap = TrimapGenerator(prob_threshold=231, kernel_size=30, erosion_iters=2)
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
    image_ = imageRemoveBackground.interface([image])[0]
    upperBound = find_upper_bound(image_)

    # Change the background to white
    new_image = Image.new("RGBA", image_.size, bg)
    new_image.paste(image_, (0, 0), image_)
    return new_image, upperBound


def centerAvatar(
    image: Image.Image, cropSize=(264, 330), upperBound=0, hScale=1.9, wScale=1.4
) -> Image.Image:
    width, height = image.size
    upperBound = int(upperBound)
    # using cv2 recognize face and points
    image = numpy.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml")
    blur = cv2.GaussianBlur(image, (1, 1), 0)
    blur = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    blur[:, :, 1] = blur[:, :, 1] * 0.6
    blur = cv2.cvtColor(blur, cv2.COLOR_HSV2BGR)
    faces = face_cascade.detectMultiScale(blur, 1.1, 4)
    if len(faces) == 0:
        print("No face detected!")
        image = numpy.zeros((cropSize[1], cropSize[0], 3), dtype=numpy.uint8)
        cv2.putText(
            image,
            "No face detected!",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )
        image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        return image

    # nearest center face
    centerX, centerY = width // 2, height // 2
    distance = [(x + w // 2 - centerX) ** 2 + (y + h // 2 - centerY) ** 2 for x, y, w, h in faces]
    index = distance.index(min(distance))
    x, y, w, h = faces[index]
    # Show Bounding Box
    cvImage = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    cv2.rectangle(cvImage, (x, y), (x + w, y + h), (255, 0, 0), 4)

    x = x + w // 2
    y = y + h // 2
    h = int(h * hScale)
    w = int(w * wScale)
    cv2.circle(cvImage, (x, y), 2, (255, 255, 0), 10)

    y = upperBound + h // 2
    # Show upperBound
    cv2.line(cvImage, (0, upperBound), (width, upperBound), (0, 0, 255), 4)
    # Show Bounding Box and Center
    cv2.rectangle(cvImage, (x - w // 2, y - h // 2), (x + w // 2, y + h // 2), (0, 255, 0), 4)
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
extension = "jpg"
SkipConverted = True
window = Window()
global imageRemoveBackground
imageRemoveBackground = ImageRemoveBackground()


def main():
    time.sleep(1)
    for root, dirs, files in os.walk(directory):
        for file in files:
            output_file = os.path.join(output_dir, root.split("\\")[-1], file)
            temp = output_file.split(".")
            temp[-1] = extension
            output_file = ".".join(temp)
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            # if file in converted folder, skip
            if os.path.isfile(output_file) and SkipConverted:
                print(F"Skipped {file}...")
                continue

            image = Image.open(os.path.join(root, file))
            print(F"Processing {file}... Image size: {image.size}")
            print("Removing background...")
            window.show(image, "Background Removing...")
            # Remove the background and white background
            image, upperBound = removeBackground(image, "WHITE")
            window.show(image, "Background Removed!")
            print("Background Removed!")

            print("Centering avatar...")
            # Center the avatar
            window.show(image, "Centering Avatar")

            image = centerAvatar(image, upperBound=upperBound - image.size[1] // 50)
            window.show(image, "Avatar Centered!")
            print("Avatar Centered!")

            # Save the image
            image.convert("RGB").save(output_file)
            print(F"Saved to {output_file} Image size: {image.size}\n")
    exit()


threading.Thread(target=main).start()
window.run()
