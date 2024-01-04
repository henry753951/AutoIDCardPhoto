from PIL import Image
from PIL import ImageDraw
from carvekit.api.interface import Interface
from carvekit.ml.wrap.fba_matting import FBAMatting
from carvekit.ml.wrap.tracer_b7 import TracerUniversalB7
from carvekit.pipelines.postprocessing import MattingMethod
from carvekit.pipelines.preprocessing import PreprocessingStub
from carvekit.trimap.generator import TrimapGenerator
import U
import os

import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'

global imageRemoveBackground
imageRemoveBackground = None


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
            pre_pipe=self.preprocessing, post_pipe=self.postprocessing, seg_pipe=self.seg_net
        )


def removeBackground(image: Image.Image, bg="WHITE") -> Image.Image:
    global imageRemoveBackground
    if imageRemoveBackground is None:
        imageRemoveBackground = ImageRemoveBackground()
    image_ = imageRemoveBackground.interface([image])[0]
    image_ = image_.convert("RGBA")

    # Change the background to white
    new_image = Image.new("RGBA", image_.size, bg)
    new_image.paste(image_, (0, 0), image_)
    return new_image


directory = './Photos'
output_dir = './Converted'
extension = '.jpg'
for root, dirs, files in os.walk(directory):
    for file in files:
        image = Image.open(os.path.join(root, file))

        # Remove the background and white background
        image = removeBackground(image, "WHITE")

        # Save the image
        output_file = os.path.join(output_dir, root.split("\\")[-1], file)
        temp = output_file.split(".")
        temp[-1] = extension
        output_file = ".".join(temp)
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        image.convert('RGB').save(output_file)
