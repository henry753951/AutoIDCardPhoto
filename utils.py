import numpy
import PIL

def find_upper_bound(image):
    image_np = numpy.array(image)

    # Check if the alpha channel exists
    if image_np.shape[2] < 4:
        return 0

    # Find the y-coordinate of the first non-transparent pixel
    alpha_channel = image_np[:,:,3]
    non_zero_y_indices = numpy.any(alpha_channel != 0, axis=1)
    upper_bound = numpy.argmax(non_zero_y_indices)
    return upper_bound