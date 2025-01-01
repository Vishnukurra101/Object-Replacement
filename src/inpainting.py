from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def perform_inpainting(pipeline, image, mask, prompt, strength=0.6):
    new_size = (512, 512)
    image_resized = Image.fromarray(image).resize(new_size)
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(new_size)

    result = pipeline(prompt=prompt, image=image_resized, mask_image=mask_resized, strength=strength)

    return result.images[0]