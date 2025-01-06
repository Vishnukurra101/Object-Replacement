from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def perform_inpainting(pipeline, image, mask, prompt, strength=1,num_inference_steps=50,guidance_scale=7):
    new_size = (512, 512)
    image_resized = Image.fromarray(image).resize(new_size)
    mask_resized = Image.fromarray((mask * 255).astype(np.uint8)).resize(new_size)

    result = pipeline(
        prompt=prompt,
        image=image_resized,
        mask_image=mask_resized, 
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        )

    return result.images[0]