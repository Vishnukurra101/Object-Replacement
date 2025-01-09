from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def perform_inpainting(pipeline, image, mask, prompt, 
                      strength=0.7,          
                      num_inference_steps=150, 
                      guidance_scale=9,       
                      negative_prompt="ugly, blurry, low quality, distorted"):
    
    # Just perform inpainting since image is already resized
    image_pil = Image.fromarray(image)
    mask_pil = Image.fromarray((mask * 255).astype(np.uint8))

    result = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        image=image_pil,
        mask_image=mask_pil,
        strength=strength,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
    
    return result.images[0]