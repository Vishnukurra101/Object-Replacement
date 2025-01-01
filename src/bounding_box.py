import cv2
from PIL import Image
import torch

def get_bounding_box_from_prompt(image, prompt, processor, model, device):
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for PIL
    prompt = prompt.lower().strip() + '.'
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[pil_image.size[::-1]]
    )

    if len(results) > 0 and 'boxes' in results[0]:
        x_min, y_min, x_max, y_max = results[0]['boxes'][0].tolist()
        return x_min, y_min, x_max, y_max
    else:
        print("No object found for the prompt.")
        return None