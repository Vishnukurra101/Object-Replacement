from flask import Flask, render_template, request
import cv2
import numpy as np
import torch
from PIL import Image
from src.model_loader import ModelLoader
from src.bounding_box import get_bounding_box_from_prompt
from src.mask_generator import create_segment_mask_from_box
from src.inpainting import perform_inpainting
import gc

app = Flask(__name__)

def resize_with_aspect_ratio(image):
    height, width = image.shape[:2]
    aspect_ratio = width / height
    if aspect_ratio > 1:
        new_size = (512, int(512/aspect_ratio))
    else:
        new_size = (int(512*aspect_ratio), 512)
    return cv2.resize(image, new_size, interpolation=cv2.INTER_LANCZOS4)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['image']
        prompt = request.form['prompt']
        new_prompt = request.form['new_prompt']

        np_img = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
        image = resize_with_aspect_ratio(image)

        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Load DINO and process
        processor, dino_model = ModelLoader.load_dino('IDEA-Research/grounding-dino-base', device)
        box = get_bounding_box_from_prompt(image, prompt, processor, dino_model, device)
        del processor, dino_model
        torch.cuda.empty_cache()
        
        if box is None:
            return "Object not found."

        # Load SAM and process
        sam_predictor = ModelLoader.load_sam('sam_vit_h_4b8939.pth', device)
        mask = create_segment_mask_from_box(image, box, sam_predictor)
        del sam_predictor
        torch.cuda.empty_cache()

        # Load inpainting model and process
        pipeline = ModelLoader.load_pipeline().to(device)
        result = perform_inpainting(pipeline, image, mask, new_prompt)
        del pipeline
        torch.cuda.empty_cache()

        original_path = 'static/original.jpg'
        mask_path = 'static/mask.jpg'
        result_path = 'static/result.jpg'

        cv2.imwrite(original_path, image)
        cv2.imwrite(mask_path, mask * 255)
        result_image = np.array(result)
        cv2.imwrite(result_path, result_image)

        return render_template('results.html', 
                             original_image=original_path, 
                             mask_image=mask_path, 
                             result_image=result_path)
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)