from flask import Flask, render_template, request
import cv2
import numpy as np
import torch
from src.model_loader import ModelLoader
from src.bounding_box import get_bounding_box_from_prompt
from src.mask_generator import create_segment_mask_from_box
from src.inpainting import perform_inpainting

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Process the uploaded file and prompt
        file = request.files['image']
        prompt = request.form['prompt']
        new_prompt = request.form['new_prompt']

        # Load image
        np_img = np.fromstring(file.read(), np.uint8)
        image = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

        # Models and setup
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        pipeline = ModelLoader.load_pipeline().to("cpu")
        sam_predictor = ModelLoader.load_sam('sam_vit_h_4b8939.pth', device)
        processor, dino_model = ModelLoader.load_dino('IDEA-Research/grounding-dino-base', device)

        # Get bounding box
        box = get_bounding_box_from_prompt(image, prompt, processor, dino_model, device)
        if box is None:
            return "Object not found."

        # Get segmentation mask
        mask = create_segment_mask_from_box(image, box, sam_predictor)

        # Inpainting
        result = perform_inpainting(pipeline, image, mask, new_prompt)

        # Convert result to displayable format
        result_image = np.array(result)
        # result_image = cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR)

        # Save result
        result_path = 'static/result.jpg'
        cv2.imwrite(result_path, result_image)

        return render_template('index.html', result_image=result_path)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
