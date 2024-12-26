## Object Replacement Using Quantized Models

This project demonstrates object replacement in images using a quantized model for image inpainting (Stable Diffusion 2 Inpainting). The process involves:

1. Automatically generating a mask for object replacement based on a prompt.
2. Using SAM (Segment Anything Model) for mask generation.
3. Replacing the object in an image using a text-based prompt.
4. Displaying the results through a Flask web interface.

---

### Features
- **Quantized Models**: Efficient and high-performance object replacement.
- **Grounding DINO**: Prompt-based object detection for bounding box generation.
- **SAM (Segment Anything Model)**: Fine-grained segmentation mask generation.
- **Stable Diffusion**: Image inpainting for object replacement.
- **Web Application**: User-friendly interface for uploading images and processing results.

---

### Directory Structure
```
object-replacement/
├── app/
│   ├── static/
│   │   └── result.jpg
│   └── templates/
│       └── index.html
├── src/
│   ├── __init__.py
│   ├── model_loader.py
│   ├── bounding_box.py
│   ├── mask_generator.py
│   └── inpainting.py
├── main.py
├── requirements.txt
└── README.md
```

---

### Installation and Usage

#### 1. Clone the repository
```bash
git clone https://github.com/Vishnukurra101/Diffusion.git
cd object-replacement
```

#### 2. Install dependencies
```bash
pip install -r requirements.txt
```

#### 3. Run the Flask application
```bash
python main.py
```

#### 4. Access the web application
Open your browser and navigate to `http://127.0.0.1:5000`.

---

### How It Works
1. **Upload an Image**: Use the provided interface to upload an image.
2. **Provide Prompts**:
   - Original object to detect and remove (e.g., "a dog").
   - New object to replace it with (e.g., "a cat running in a field").
3. **Processing**:
   - The model detects the specified object and creates a mask using SAM.
   - The inpainting model replaces the object based on the new prompt.
4. **View Results**: The processed image is displayed on the webpage.

---

### Dependencies
- Python 3.8+
- PyTorch
- Transformers
- Diffusers
- OpenCV
- Flask
- segment-anything==0.1.0

---

### Acknowledgments
- [Hugging Face](https://huggingface.co/): For providing Stable Diffusion and Grounding DINO models.
- [Meta AI](https://github.com/facebookresearch/segment-anything): For the Segment Anything Model (SAM).

---

### Contact
For questions or suggestions, feel free to reach out at kurra.vishnuv@gmail.com.
