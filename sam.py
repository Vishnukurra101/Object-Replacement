import cv2
import torch
import numpy as np
from segment_anything import SamPredictor, sam_model_registry
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection

# Load the Segment Anything model (SAM)
model_path = 'sam_vit_h_4b8939.pth'
device = 'cuda' if torch.cuda.is_available() else 'cpu'

sam = sam_model_registry['vit_h'](checkpoint=model_path)
sam.to(device)
predictor = SamPredictor(sam)

# Load Grounding DINO model for prompt-based object detection
model_id = "IDEA-Research/grounding-dino-base"
processor = AutoProcessor.from_pretrained(model_id)
dino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(device)

def get_bounding_box_from_prompt(image, prompt):
    """
    Uses the Grounding DINO model to get the bounding box for the given prompt.
    
    Args:
        image: The input image.
        prompt: The text prompt for the object to find.
    
    Returns:
        Bounding box (x_min, y_min, x_max, y_max) for the object.
    """
    from PIL import Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))  # Convert BGR to RGB for PIL
    
    # Grounding DINO requires lowercase text prompts ending with a period
    prompt = prompt.lower().strip() + '.'
    
    inputs = processor(images=pil_image, text=prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = dino_model(**inputs)
    
    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs.input_ids,
        box_threshold=0.4,
        text_threshold=0.3,
        target_sizes=[pil_image.size[::-1]]
    )
    
    if len(results) > 0 and 'boxes' in results[0]:
        # Use the box with the highest confidence score
        best_box_idx = 0  # Assuming the first box is the most relevant
        x_min, y_min, x_max, y_max = results[0]['boxes'][best_box_idx].tolist()
        return x_min, y_min, x_max, y_max
    else:
        print("No object found for the prompt.")
        return None


def create_segment_mask_from_box(image, box):
    """
    Create a segmentation mask using SAM for a given bounding box.
    
    Args:
        image: The input image.
        box: The bounding box (x_min, y_min, x_max, y_max) for the object.
    
    Returns:
        The binary mask for the segmented area.
    """
    x_min, y_min, x_max, y_max = map(int, box)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    input_point = np.array([[center_x, center_y]])  # Center of the bounding box
    input_label = np.ones(input_point.shape[0])
    
    predictor.set_image(image)
    
    masks, scores, logits = predictor.predict(
        point_coords=input_point, 
        point_labels=input_label,
        multimask_output=True  # This will generate multiple masks
    )
    
    best_mask_idx = np.argmax(scores)
    mask = masks[best_mask_idx]
    
    
    return mask


def main(image_path, prompt):
    """
    Main function to load an image, find an object based on a prompt, and create a segmentation mask.
    
    Args:
        image_path: Path to the input image.
        prompt: Text description of the object to remove.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found!")
        return
    
    # Resize image for better display if it's too large
    height, width = image.shape[:2]
    max_dimension = 800  # Maximum height or width
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, (int(width * scale), int(height * scale)))
    
    # Get the bounding box for the object described in the prompt
    print(f"Searching for '{prompt}' in the image...")
    box = get_bounding_box_from_prompt(image, prompt)
    if box is None:
        print("No object found matching the prompt.")
        return
    
    print(f"Found bounding box: {box}")
    
    # Generate the segmentation mask from the bounding box
    mask = create_segment_mask_from_box(image, box)
    
    # Create a white background with the segmented part black
    # segmented_image = np.ones_like(image) * 255  # White background
    # for i in range(3):  # Apply the mask to each channel (R, G, B)
    #     segmented_image[:, :, i] = segmented_image[:, :, i] * (1 - mask)  # Black for the segmented area
    


    # Save the result as an image
    cv2.imwrite('mask5.jpg', (mask * 255).astype(np.uint8))


if __name__ == "__main__":
    # Path to the input image (replace with your image path)
    image_path = 'image5.jpg'
    prompt = 'a dog'  # Replace with your text prompt (e.g., 'cat', 'car', etc.)
    main(image_path, prompt)
