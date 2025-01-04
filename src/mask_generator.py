import numpy as np

def create_segment_mask_from_box(image, box, predictor):
    """
    Creates a segmentation mask for an object within a given bounding box using the Segment Anything Model (SAM).

    Args:
        image: The input image as a NumPy array.
        box: A tuple or list containing the bounding box coordinates (x_min, y_min, x_max, y_max).
        predictor: The pre-loaded SAM predictor object.

    Returns:
        A NumPy array representing the segmentation mask of the object within the box.
    """

    x_min, y_min, x_max, y_max = map(int, box)

    # Calculate box center coordinates
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2

    # Create box coordinates for SAM input
    box_coords = np.array([x_min, y_min, x_max, y_max])

    # Create input_label
    input_label = np.ones(1) 

    # Set image for the predictor
    predictor.set_image(image)

    # Generate masks using box coordinates as input
    masks, scores, logits = predictor.predict(
        box=box_coords,
        point_labels=input_label,
        multimask_output=True
    )

    # Select the mask with the highest score
    best_mask_idx = np.argmax(scores)
    return masks[best_mask_idx]