import numpy as np

def create_segment_mask_from_box(image, box, predictor):
    x_min, y_min, x_max, y_max = map(int, box)
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    input_point = np.array([[center_x, center_y]])
    input_label = np.ones(input_point.shape[0])

    predictor.set_image(image)
    masks, scores, logits = predictor.predict(
        point_coords=input_point, 
        point_labels=input_label,
        multimask_output=True
    )
    best_mask_idx = np.argmax(scores)
    return masks[best_mask_idx]