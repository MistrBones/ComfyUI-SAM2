import numpy as np

def get_positive_points(boxes, num_points_per_box=1):
    """
    Generates positive point prompts from the center of each bounding box.

    Args:
        boxes (np.ndarray): An array of bounding boxes with shape (N, 4),
        in (x1, y1, x2, y2) format.
        num_points_per_box (int): Number of positive points to generate per box.
        Currently supports 1 (center point).

    Returns:
        np.ndarray: An array of positive points with shape (N * num_points_per_box, 2)
        in (x, y) format.
    """
    if num_points_per_box != 1:
        # Placeholder for more complex logic (e.g., grid sampling)
        # For now, we'll just stick to the center.
        print(f"Warning: num_points_per_box={num_points_per_box} requested, but only 1 (center) is implemented. Using 1.")
    
    if boxes.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float32)

    # Ensure boxes is 2D
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)

    # Calculate centers
    # boxes shape is (N, 4) where [:, 0]=x1, [:, 1]=y1, [:, 2]=x2, [:, 3]=y2
    centers_x = (boxes[:, 0] + boxes[:, 2]) / 2
    centers_y = (boxes[:, 1] + boxes[:, 3]) / 2

    centers_x = np.round(centers_x)
    centers_y = np.round(centers_y)
    
    # Stack into (N, 2) array
    positive_points = np.stack([centers_x, centers_y], axis=1)
    
    return positive_points.astype(np.float32)

def get_negative_points(boxes, image_shape, num_points=5):
    """
    Generates random negative point prompts that fall outside all bounding boxes.

    Args:
        boxes (np.ndarray): An array of bounding boxes with shape (N, 4),
        in (x1, y1, x2, y2) format.
        image_shape (tuple): The (height, width) of the image.
        num_points (int): The desired number of negative points.

    Returns:
        np.ndarray: An array of negative points with shape (num_points, 2)
        in (x, y) format.
    """
    if num_points == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    # Ensure boxes is 2D
    if boxes.ndim == 1:
        boxes = boxes.reshape(1, -1)
        
    height, width = image_shape
    negative_points = []
    
    attempts = 0
    max_attempts = num_points * 100  # Avoid infinite loop if boxes cover image

    while len(negative_points) < num_points and attempts < max_attempts:
        attempts += 1
        
        # Sample a random point
        x = np.random.randint(0, width)
        y = np.random.randint(0, height)
        
        # Check if the point is inside ANY box
        is_inside = False
        for box in boxes:
            x1, y1, x2, y2 = box
            if x1 <= x < x2 and y1 <= y < y2:
                is_inside = True
                break
        
        if not is_inside:
            negative_points.append([x, y])
    
    # Ensure we return the correct shape even if we couldn't find enough points
    if len(negative_points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    
    return np.array(negative_points, dtype=np.float32)

def combine_points_and_labels(positive_points, negative_points):
    """
    Combines positive and negative points into formats required by SAM2.

    Args:
        positive_points (np.ndarray): (N, 2) array of positive points.
        negative_points (np.ndarray): (M, 2) array of negative points.

    Returns:
        tuple:
            - np.ndarray: `point_coords` (N+M, 2) array.
            - np.ndarray: `point_labels` (N+M,) array (1 for pos, 0 for neg).
    """
    # Ensure both arrays are 2D with shape (N, 2)
    if positive_points.size > 0 and positive_points.ndim == 1:
        positive_points = positive_points.reshape(-1, 2)
    if negative_points.size > 0 and negative_points.ndim == 1:
        negative_points = negative_points.reshape(-1, 2)
    
    # Handle case where both are empty
    if positive_points.shape[0] == 0 and negative_points.shape[0] == 0:
        return None, None

    # Handle case where only one type exists
    if positive_points.shape[0] == 0:
        labels_neg = np.zeros(negative_points.shape[0], dtype=np.int32)
        return negative_points, labels_neg
    
    if negative_points.shape[0] == 0:
        labels_pos = np.ones(positive_points.shape[0], dtype=np.int32)
        return positive_points, labels_pos

    # Create labels
    labels_pos = np.ones(positive_points.shape[0], dtype=np.int32)
    labels_neg = np.zeros(negative_points.shape[0], dtype=np.int32)
    
    # Concatenate points and labels
    point_coords = np.concatenate([positive_points, negative_points], axis=0)
    point_labels = np.concatenate([labels_pos, labels_neg], axis=0)
    
    return point_coords, point_labels