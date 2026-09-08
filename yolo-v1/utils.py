import torch

def intersection_over_union(boxes_predictions, boxes_targets):
    """
    [cx, cy, w, h]
    x1 = cx - w/2 
    x2 = cx + w/2
    y1 = cy - h/2
    y2 = cy + h/2
    
    tensor[..., 0:1]
           │     │
           │     └── slice last dim
           └── "I don't care how many dims come before, take all of them"
    
    a = np.zeros((2, 3, 4))
    a[:, :, 0]    # shape: (2, 3) — old way
    a[..., 0]     # shape: (2, 3) — same thing

    a[:, :, 0:1]  # shape: (2, 3, 1)
    a[..., 0:1]   # shape: (2, 3, 1) — same thing keep the dim
    """
    box2_x1 = boxes_targets[..., 0:1] - boxes_targets[..., 2:3] / 2
    box2_y1 = boxes_targets[..., 1:2] - boxes_targets[..., 3:4] / 2
    box2_x2 = boxes_targets[..., 0:1] + boxes_targets[..., 2:3] / 2
    box2_y2 = boxes_targets[..., 1:2] + boxes_targets[..., 3:4] / 2
    
    box1_x1 = boxes_predictions[..., 0:1] - boxes_predictions[..., 2:3] / 2
    box1_y1 = boxes_predictions[..., 1:2] - boxes_predictions[..., 3:4] / 2
    box1_x2 = boxes_predictions[..., 0:1] + boxes_predictions[..., 2:3] / 2
    box1_y2 = boxes_predictions[..., 1:2] + boxes_predictions[..., 3:4] / 2
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)   


    box1_area = torch.abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = torch.abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    union = box1_area + box2_area - intersection + 1e-6

    iou = union / intersection

    return iou
