import torch
import torch.nn as nn

from utils import intersection_over_union

class YOLOv1Loss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YOLOv1Loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum") # not using average

        self.S = S
        self.B = B
        self.C = C

        self.lambda_coord = 5
        self.lambda_noobj = 0.5

    def forward(self, predictions, targets):
        # target ->  (batch, 7, 7, 25)
        # reshape the predictions to be (batch, 7, 7, 30)
        predictions = predictions.reshape(
            -1,
            self.S,
            self.S,
            self.C+self.B*5
        )
        # get the IOU for two predicted bbox
        iou_pred_bbox1 = intersection_over_union(
            boxes_predictions=predictions[..., 21: 25],
            boxes_targets=targets[..., 21: 25]
        )
        iou_pred_bbox2 = intersection_over_union(
            boxes_predictions=predictions[..., 26:30],
            boxes_targets=targets[..., 21: 25]
        )
        ious = torch.cat([
            iou_pred_bbox1.unsqueeze(0), # (1, batch, 7, 7, 1)
            iou_pred_bbox2.unsqueeze(0)
        ], dim=0)

        # Take the box with highest IoU out of the two prediction
        # Note that bestbox will be indices of 0, 1 for which bbox was best
        # best IoU and responsible box
        # bestbox:
        # 0 -> Box 1
        # 1 -> Box 2
        iou_maxes, bestbox = torch.max(ious, dim=0)
        # Object exists?
        # (batch, 7, 7, 1)
        exists_box = targets[..., 20].unsqueeze(3)

        #####################
        # box loss
        ####################
        box_predictions = exists_box * (
            bestbox * predictions[..., 26:30] + # bbox2
            (1 - bestbox) * predictions[..., 21:25] # bbox1
        )
        box_targets = exists_box * targets[..., 21:25]

        # the predicted width and height,
        # it sign here keep the original sign of the w, h after abs
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])
        box_predictions[..., 2: 4] = torch.sign(box_predictions[..., 2:4]) * \
                                    torch.sqrt(torch.abs(box_predictions[..., 2:4]) + 1e-6)
        box_loss = self.mse(
            torch.flatten(box_targets, end_dim=-2),
            torch.flatten(box_predictions, end_dim=-2)
        )

        #####################
        # object loss
        ####################
        # that's for confidence score
        box_pred_conf = (
            bestbox * predictions[..., 25:26] + 
            (1 - bestbox) * predictions[..., 20: 21] 
        )

        object_loss = self.mse(
            torch.flatten(iou_maxes * exists_box),
            torch.flatten(exists_box * box_pred_conf)
        )

        #####################
        # NO - object loss
        ####################
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21]),
            torch.flatten((1 - exists_box) * targets[..., 20:21]),
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26]),
            torch.flatten((1 - exists_box) * targets[..., 20:21])
        )

        #####################
        # class loss
        ####################
        class_loss = self.mse(
            torch.flatten(exists_box * predictions[..., :20], end_dim=-2,),
            torch.flatten(exists_box * targets[..., :20], end_dim=-2,),
        )

        loss = (
            self.lambda_coord * box_loss  # first two rows in paper
            + object_loss  # third row in paper
            + self.lambda_noobj * no_object_loss  # forth row
            + class_loss  # fifth row
        )

        return loss





















# def loss(targets, predictions):
#     targets = targets.reshape(-1, S, S, C+5) # (batch, 7, 7, 25)
#     predictions = predictions.reshape(-1, S, S, C+B*5) # (batch, 7, 7, 30)

#     obj = targets[..., 20].unsqueeze(3) # index 20 in confidence score
#     noobj = 1 - obj

#     targets_bbox = targets[..., 21:25]
#     pred_bbox1 = predictions[..., 21:25]
#     pred_bbox2 = predictions[..., 26:30]

#     iou_pred_bbox1 = intersection_over_union(boxes_targets=targets_bbox,
#                                              boxes_predictions=pred_bbox1) #  shape: (batch, 7, 7, 1)
#     iou_pred_bbox2 = intersection_over_union(boxes_targets=targets_bbox,
#                                                  boxes_predictions=pred_bbox2)

#     iou_pred_bboxes = torch.cat([
#             iou_pred_bbox1.unsqueeze(0), # (1, batch, 7, 7, 1)
#             iou_pred_bbox2.unsqueeze(0)], # (1, batch, 7, 7, 1)
#             dim=0) # (2, batch, 7, 7, 1)
#     """
#     across the 2 boxes
# #          → 0 means Box1 wins, 1 means Box2 wins
#     """
#     best_iou, best_bbox_index = torch.max(iou_pred_bboxes, dim=0) # shape: (batch, 7, 7, 1)

#     targets_bbox = obj * targets_bbox
#     best_bbox = obj * (
#             best_bbox_index * pred_bbox2 + 
#             (1 - best_bbox_index) * pred_bbox1
#         )

#     targets_bbox[..., 2:4] = torch.sqrt(targets_bbox[..., 2:4]) # width and height,
#     # the predicted width and height,
#     # it sign here keep the original sign of the w, h after abs
#     best_bbox[..., 2: 4] = torch.sign(best_bbox[..., 2:4]) * torch.sqrt(torch.abs(best_bbox[..., 2:4]) + 1e-6)

#     bbox_loss = mse(
#         torch.flatten(targets_bbox, end_dim=-2),
#         torch.flatten(best_bbox, end_dim=-2)
#     )

#     target_bbox_conf = targets[..., 20:21]
#     pred_bbox1_conf = predictions[..., 20:21]
#     pred_bbox2_conf = predictions[..., 25:26]


#     """
#     If object exists:
#         keep target confidence

#     If no object:
#         confidence = 0
#     """
#     target_bbox_conf = obj * target_bbox_conf
#     best_box_conf = obj * (
#         best_bbox_index * pred_bbox1_conf +
#         (1 - best_bbox_index) * pred_bbox2_conf
#     )

#     object_loss = mse(
#         torch.flatten(obj * target_bbox_conf * best_iou),
#         torch.flatten(obj * best_box_conf)
#     )
#     no_object_loss = mse(
#         torch.flatten(noobj * target_bbox_conf),
#         torch.flatten(noobj * pred_bbox1_conf)
#     )
#     no_object_loss += mse(
#         torch.flatten(noobj * target_bbox_conf),
#         torch.flatten(noobj * pred_bbox2_conf)
#     )

#     target_class = targets[..., :20]
#     pred_class = predictions[..., :20]

    
#     class_loss = mse(      #(3)
#         torch.flatten(obj * target_class, end_dim=-2),
#         torch.flatten(obj * pred_class, end_dim=-2),
#     )

#     total_loss = (
#         lambda_coord * bbox_loss           #(1)
#         + object_loss
#         + lambda_noobj * no_object_loss    #(2)
#         + class_loss
#     )
    
#     return bbox_loss, object_loss, no_object_loss, class_loss, total_loss