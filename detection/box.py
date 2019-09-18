# Originally by Alexander (Max) deGroot
# https://github.com/amdegroot/ssd.pytorch.git


from tools import struct, Table, show_shapes

import torch
from enum import Enum
import math
import gc

import extern._C as extern


def split(boxes):
    return boxes[..., :2],  boxes[..., 2:]

def split4(boxes):
    return boxes[..., 0],  boxes[..., 1], boxes[..., 2], boxes[..., 3]

def extents_form(boxes):
    lower, upper = split(boxes)
    return torch.cat([(lower + upper) * 0.5, upper - lower], 1)

def point_form(boxes):
    centre, size = split(boxes)
    extents = size * 0.5
    return torch.cat([centre - extents, centre + extents], 1)





def transform(boxes, offset=(0, 0), scale=(1, 1)):
    lower, upper = boxes[:, :2], boxes[:, 2:]

    offset, scale = torch.Tensor(offset), torch.Tensor(scale)

    lower = lower.add(offset).mul(scale)
    upper = upper.add(offset).mul(scale)

    return torch.cat([lower.min(upper), lower.max(upper)], 1)


def transpose(boxes):
    x1, y1, x2, y2 = split4(boxes)
    return torch.stack([y1, x1, y2, x2], boxes.dim() - 1)


def flip_horizontal(boxes, width):
    x1, y1, x2, y2 = split4(boxes)
    return torch.stack([width - x2, y1, width - x1, y2], boxes.dim() - 1)

def flip_vertical(boxes, height):
    x1, y1, x2, y2 = split4(boxes)
    return torch.stack([x1, height - y2, x2, height - y1], boxes.dim() - 1)



def filter_invalid(target):
    boxes = target.bbox

    valid = (boxes[:, 2] - boxes[:, 0] > 0) & (boxes[:, 3] - boxes[:, 1] > 0)
    return target[valid.nonzero().squeeze(1)]

def filter_hidden(target, lower, upper, min_visible=0.0):
    bounds = torch.Tensor([[*lower, *upper]])
    overlaps = (intersect(bounds, target.bbox) / area(target.bbox)).squeeze(0)
    return target._index_select(overlaps.gt(min_visible).nonzero().squeeze(1))



def area(boxes):
    x1, y1, x2, y2 = boxes[:,0], boxes[:,1], boxes[:,2], boxes[:,3]
    return (x2-x1) * (y2-y1)

def clamp(boxes, lower, upper):

    boxes[:, 0].clamp_(min = lower[0])
    boxes[:, 1].clamp_(min = lower[1])
    boxes[:, 2].clamp_(max = upper[0])
    boxes[:, 3].clamp_(max = upper[1])

    return boxes


def intersect(box_a, box_b):
    """ Intersection of bounding boxes
    Args:
      box_a: (tensor) bounding boxes, Shape: [n,4].
      box_b: (tensor) bounding boxes, Shape: [m,4].
    Return:
      (tensor) intersection area, Shape: [n,m].
    """
    n = box_a.size(0)
    m = box_b.size(0)


    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(n, m, 2),
                       box_b[:, 2:].unsqueeze(0).expand(n, m, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(n, m, 2),
                       box_b[:, :2].unsqueeze(0).expand(n, m, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def intersect_matched(box_a, box_b):
    assert(box_a.size(0) == box_b.size(0))

    max_xy = torch.min(box_a[:, 2:], box_b[:, 2:])
    min_xy = torch.max(box_a[:, :2], box_b[:, :2])

    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, 0] * inter[:, 1]

def iou(box_a, box_b):
    """Compute the IOU of two sets of boxes in point form.
    Args:
        box_a, box b: Bounding boxes in point form. shapes ([n, 4], [m, 4])
    Return:
        jaccard overlap: (tensor) Shape: [n, m]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [n,m]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [n,m]
    union = area_a + area_b - inter
    return inter / union  # [n,m]

def iou_matched(box_a, box_b):  
    inter = intersect_matched(box_a, box_b)

    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1]))
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1]))
    union = area_a + area_b - inter
    return inter / union 




nms_defaults = struct(
    nms         = 0.5,
    threshold   = 0.05,
    detections  = 500
)





def nms(prediction, params, max_box_factor=200):
    # max_boxes is a 'safety' parameter, otherwise nms will can sometimes all the gpu ram

    inds = (prediction.confidence >= params.threshold).nonzero().squeeze(1)
    prediction = prediction._index_select(inds)._extend(index = inds)

    prediction = prediction._sort_on('confidence', descending=True)
    prediction = prediction._take(max_box_factor * params.detections)

    inds = extern.nms(prediction.bbox, prediction.confidence, params.nms)
    
    return prediction._index_select(inds)._take(params.detections)



    

def make_boxes(box_sizes, box_dim, image_dim):
    stride, w, h = box_dim
    n = len(box_sizes)

    sx, sy = stride, stride
    #sx, sy = image_dim[0] / w, image_dim[1] / h

    xs = torch.arange(0, w, dtype=torch.float).add_(0.5).mul_(sx).view(1, w, 1, 1).expand(h, w, n, 1)
    ys = torch.arange(0, h, dtype=torch.float).add_(0.5).mul_(sy).view(h, 1, 1, 1).expand(h, w, n, 1)


    box_sizes = torch.FloatTensor(box_sizes).view(1, 1, n, 2).expand(h, w, n, 2)
    boxes = torch.cat([xs, ys, box_sizes], 3).view(-1, 4)

    return boxes


def make_anchors(box_sizes, layer_dims, image_dim, crop_boxes=True):
    boxes = [make_boxes(boxes, box_dim, image_dim) for boxes, box_dim in zip(box_sizes, layer_dims)]
    boxes = torch.cat(boxes, 0)

    if crop_boxes:
        return extents_form(clamp(point_form(boxes), (0, 0), image_dim))

    return boxes

def anchor_sizes(size, aspects, scales):
    def anchor(s, ar):
        return (s * math.sqrt(ar), s / math.sqrt(ar))

    return [anchor(size * scale, ar) for scale in scales for ar in aspects]


default_match = struct(
    crop_boxes = False,
    match_thresholds=(0.4, 0.5), 
    match_nearest = 0,
    class_weights = None,
    overlap_attenuation = False
)



def encode(target, anchor_boxes, match_params=default_match):
    return encode_thresholds(target, anchor_boxes, match_params)



def encode_thresholds(target, anchor_boxes, match_params=default_match):
    n = anchor_boxes.size(0)
    m = target.bbox.size(0)

    if m == 0: return struct (
        location        = torch.FloatTensor(n, 4).fill_(0), 
        classification  = torch.LongTensor(n).fill_(0))

    ious = iou(point_form(anchor_boxes), target.bbox)

    if match_params.match_nearest > 0:
        top_ious, inds = ious.topk(match_params.match_nearest, dim = 0)
        ious = ious.scatter(0, inds, top_ious * 2)

    max_ious, max_ids = ious.max(1)

    class_target = encode_classes(target.label, max_ious, max_ids, 
        match_thresholds=match_params.match_thresholds, class_weights=match_params.class_weights)

    return struct (
        location  = encode_boxes(target.bbox[max_ids], anchor_boxes),
        classification = class_target
    )


def encode_classes(label, max_ious, max_ids, match_thresholds=(0.4, 0.5), class_weights=None):

    match_neg, match_pos = match_thresholds
    assert match_pos >= match_neg

    class_target = 1 + label[max_ids]
    class_target[max_ious <= match_neg] = 0 # negative label is 0

    ignore = (max_ious > match_neg) & (max_ious <= match_pos)  # ignore ious between [0.4,0.5]
    class_target[ignore] = -1  # mark ignored to -1

    return class_target

def encode_boxes(boxes, anchor_boxes):
    '''We obey the Faster RCNN box coder:
        tx = (x - anchor_x) / anchor_w
        ty = (y - anchor_y) / anchor_h
        tw = log(w / anchor_w)
        th = log(h / anchor_h)'''
    boxes_pos, boxes_size = split(extents_form(boxes))
    anchor_pos, anchor_size = split(anchor_boxes)

    loc_pos = (boxes_pos - anchor_pos) / anchor_size
    loc_size = torch.log(boxes_size/anchor_size)
    return torch.cat([loc_pos,loc_size], 1)


def decode(prediction, anchor_boxes):
    '''Decode (encoded) prediction and anchor boxes to give detected boxes.
    Args:
      preditction: (tensor) box prediction in encoded form, sized [n, 4].
      anchor_boxes: (tensor) bounding boxes in extents form, sized [m, 4].
    Returns:
      boxes: (tensor) detected boxes in point form, sized [k, 4].
      label: (tensor) detected class label [k].
    '''

    loc_pos, loc_size = split(prediction)
    anchor_pos, anchor_size = split(anchor_boxes)

    pos = loc_pos * anchor_size + anchor_pos
    sizes = loc_size.exp() * anchor_size
    

    return point_form(torch.cat([pos, sizes], 1))



def decode_nms(loc_preds, class_preds, anchor_boxes, nms_params):
    assert loc_preds.dim() == 2 and class_preds.dim() == 2

    prediction = decode(loc_preds, class_preds, anchor_boxes)
    return nms(prediction, nms_params).type_as(prediction.label)

