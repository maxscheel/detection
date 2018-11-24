
import torch
import torch.nn.functional as F
from torch.autograd import Variable

from detection import box
from tools import Struct, const

def bookend(*xs, dim=0):
    def to_tensor(xs):
        return xs if torch.is_tensor(xs) else torch.FloatTensor([xs])
    return torch.cat([to_tensor(x) for x in xs], dim)

def max_1d(t):
    assert t.dim() == 1
    x, i = t.max(0)
    return x.item(), i.item()



def match_boxes(prediction, target,  threshold=0.5, eps=1e-7):
    n = prediction.label.size(0)
    matches = []

    ious = box.iou(prediction.bbox, target.bbox)

    for i, p in enumerate(prediction._sequence()):
        match = None
        if ious.size(1) > 0:
            iou, j = max_1d(ious[i])
            
            label = target.label[j]
            matches_box = iou > threshold
            
            if matches_box:
                ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice

                if p.label == label:
                    match = (j, iou)

        matches.append(p._extend(match = match))
    return matches



def rev_cummax(v):
    for i in range(v.size(0) - 1, 0, -1):
        v[i - 1] = max(v[i - 1], v[i])

    return v

def area_under_curve(xs, ys):
    i = (xs[1:] != xs[:-1]).nonzero().squeeze(1)

    return ((xs[i + 1] - xs[i]) * ys[i + 1]).sum().item()



def compute_mAP(true_positives, num_target, eps=1e-7):

    false_positives = (1 - true_positives).cumsum(0)
    true_positives = true_positives.cumsum(0)

    recall = true_positives / (num_target if num_target > 0 else 1)
    precision = true_positives / (true_positives + false_positives).clamp(min = eps)

    recall = bookend(0.0, recall, 1.0)
    precision = rev_cummax(bookend(1.0, precision, 0.0))

    false_positives = bookend(0, false_positives, 0)

    true_positives = bookend(num_target, true_positives, 0)

    return Struct(
        recall = recall, 
        precision = precision, 

        false_positives = false_positives,
        true_positives = true_positives,

        false_negatives = num_target - true_positives,  
        true_negatives = true_positives.clone().zero_(),

        mAP = area_under_curve(recall, precision))

def mAP_matches(matches, num_target, eps=1e-7):
    true_positives = torch.FloatTensor([0 if m.match is None else 1 for m in matches])
    return compute_mAP(true_positives, num_target, eps)


def _match_positives(labels_pred, labels_target, ious, threshold=0.5):
    n, m = labels_pred.size(0), labels_target.size(0)
    assert ious.size() == torch.Size([n, m]) 
    ious = ious.clone()

    matches = torch.FloatTensor(n).zero_()

    for i in range(0, n):
        iou, j = ious[i].max(0)
        iou = iou.item()

        if iou > threshold:
            ious[:, j] = 0  # mark target overlaps to 0 so they won't be selected twice
            if labels_pred[i] == labels_target[j]:
                matches[i] = 1

    return matches



def match_positives(pred, target):
    assert pred.label.dim() == 1 and target.label.dim() == 1
    n, m = pred._size, target._size

    if m == 0 or n == 0:
        return const(torch.FloatTensor(n).zero_())

    ious = box.iou(pred.bbox, target.bbox)
    return lambda threshold: _match_positives(pred.label, target.label, ious, threshold=threshold)


def mAP(images, threshold=0.5, eps=1e-7):   
    return mAP_at(images, eps)(threshold)


def mAP_classes(image_pairs, num_classes, eps=1e-7):
    confidence    = torch.cat([i.prediction.confidence for i in image_pairs])
    confidence, order = confidence.sort(0, descending=True)    

    matchers =  [match_positives(i.prediction, i.target) for i in image_pairs]

    predicted_label = torch.cat([i.prediction.label for i in image_pairs])[order]
    target_label = torch.cat([i.target.label for i in image_pairs])

    num_targets = torch.bincount(target_label, minlength=num_classes)
    
    def f(threshold):      

        matches = torch.cat([match(threshold) for match in matchers])[order]
        def class_matches(i):
            return matches[(predicted_label == i).nonzero().squeeze(1)]

        return Struct(
            total = compute_mAP(matches, target_label.size(0)), 
            classes = [compute_mAP(class_matches(i), num_targets[i].item()) for i in range(0, num_classes)]
        )

    return f

