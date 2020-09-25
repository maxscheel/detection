import json
import torch

from os import path
from torch2trt import TRTModule
from time import time

from tools import struct
from tools.parameters import param, parse_args
from tools.image import cv
from evaluate import evaluate_image
from detection import display, detection_table #,box
from detection.export import encode_shape

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),

    input = param('',    required = True,   help = "input video sequence for detection"),
    output = param(None, type='str',        help = "output annotated video sequence"),

    scale = param(None, type='float', help = "scaling of input"),

    log = param(None, type='str', help="output json log of detections"),

    start = param(0, help = "start frame number"),
    end = param(None, type='int', help = "start end number"),

    show = param(False, help='show progress visually'),

    fp16 = param(False, help="use fp16 mode for inference"),

    threshold = param(0.3, "detection threshold")
)


def export_detections(predictions):
    def detection(p):
        object_class = classes[p.label]
        config = object_class.name

        t, box = encode_shape(p.bbox.cpu(), config)
        return struct (
            box      = box, 
            label      =  p.label,
            confidence = p.confidence.item(),
            match = p.match.item() if 'match' in p else None
        )
        
    return list(map(detection, predictions._sequence()))

def evaluate_tensorrt(trt_model, size, encoder, device = torch.cuda.current_device()):
    encoder.to(device)
    trt_model.to(device)
    
    def f(frame, nms_params=detection_table.nms_defaults):
        return evaluate_image(trt_model, frame, encoder, nms_params=nms_params, device=device).detections
    return f

def evaluate_video(frames, evaluate, size, args, classes, fps=20, scale=1):

    detection_frames = []

    start = time()
    last = start

    output_size = (int(size[0]), int(size[1]))
    nms_params = detection_table.nms_defaults._extend(threshold = args.threshold)

    out = None
    if args.output:
        import cv2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, size)

    for i, frame in enumerate(frames()):
        if i > args.start:
            if args.scale is not None:
                frame = cv.resize(frame, size)

            detections = evaluate(frame, nms_params=nms_params)

            if args.log:
                detection_frames.append(export_detections(detections))

            if args.show or args.output:
                for prediction in detections._sequence():
                    label_class = classes[prediction.label]
                    display.draw_box(frame, prediction.bbox, confidence=prediction.confidence, 
                        name=label_class.name, color=(int((1.0 - prediction.confidence) * 255), 
                        int(255 * prediction.confidence), 0))

            if args.show:
                frame = cv.resize(frame, output_size)
                cv.imshow(frame)
            
            if args.output:
                frame = cv.resize(frame, output_size)
                out.write(frame.numpy())

        if args.end is not None and i >= args.end:
            break

        if i % 50 == 49:        
            torch.cuda.current_stream().synchronize()

            now = time()
            elapsed = now - last

            print("frame: {} 50 frames in {:.1f} seconds, at {:.2f} fps".format(i, elapsed, 50./elapsed))
            last = now

    if out:
        out.release()

    if args.log:
        with open(args.log, "w") as f:
            text = json.dumps(info._extend(filename=args.input, frames=detection_frames)._to_dicts())
            f.write(text)


def main():
    args = parse_args(parameters, "video detection", "video evaluation parameters")
    print(args)

    frames, info  = cv.video_capture(args.input)
    print("Input video", info)
    scale = args.scale or 1
    size = (int(info.size[0] * scale), int(info.size[1] * scale))
    print('Scaled to:', size)

    device = torch.cuda.current_device()
    
    #x = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)       
   
    print('load model')
    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load(args.model))
    print('Done.')

    print('load encoder')
    from detection.models.ttf.model import Encoder
    encoder = Encoder(2, class_weights=[0.25], params=struct(alpha=0.54, balance=1.0))
    classes = [struct(id=0, count_weight=1, weighting=0.25, name='buoy', shape='box', colour=16776960)]
    print('done') 
    evaluate_image = evaluate_tensorrt(trt_model, size, encoder, device=device)
    print('inference time!') 
    evaluate_video(frames, evaluate_image, size, args, classes=classes, fps=info.fps)



if __name__ == "__main__":
    main()
