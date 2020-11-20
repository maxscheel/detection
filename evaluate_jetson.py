import json
import datetime
from os import path
from time import time

import torch
from torch2trt import TRTModule

from tools import struct
from tools.image import cv
from tools.parameters import param, parse_args

from evaluate import evaluate_image
from detection import display, detection_table

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),
    input = param('',    required = False,   help = "input video sequence for detection"),
    output = param(None, type='str',        help = "output annotated video sequence"),

    log = param(None, type='str', help="output json log of detections"),

    start = param(0, help = "start frame number"),
    end = param(None, type='int', help = "start end number"),

    show = param(False, help='show progress visually'),

    threshold = param(0.3, "detection threshold")
)


def evaluate_tensorrt(trt_model, encoder, device):
    encoder.to(device)
    trt_model.to(device)
    
    def f(frame, nms_params=detection_table.nms_defaults):
        return evaluate_image(trt_model, frame, encoder, nms_params=nms_params, device=device).detections
    return f

def evaluate_video(frames, evaluate, args):
    start = time()
    last = start
    nms_params = detection_table.nms_defaults._extend(threshold = args.threshold)

    device = torch.cuda.current_device()
    
    batchSize = 4
    frameBatch = []
    detectionBatch = []
    for i, frame in enumerate(frames()):

        frameBatch.append(frame.half())
        if args.end is not None and i >= args.end:
            break
        if len(frameBatch)<batchSize:
            continue
        frameBatch = torch.stack(frameBatch).to(device)
        print('Submitting new Batch')
        for frame in frameBatch:
            d = frame.detach().clone()
            detections = evaluate(d, nms_params=nms_params)
            if args.log:
                detectionBatch.append(detections)

            if args.show or args.output:
                classes = [struct(id=0, count_weight=1, weighting=0.25, name='buoy', shape='box', colour=16776960)]
                frame = frame.cpu().byte()
                for (label, bbox, conf) in zip(detections.label.cpu(),detections.bbox.cpu(), detections.confidence.cpu()):
                    color = (int((1.0 - conf) * 255), int(255 * conf), 0)
                    label_class = classes[label]
                    display.draw_box(frame, bbox, confidence=conf, name=label_class.name, color=color)
            if args.show:
                cv.imshow(frame)
        
        if args.log:
            df = []
            for detFrame in detectionBatch:
                df_i = []
                print('## Frame')
                for (bbox, conf) in zip(detFrame.bbox.cpu().short(), detFrame.confidence.cpu()):
                    df_i.append({'bbox':bbox.tolist(), 'confidence':conf.tolist()})
                    print('### {:.2f} '.format(conf, str(bbox.tolist())) )
                df.append({'num_detections': len(df_i), 'detections':df_i})

            print('batchLog:', len(df),'', [d['num_detections'] for d in df])
            with open("{}/detections_{}.json".format(args.log, datetime.datetime.utcnow().isoformat()), "w") as f:
                json.dump({'filename': args.input, 'frames':df}, f)
            detectionBatch = []

        fiveB = batchSize*5
        if i % fiveB == (fiveB-1):        
            #torch.cuda.current_stream().synchronize()
            now = time()
            elapsed = now - last
            print("frame: {} - {} frames in {:.1f} seconds, at {:.2f} fps".format(i, fiveB, elapsed, fiveB/elapsed))
            last = now
        frameBatch = []


def main():
    '''
    https://github.com/mdegans/nano_build_opencv
    compile opencv with gstreamer support.. this takes hours/overnight.
    '''
    
    args = parse_args(parameters, "video detection", "video evaluation parameters")
    print(args)
    device = torch.cuda.current_device()
    
    from nvgst import videoin

    print('load model')
    trt_model = TRTModule()
    trt_model.load_state_dict(torch.load(args.model))

    print('load encoder')
    from detection.models.ttf.model import Encoder
    encoder = Encoder(2, class_weights=[0.25], params=struct(alpha=0.54, balance=1.0))
    evaluate_configured = evaluate_tensorrt(trt_model, encoder, device=device)
    
    print('starting video!') 
    if args.input:
        print('from file')
        frames, info = videoin(args.input)
    else:
        print('From Camera')
        frames, info = videoin()

    print('Begin inference:') 
    evaluate_video(frames, evaluate_configured, args)



if __name__ == "__main__":
    main()
