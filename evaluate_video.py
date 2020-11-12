import numpy as np
import cv2

import torch

from tools import struct, tensors_to, shape, map_tensors
from tools.image.transforms import normalize_batch
from tools.parameters import param, parse_args

from tools.image import cv
from checkpoint import load_model, try_load

from os import path

from evaluate import evaluate_image
from detection import box, display, detection_table
from detection.export import encode_shape

from dataset.annotate import tagged


from time import time
import json

parameters = struct (
    model = param('',  required = True,     help = "model checkpoint to use for detection"),

    input = param('',    required = True,   help = "input video sequence for detection"),
    output = param(None, type='str',        help = "output annotated video sequence"),

    scale = param(None, type='float', help = "scaling of input"),

    log = param(None, type='str', help="output json log of detections"),

    start = param(0, help = "start frame number"),
    end = param(None, type='int', help = "start end number"),

    show = param(False, help='show progress visually'),
    recompile = param(False, help='recompile exported model'),

    fp16 = param(False, help="use fp16 mode for inference"),

    backend = param('pytorch', help='use specific backend (onnx | pytorch | tensorrt)'),
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


def evaluate_onnx(model, encoder, size, onnx_file):   
    from export_model import export_onnx
    import onnxruntime as ort

    export_onnx(model, size, onnx_file)
       

    print("Device: " + ort.get_device())
    ort_session = ort.InferenceSession(onnx_file)

    def f(image, nms_params=detection_table.nms_defaults):
        assert image.dim() == 3, "evaluate_onnx: expected image of 3d [H,W,C], got: " + str(image.shape)
        image_size = (image.shape[1], image.shape[0])
        image = image.unsqueeze(0)

        prediction = ort_session.run(None, {'input': image.numpy()})
        prediction = map_tensors(prediction, torch.squeeze, 0)

        return encoder.decode(image_size, prediction, nms_params=nms_params)
    return f


def evaluate_pytorch(model, encoder, device = torch.cuda.current_device()):
    model.to(device)
    encoder.to(device)

    def f(image, nms_params=detection_table.nms_defaults):
        return evaluate_image(model, image, encoder, nms_params=nms_params, device=device).detections
    return f


def replace_ext(filename, ext):
    return path.splitext(filename)[0]+ext   


def build_tensorrt(trt_file, model, size, device, recompile=False, fp16=True):
    from torch2trt import torch2trt, TRTModule
    import tensorrt as trt

    x = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)       

    if path.isfile(trt_file) and not recompile:
        print("Found TensorRT model file, loading...")

        # try:
        trt_model = TRTModule()
        weights = torch.load(trt_file)
        trt_model.load_state_dict(weights)

        trt_model(x)
        return trt_model

        # except Exception as e:
        #     print("Error occured: ")
        #     print(e)  

    print ("Compiling with tensorRT...")       
    trt_model = torch2trt(model, [x], max_workspace_size=1<<27, fp16_mode=fp16, 
        log_level=trt.Logger.INFO, strict_type_constraints=True, max_batch_size=1)

    torch.save(trt_model.state_dict(), trt_file)

    return trt_model

def evaluate_tensorrt(model, size, encoder, filename, device = torch.cuda.current_device()):
    model.to(device)
    encoder.to(device)

    trt_file = replace_ext(filename, ".trt")
    trt_model = build_tensorrt(trt_file, model, size, device)
    

    def f(image, nms_params=detection_table.nms_defaults):
        return evaluate_image(trt_model, image, encoder, nms_params=nms_params, device=device).detections

    return f


def initialise(model_file, model, encoder, size, backend="pytorch"):

    device = torch.cuda.current_device()

    evaluate = None
    if backend == "tensorrt":
        return evaluate_tensorrt(model, size, encoder, model_file, device=device)
    elif backend == "onnx":
        filename = args.model + ".onnx"
        return evaluate_onnx(model, encoder, size, filename)
    elif backend == "pytorch":
        return evaluate_pytorch(model, encoder, device=device)
    else:
        assert False, "unknown backend: " + backend

    print("Ready")


def evaluate_video(frames, evaluate, size, args, classes, fps=20, scale=1):

    detection_frames = []

    start = time()
    last = start

    output_size = (int(size[0] // 2), int(size[1] // 2))
    nms_params = detection_table.nms_defaults._extend(threshold = args.threshold)

    out = None
    if args.output:
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

    model, encoder, model_args = load_model(args.model)
    print("model parameters:")
    print(model_args)

    classes = model_args.dataset.classes

    frames, info  = cv.video_capture(args.input)
    print("Input video", info)

    scale = args.scale or 1
    size = (int(info.size[0] * scale), int(info.size[1] * scale))

    evaluate_image = initialise(args.model, model, encoder, size, args.backend)
    evaluate_video(frames, evaluate_image, size, args, classes=classes, fps=info.fps)



if __name__ == "__main__":
    main()
