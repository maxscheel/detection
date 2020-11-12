from detection.models.ttf.model import Encoder
from cv2 import VideoCapture, VideoWriter_fourcc, VideoWriter, CAP_GSTREAMER, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_COUNT
import torch
from tools import struct
from tools.parameters import param, parse_args
from tools.image import cv
from evaluate import evaluate_image
from detection import display, detection_table
from os import path
from time import time
import json
import datetime

parameters = struct(
    model=param('',  required=True,
                help="model checkpoint to use for detection"),
    input=param('',    required=False,
                help="input video sequence for detection"),
    output=param(None, type='str',
                 help="output annotated video sequence"),
    outputscale=param(
        None, type='float', help="scaling of output (relative to inprelative to inputt)"),
    scale=param(None, type='float', help="scaling of input"),
    log=param(None, type='str', help="output json log of detections"),
    start=param(0, help="start frame number"),
    end=param(None, type='int', help="start end number"),
    show=param(False, help='show progress visually'),
    threshold=param(0.3, "detection threshold")
)


args = parse_args(parameters, "video detection", "video evaluation parameters")

encoder = Encoder(2, class_weights=[0.25],
                  params=struct(alpha=0.54, balance=1.0))
model = None
classes = [struct(id=0, count_weight=1, weighting=0.25,
                  name='buoy', shape='box', colour=16776960)]

model = None

def videocapture(fileName=None):
    def gstreamer():
        return ('nvarguscamerasrc ! '
                'video/x-raw(memory:NVMM), '
                'width=3264, height=1848, '
                'format=(string)NV12, framerate=28/1 ! '
                'nvvidconv flip-method=2 ! '
                'video/x-raw, width=1920, height=1080, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink')

    def gstreamer4x3_half():
        return ('nvarguscamerasrc ! '
                'video/x-raw(memory:NVMM), '
                'width=3264, height=2464, '
                'format=(string)NV12, framerate=21/1 ! '
                'nvvidconv flip-method=2 ! '
                'video/x-raw, width=1920, height=1450, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink')

    def gstreamer_half():
        return ('nvarguscamerasrc ! '
                'video/x-raw(memory:NVMM), '
                'width=3264, height=1848, '
                'format=(string)NV12, framerate=28/1 ! '
                'nvvidconv flip-method=2 ! '
                'video/x-raw, width=3264, height=1848, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink')

    def gstreamer4x3_full():
        return ('nvarguscamerasrc ! '
                'video/x-raw(memory:NVMM), '
                'width=3264, height=2464, '
                'format=(string)NV12, framerate=21/1 ! '
                'nvvidconv flip-method=2 ! '
                'video/x-raw, width=3264, height=2464, format=(string)BGRx ! '
                'videoconvert ! '
                'video/x-raw, format=(string)BGR ! appsink')

    #cap = cv2.VideoCapture(gstreamer_full(),cv2.CAP_GSTREAMER)
    #'video/x-raw, width=3264, height=1848, format=(string)BGRx ! '
    # return ('nvarguscamerasrc gainrange="1 5" ispdigitalgainrange="2 2" ! '
    if fileName:
        print(fileName)
        cap = VideoCapture(fileName)
    else:
        cap = VideoCapture(gstreamer4x3_full(),CAP_GSTREAMER)

    # cap = VideoCapture(gstreamer4x3_half(), CAP_GSTREAMER)
    #cap = VideoCapture(gstreamer_half(),CAP_GSTREAMER)
    #cap = VideoCapture(gstreamer(),CAP_GSTREAMER)

    def frames(start=0):
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                rgb_image = cv._bgr_rgb(frame)
                image = torch.from_numpy(rgb_image)
                if(image.dim() == 2):
                    image = image.view(*image.size(), 1)
                yield image
            else:
                raise StopIteration
        raise StopIteration
    if cap.isOpened():
        return frames, struct(
            fps=cap.get(CAP_PROP_FPS),
            size=(cap.get(CAP_PROP_FRAME_WIDTH),
                  cap.get(CAP_PROP_FRAME_HEIGHT)),
            frames=cap.get(CAP_PROP_FRAME_COUNT)
        )

    assert False, "video_capture: failed to load " + str(path)


def encode_shape(box, config):
    lower, upper = box[:2], box[2:]
    return 'box', struct(lower=lower.tolist(), upper=upper.tolist())


def export_detections(predictions):
    def detection(p):
        object_class = classes[p.label]
        config = object_class.name

        t, box = encode_shape(p.bbox.cpu(), config)
        return struct(
            box=box,
            label=p.label,
            confidence=p.confidence.item(),
            match=p.match.item() if 'match' in p else None
        )

    return list(map(detection, predictions._sequence()))


def replace_ext(filename, ext):
    return path.splitext(filename)[0]+ext


def load_tensorrt(trt_file, device=torch.cuda.current_device()):
    if path.isfile(trt_file) and trt_file[-3:]=='trt':
        from torch2trt import TRTModule
        x = torch.ones(1, 3, int(size[1]), int(size[0])).to(device)
        trt_model = TRTModule()
        trt_model.load_state_dict(torch.load(trt_file))
        trt_model.eval().to(device)
        trt_model(x)
        return trt_model

def evaluate_tensorrt(trt_model, encoder, device=torch.cuda.current_device()):
    encoder.to(device)
    model = trt_model
    def f(image, nms_params=detection_table.nms_defaults):
        return evaluate_image(model, image, encoder, nms_params=nms_params, device=device).detections
    return f


if __name__ == '__main__':
    in_p = args.input if 'mp4' in args.input else None
    frames, info = videocapture(in_p)
    
    scale = args.scale or 1
    output_scale = args.outputscale or 1
    size = (int(info.size[0] * scale), int(info.size[1] * scale))
    output_size = (int(info.size[0] * output_scale),
                   int(info.size[1] * output_scale))
    nms_params = detection_table.nms_defaults._extend(threshold=args.threshold)
    out = None
    if args.output:
        fourcc = VideoWriter_fourcc(*'mp4v')
        out = VideoWriter(args.output, fourcc, 2.0, output_size)

    device = torch.cuda.current_device()

    trt_model = load_tensorrt(args.model, device=device)
    evaluate = evaluate_tensorrt(trt_model, encoder, device=device)
    print("Setup done.")

    detection_frames = []
    start = time()
    last = start
    for i, frame in enumerate(frames()):
        if i > args.start:
            if scale != 1:
                print('scaling.')
                frame = cv.resize(frame, size)
            detections = evaluate(frame, nms_params=nms_params)
            print(len(list(detections._sequence())))

            if args.log:
                detection_frames.append(export_detections(detections))

            if args.show or args.output:
                for prediction in detections._sequence():
                    label_class = classes[prediction.label]
                    display.draw_box(frame, prediction.bbox,
                                     confidence=prediction.confidence,
                                     name=label_class.name,
                                     color=(int((1.0 - prediction.confidence) * 255),
                                            int(255 * prediction.confidence), 0))
                if output_size != size:
                    frame = cv.resize(frame, output_size)

            if args.show:
                cv.imshow(frame)
            if args.output:
                frame = cv.rgb_to_bgr(frame)
                out.write(frame.numpy())

        if args.end is not None and i >= args.end:
            break

        if i % 50 == 49:
            torch.cuda.current_stream().synchronize()
            now = time()
            elapsed = now - last
            print("frame: {} 50 frames in {:.1f} seconds, at {:.2f} fps".format(
                i, elapsed, 50./elapsed))
            last = now

    if out:
        out.release()

    if args.log:
        with open(args.log, "w") as f:
            dt = datetime.datetime.utcnow().isoformat()
            text = json.dumps(info._extend(
                timestamp=dt, frames=detection_frames)._to_dicts())
            f.write(text)
