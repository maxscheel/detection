from cv2 import VideoCapture, CAP_GSTREAMER, CAP_PROP_FPS, CAP_PROP_FRAME_WIDTH, CAP_PROP_FRAME_HEIGHT, CAP_PROP_FRAME_COUNT
from tools.image import cv
from torch import from_numpy

def videoin(fPath=None, width=960, height=540):
    if fPath:
        gst =  ' filesrc location={} ! qtdemux ! queue ! h264parse ! nvv4l2decoder ! nvvidconv ! '.format(fPath)
    else:
        gst =( ' nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1920, height=(int)1080, ' + 
             'format=(string)NV12, framerate=(fraction)5/1 ! ' +
            'nvvidconv ! ')

    gst =  gst + ('video/x-raw, width=(int){}, height=(int){}, format=(string)BGRx ! ' +
            'videoconvert ! appsink').format(width, height)
    
    def getInfo(cap):
        ret = {"fps":cap.get(CAP_PROP_FPS),
                "size":(cap.get(CAP_PROP_FRAME_WIDTH), cap.get(CAP_PROP_FRAME_HEIGHT)),
                "frames":cap.get(CAP_PROP_FRAME_COUNT) }
        return ret
    cap = VideoCapture(gst, CAP_GSTREAMER)
    def frames(start=0):
        while(cap.isOpened()):
            ret, frame = cap.read()
            if ret:
                rgb_image = cv._bgr_rgb(frame)
                image = from_numpy(rgb_image)
                if(image.dim() == 2):
                    image = image.view(*image.size(), 1)
                yield image
            else:
                raise StopIteration
        raise StopIteration
    return frames, getInfo(cap)

if __name__ == '__main__':
    frames, info = videoin('snap6-15e7-1529991731.enc.mp4')
    
    for i, frame in enumerate(frames()):
        print( i)
        if i == 10:
            print('success')
            break
    
    
    frames, info = videoin()
    for i, frame in enumerate(frames()):
        print(i)
        print('success')
        if i == 10:
            break
