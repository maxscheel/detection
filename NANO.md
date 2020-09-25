

# Convert model.pth to TensorRT model

```bash
python3 pth_2_trt.py --model model.pth --fp16 true
```

# Run Inference on Nano

```bash
python3 evaluate_trt.py --model pth.trt --input ./snap6-15e7-1529991731.enc.mp4 --scale 0.50
```

```bash
python3 evaluate_trt.py --model model.trt --input ./snap6-15e7-1529991731.enc.mp4 --scale 0.50
{model='model.trt', input='./snap6-15e7-1529991731.enc.mp4', output=None, scale=0.5, log=None, start=0, end=None, show=False, fp16=False, threshold=0.3}
Input video {fps=14.99992501147613, size=(1920.0, 1080.0), frames=13512.0}
Scaled to: (960, 540)
load model
Done.
load encoder
done
inference time!
frame: 49 50 frames in 12.5 seconds, at 4.00 fps
frame: 99 50 frames in 12.4 seconds, at 4.04 fps
frame: 149 50 frames in 12.4 seconds, at 4.03 fps
frame: 199 50 frames in 12.4 seconds, at 4.03 fps
frame: 249 50 frames in 12.4 seconds, at 4.03 fps
```


