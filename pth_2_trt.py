from tools import struct, shape
from tools.parameters import param, parse_args
from os import path
import torch
import tensorrt as trt
from torch2trt import torch2trt, TRTModule
from checkpoint import load_model

#def load_model(model_path):
    #print('loading pth...')
    #loaded = torch.load(model_path)
    #model, encoder, model_args = load_model(model)
    #args = loaded.args
    #if args.model.choice == 'ttf':
    #    from detection.models.ttf import model as m 
    #else:
    #    from detection.models.retina import model as m
    #model, encoder = m.create(args.model.parameters, args.dataset)
    #print('done.')
    #return model, encoder, model_args

def build_tensorrt(model, size, fp16=False):
    x = torch.ones(1, 3, int(size[1]), int(size[0])).to(torch.cuda.current_device())
    trt_file = args.model[:-4] + ".trt"
    print ("Compiling with tensorRT to {} @ {}".format(trt_file, size))       
    model = model.float().eval()
    x = x.float()
    trt_model = torch2trt(model, [x],
                          max_workspace_size=1<<28,
                          fp16_mode=fp16,
                          log_level=trt.Logger.Severity.VERBOSE,
                          strict_type_constraints=True,
                          max_batch_size=1)
    
    torch.save(trt_model.state_dict(), trt_file)
    return trt_model

if __name__ == '__main__':    
    parameters = struct (
        model = param('', required=True, help="model checkpoint to use for detection"),
        fp16 = param(True, help="use fp16 mode for inference"),
    )
    args = parse_args(parameters, "Provide pth file", "convert to tensorrt (trt)")
    print(args.fp16)
    model, encoder, model_args = load_model(args.model)
    model.to(torch.cuda.current_device())
    size = [1920/2,1080/2]
    trt_model = build_tensorrt(model, size, args.fp16)
