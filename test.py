import torch
from PIL import Image
from torchvision import transforms
import torchvision
from tqdm import tqdm
import argparse
from efficientnet_pytorch import EfficientNet
from onnxsim import simplify
import onnx
import rexnetv1
import osnet
from thop import profile, clever_format
import numpy as np


def argparser():
    parser = argparse.ArgumentParser(description='Measure Accuracy in ImageNet Val')
    parser.add_argument('--model', default='resnet18', choices=['resnet18','resnet34','resnet50','resnet101', 'resnet152','efficientnet_b0','efficientnet_b1','efficientnet_b2','efficientnet_b3','efficientnet_b4','rexnetv1_1.0','rexnetv1_1.3','rexnetv1_1.5','rexnetv1_2.0','mobilenetv2_1.0','mobilenetv2_1.4','mobilenetv3-s','mobilenetv3-l', 'osnet_1.0','osnet_0.75','osnet_0.5', 'osnet_0.25'],
                        help='Detector model name')
    parser.add_argument('--export', default=False,
                        action='store_true',
                        help='dump network to onnx format')
    parser.add_argument('--data_path', default='/home/ysjo/dataset/dataset/imagenet/val/', type=str,
                        help='imagenet val dataset path')
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Batch size for training')
    parser.add_argument('--num_workers', default=16, type=int,
                        help='Number of workers used in dataloading')
    args = parser.parse_args()

    return args

def accuracy(outputs, targets):
    """Compute top-5 top-1"""
    _, pred = outputs.topk(5, dim=1, largest=True, sorted=True)
    y_ = targets.unsqueeze(1).expand_as(pred)
    correct = (pred == y_).squeeze()
    correct = correct.detach().cpu()
    correct_1 = torch.sum(correct[:,:1]).item()
    correct_5 = torch.sum(correct[:,:5]).item()

    return correct_1, correct_5

if __name__ == "__main__":
    args = argparser()

    # select model
    # resnet
    if args.model == 'resnet18':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)
        input_sizes = (224,224)
    elif args.model == 'resnet34':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=True)
        input_sizes = (224,224)
    elif args.model == 'resnet50':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
        input_sizes = (224,224)
    elif args.model == 'resnet101':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet101', pretrained=True)
        input_sizes = (224,224)
    elif args.model == 'resnet152':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet152', pretrained=True)
        input_sizes = (224,224)
    # efficientnet
    elif args.model == 'efficientnet_b0':
        model = EfficientNet.from_pretrained('efficientnet-b0')
        model.set_swish(memory_efficient=False)
        image_size = EfficientNet.get_image_size('efficientnet-b0') # 224
        input_sizes = (image_size,image_size)
    elif args.model == 'efficientnet_b1':
        model = EfficientNet.from_pretrained('efficientnet-b1')
        model.set_swish(memory_efficient=False)
        image_size = EfficientNet.get_image_size('efficientnet-b1') # 224
        input_sizes = (image_size,image_size)
    elif args.model == 'efficientnet_b2':
        model = EfficientNet.from_pretrained('efficientnet-b2')
        model.set_swish(memory_efficient=False)
        image_size = EfficientNet.get_image_size('efficientnet-b2') # 224
        input_sizes = (image_size,image_size)
    elif args.model == 'efficientnet_b3':
        model = EfficientNet.from_pretrained('efficientnet-b3')
        model.set_swish(memory_efficient=False)
        image_size = EfficientNet.get_image_size('efficientnet-b3') # 224
        input_sizes = (image_size,image_size)
    elif args.model == 'efficientnet_b4':
        model = EfficientNet.from_pretrained('efficientnet-b4')
        model.set_swish(memory_efficient=False)
        image_size = EfficientNet.get_image_size('efficientnet-b4') # 224
        input_sizes = (image_size,image_size)
    # rexnet
    elif args.model == 'rexnetv1_1.0':
        model = rexnetv1.ReXNetV1(width_mult=1.0)
        model.load_state_dict(torch.load('./rexnetv1_1.0.pth'))
        input_sizes = (224,224)
    elif args.model == 'rexnetv1_1.3':
        model = rexnetv1.ReXNetV1(width_mult=1.3)
        model.load_state_dict(torch.load('./rexnetv1_1.3.pth'))
        input_sizes = (224,224)
    elif args.model == 'rexnetv1_1.5':
        model = rexnetv1.ReXNetV1(width_mult=1.5)
        model.load_state_dict(torch.load('./rexnetv1_1.5.pth'))
        input_sizes = (224,224)
    elif args.model == 'rexnetv1_2.0':
        model = rexnetv1.ReXNetV1(width_mult=2.0)
        model.load_state_dict(torch.load('./rexnetv1_2.0.pth'))
        input_sizes = (224,224)
    # mobilenet
    elif args.model == 'mobilenetv2_1.0':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        input_sizes = (224,224)
    elif args.model == 'mobilenetv2_1.4':
        model = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        input_sizes = (224,224)
    elif args.model == 'mobilenetv3-s':
        model = torchvision.models.mobilenet_v3_small(weights='DEFAULT')
        input_sizes = (224,224)
    elif args.model == 'mobilenetv3-l':
        model = torchvision.models.mobilenet_v3_large(weights='DEFAULT')
        input_sizes = (224,224)
    # osnet
    elif args.model == 'osnet_1.0':
        model = osnet.osnet_x1_0()
        input_sizes = (224, 224)
    elif args.model == 'osnet_0.75':
        model = osnet.osnet_x0_75()
        input_sizes = (224, 224)
    elif args.model == 'osnet_0.5':
        model = osnet.osnet_x0_5()
        input_sizes = (224, 224)
    elif args.model == 'osnet_0.25':
        model = osnet.osnet_x0_25()
        input_sizes = (224, 224)
    else:
        raise ValueError("no model")
    input = torch.randn(1, 3, input_sizes[0],  input_sizes[1])
    flops, params = profile(model, inputs=(input, ))
    flops, params = clever_format([flops, params], "%.3f")
    model.eval()
    print(f" INPUT SIZE : {input_sizes} ")

    # export model
    if args.export:
        dummy_input = torch.randn(1, 3, input_sizes[0], input_sizes[1])
        torch.onnx.export(model, dummy_input, args.model+".onnx", verbose=False, input_names=['images'], output_names=['output'])
        onnx.checker.check_model(args.model+".onnx")
        model_onnx, check = simplify(args.model+".onnx")
        onnx.save(model_onnx, args.model+".onnx")

    if args.model[:12] == 'efficientnet':
        preprocess = transforms.Compose([
            transforms.Resize(input_sizes[0]+32),
            transforms.CenterCrop(input_sizes),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    else:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(input_sizes),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    testset=torchvision.datasets.ImageFolder(root=args.data_path, transform=preprocess)
    testloader=torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=False)

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        model = model.to('cuda')

    top_1 = 0
    top_5 = 0
    p_bar = tqdm(testloader)
    with torch.no_grad():
        for data in p_bar:
            imgs, labels = data
            if torch.cuda.is_available():
                imgs = imgs.to('cuda')
                labels = labels.to('cuda')
            outputs = model(imgs)
            probabilities = torch.exp(outputs)
            ret1, ret2 = accuracy(probabilities, labels)
            top_1 += ret1
            top_5 += ret2

    # for measure inference time average 100 repetitions
    repetitions = 100
    timings = np.zeros((repetitions,1))
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    img = imgs[0].unsqueeze(0)   # 1 image
    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(imgs)
            ender.record()
            # WAIT FOR GPU SYNC
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time
    print("!! Sumary !!")
    print(f"FLOPS : {flops}")
    print(f"params : {params}")
    print(f"MACs : {float(flops[:-1]) / 2}{flops[-1]}")
    print("!! average Inference Time !!")
    print(f"Inference time : {np.mean(timings)}ms")
    print("!! Accuracy !!")
    print(f"top-1 : {round(top_1/500, 2)}")
    print(f"top-5 : {round(top_5/500, 2)}")
    print(f"top-1-error : {round(100 - top_1/500,2)}")
    print(f"top-5-error : {round(100 - top_5/500,2)}")
    print("!! DONE !!")
