# For test model (imagenet val)

## model
- resnet18
- resnet34
- resnet50
- resnet101
- resnet152
- efficientnet_b0
- efficientnet_b1
- efficientent_b2
- efficientnet_b3
- efficientnet_b4
- rexnetv1_1.0
- rexnetv1_1.3
- rexnetv1_1.5
- rexnetv1_2.0
- mobilenetv2
- mobilenetv3-small
- mobilenetv3-large
- osnet_1.0
- osnet_0.75
- osnet_0.5

## Background
- download imagnet dataset 
- download required library


## How to use
```bash
pip install efficientnet_pytorch
```
```bash
python3 test.py --model resnet18 --data_path {imagenet dataset path}
```

## Reference
https://kaiyangzhou.github.io/deep-person-reid/MODEL_ZOO.html  
https://pytorch.org/vision/master/models/mobilenetv3.html  
https://github.com/clovaai/rexnet  
https://pytorch.org/hub/pytorch_vision_resnet/  
https://github.com/lukemelas/EfficientNet-PyTorch  
https://github.com/KaiyangZhou/deep-person-reid