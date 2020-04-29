import yaml
import torch
import argparse
import numpy as np
import json
import matplotlib.pyplot as plt

from model import get_model
from dataset import get_dataloader
from metric import get_metric
from coremltools.proto import NeuralNetwork_pb2
from PIL import Image
from coremltools.models import MLModel
from torchvision import transforms

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('exper_folder', help='Provide experiment folder')
    parser.add_argument('dataset', help='Provide dataset')
    args = parser.parse_args()
    
    print('Evaluation {} started'.format(args.exper_folder))
    
    config_file = 'experiment/' + args.exper_folder + '/config.yaml'
    with open(config_file, 'r') as f:
        config = yaml.load(f)
    config['dir'] = args.exper_folder
    config['dataset'] = args.dataset
    
    
    if args.dataset in ['stereo', 'dexter+object']:
        config['is_real'] = 1
    else:
        config['is_real'] = 0
    
    with torch.no_grad():
        model = get_model(config)
        weight_file = 'experiment/' + config['dir'] + '/' + config['weights']
        model.load_state_dict(torch.load(weight_file)['model'])
        model.eval()
    
    image_size = 128
    transform_image = transforms.Compose([
                               transforms.Resize((image_size, image_size)), 
                               transforms.ToTensor(),
                               transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                               ])
    testImage = Image.open("testImage.jpeg")
    
    # Test PyTorch model
    testImageTensor = transform_image(testImage).resize_((1, 3, 128, 128))
    predictionsPyTorch = model({'image': testImageTensor})
    print("Test PyTorch model")
    print(predictionsPyTorch)
    # Test CoreML model
    testImageResized = testImage.resize((image_size, image_size), Image.ANTIALIAS)
    coreMLModel = MLModel('hand-pose-3d.mlmodel')
    predictionsCoreML = coreMLModel.predict({'image': testImageResized}, useCPUOnly=True)
    print("Test CoreML model")
    print(predictionsCoreML)