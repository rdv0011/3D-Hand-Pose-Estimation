import torch.nn as nn
import torch
from torchvision import models
import numpy as np


'''Encoder-Decoder - direct heatmaps for 2D and tree for 3D'''

class UpSampleBlock(nn.Module):

    def __init__(self, input_depth, output_depth):
        super(UpSampleBlock, self).__init__()
        
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
        self.convolution = nn.Sequential(
            nn.Conv2d(input_depth, output_depth, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_depth, output_depth, kernel_size=(3,3), padding=1, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, features):
        upsampled = self.upsample(x)
        concatenated = torch.cat([upsampled, features], 1)
        out = self.convolution(concatenated)
        
        return out
    

class HeatmapToCoordinated(nn.Module):

    def __init__(self, input_size):
        super(HeatmapToCoordinated, self).__init__()
        
        self.input_size = input_size
        

    def forward(self, heatmaps, cuda):
        
        dims = heatmaps.shape
        sums = torch.sum(heatmaps, dim=[2,3])
        sums = sums.unsqueeze(2).unsqueeze(3)

        normalized = heatmaps / sums
        arr = torch.tensor(np.float32(np.arange(0,dims[3]))).repeat(dims[0],dims[1],1)
        arr = arr.cuda(cuda)
        x_prob = torch.sum(normalized, dim=2)
        y_prob = torch.sum(normalized, dim=3)

        x = torch.sum((arr * x_prob), dim=2)
        y = torch.sum((arr * y_prob), dim=2)
    
        vector = torch.cat([x,y], dim=1)
        vector = vector.view(dims[0],2,dims[1]).transpose(2,1)
        vector = vector.contiguous().view(dims[0],-1)
        return vector / self.input_size
    
    
class BonesToKeypoints(nn.Module):
    def __init__(self):
        super(BonesToKeypoints, self).__init__()
        
        
    def forward(self, bones_vector, cuda):
        dims = bones_vector.shape
        bones_vector = bones_vector.reshape((dims[0],20,3))
        keypoints = torch.zeros((dims[0],21,3)).cuda(cuda)
        
        for i in range(21):
            keypoints[:,i,:] -= bones_vector[:,8,:]
        
        keypoints[:,1,:] += bones_vector[:,0,:]
        keypoints[:,2,:] += bones_vector[:,0,:] + bones_vector[:,1,:]
        keypoints[:,3,:] += bones_vector[:,0,:] + bones_vector[:,1,:] + bones_vector[:,2,:]
        keypoints[:,4,:] += bones_vector[:,0,:] + bones_vector[:,1,:] + bones_vector[:,2,:]+ bones_vector[:,3,:]
        keypoints[:,5,:] += bones_vector[:,4,:]
        keypoints[:,6,:] += bones_vector[:,4,:] + bones_vector[:,5,:]
        keypoints[:,7,:] += bones_vector[:,4,:] + bones_vector[:,5,:] + bones_vector[:,6,:]
        keypoints[:,8,:] += bones_vector[:,4,:] + bones_vector[:,5,:] + bones_vector[:,6,:] + bones_vector[:,7,:]
        keypoints[:,9,:] += bones_vector[:,8,:]
        keypoints[:,10,:] += bones_vector[:,8,:] + bones_vector[:,9,:]
        keypoints[:,11,:] += bones_vector[:,8,:] + bones_vector[:,9,:] + bones_vector[:,10,:]
        keypoints[:,12,:] += bones_vector[:,8,:] + bones_vector[:,9,:] + bones_vector[:,10,:] + bones_vector[:,11,:]
        keypoints[:,13,:] += bones_vector[:,12,:]
        keypoints[:,14,:] += bones_vector[:,12,:] + bones_vector[:,13,:]
        keypoints[:,15,:] += bones_vector[:,12,:] + bones_vector[:,13,:] + bones_vector[:,14,:]
        keypoints[:,16,:] += bones_vector[:,12,:] + bones_vector[:,13,:] + bones_vector[:,14,:] + bones_vector[:,15,:]
        
        keypoints[:,17,:] += bones_vector[:,16,:]
        keypoints[:,18,:] += bones_vector[:,16,:] + bones_vector[:,17,:]
        keypoints[:,19,:] += bones_vector[:,16,:] + bones_vector[:,17,:] + bones_vector[:,18,:]
        keypoints[:,20,:] += bones_vector[:,16,:] + bones_vector[:,17,:] + bones_vector[:,18,:] + bones_vector[:,19,:]
        
        keypoints = keypoints.reshape((dims[0],21*3))
        return keypoints
        

class EncoderDecoder(nn.Module):
    def __init__(self, input_size = 128):
        super(EncoderDecoder, self).__init__()
        
        self.input_size = input_size
        model = models.resnet34(pretrained=True)

        self.conv1 = nn.Sequential(
                        model.conv1, 
                        model.bn1, 
                        model.relu)

        self.layer1 = nn.Sequential(
                    model.maxpool,
                    model.layer1)
        self.layer2 = model.layer2
        self.layer3 = model.layer3
        self.layer4 = model.layer4
       
        self.upsample1 = UpSampleBlock(input_depth=512+256, output_depth=256)
        self.upsample2 = UpSampleBlock(input_depth=256+128, output_depth=128)
        self.upsample3 = UpSampleBlock(input_depth=128+64, output_depth=64)
        self.upsample4 = UpSampleBlock(input_depth=64+64, output_depth=64)
        
    
        self.pose_regressor_2d = nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 21, kernel_size=(3, 3), padding=1, bias=False),
                    nn.Sigmoid()
        )
        
        self.heatmaps_to_coordinates = HeatmapToCoordinated(self.input_size)
        
        self.heatmaps_processing = nn.Sequential(
                    nn.Conv2d(21, 32, kernel_size=(3, 3), padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2,2)),
                    nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=(2,2))            
        )
        
            
        self.features_3d_processing = nn.Sequential(
                    nn.Conv2d(128, 64, kernel_size=(3, 3), padding=1, bias=False),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, kernel_size=(3, 3), padding=1, bias=False),
                    nn.ReLU(inplace=True)
                
        )
        
        self.pose_regressor_3d = nn.Sequential(
                    nn.Linear(32 ** 3, 200, bias=True),  
                    nn.ReLU(inplace=True),
                    nn.Linear(200, 3*20, bias=True)
        )
                
        self.bones_to_keypoints = BonesToKeypoints()
        

    def forward(self, sample):
        features_64 = self.conv1(sample['image'])
        features_32 = self.layer1(features_64)
        features_16 = self.layer2(features_32)
        features_8 = self.layer3(features_16)
        features_4 = self.layer4(features_8)
        
        upsampled_8 = self.upsample1(features_4, features_8)
        upsampled_16 = self.upsample2(upsampled_8, features_16)
        upsampled_32 = self.upsample3(upsampled_16, features_32)
        upsampled_64 = self.upsample4(upsampled_32, features_64)
        
        heatmaps = self.pose_regressor_2d(upsampled_64)
        vector_2d = self.heatmaps_to_coordinates(heatmaps, sample['image'].device)
        
        heatmaps_proc = self.heatmaps_processing(heatmaps)
        
        features_3d = torch.cat([heatmaps_proc, upsampled_32], 1)
        features_3d = self.features_3d_processing(features_3d)
        
        features_3d = features_3d.view(features_3d.size(0), -1)
        bones = self.pose_regressor_3d(features_3d)
        out_3d = self.bones_to_keypoints(bones, sample['image'].device)

        return {'heatmaps': heatmaps, 
                'vector_2d': vector_2d,
                'vector_3d': out_3d}