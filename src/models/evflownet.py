import torch
from torch import nn
from src.models.base import *
from typing import Dict, Any
import torch.nn.functional as F

_BASE_CHANNELS = 64

# class OpticalFlowNet(nn.Module):
#     def __init__(self, base_channels=64):
#         super(OpticalFlowNet, self).__init__()

#         # Encoder
#         self.encoder1 = self.conv_block(6, base_channels)
#         self.encoder2 = self.conv_block(base_channels, base_channels * 2)
#         self.encoder3 = self.conv_block(base_channels * 2, base_channels * 4)
#         self.encoder4 = self.conv_block(base_channels * 4, base_channels * 8)

#         # Cost volume and update block
#         self.cost_volume = CostVolumeLayer()
#         self.update_block = UpdateBlock(base_channels * 8)

#         # Decoder
#         self.decoder1 = self.upconv_block(base_channels * 8, base_channels * 4)
#         self.decoder2 = self.upconv_block(base_channels * 4, base_channels * 2)
#         self.decoder3 = self.upconv_block(base_channels * 2, base_channels)

#         # Final flow prediction
#         self.predict_flow = nn.Conv2d(base_channels, 2, kernel_size=3, padding=1)

#     def conv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def upconv_block(self, in_channels, out_channels):
#         return nn.Sequential(
#             nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )

#     def forward(self, img1, img2):
#         # Encoder
#         feat1_1 = self.encoder1(img1)
#         feat1_2 = self.encoder2(feat1_1)
#         feat1_3 = self.encoder3(feat1_2)
#         feat1_4 = self.encoder4(feat1_3)

#         feat2_1 = self.encoder1(img2)
#         feat2_2 = self.encoder2(feat2_1)
#         feat2_3 = self.encoder3(feat2_2)
#         feat2_4 = self.encoder4(feat2_3)

#         # Cost volume and update
#         cost_volume = self.cost_volume(feat1_4, feat2_4)
#         flow = self.update_block(cost_volume, feat1_4)

#         # Decoder
#         flow = self.decoder1(flow)
#         flow = self.decoder2(flow)
#         flow = self.decoder3(flow)
#         flow = self.predict_flow(flow)

#         return flow

# class CostVolumeLayer(nn.Module):
#     def __init__(self):
#         super(CostVolumeLayer, self).__init__()

#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.shape
#         cost_volume = torch.zeros(B, H, W, H, W, device=feat1.device)
#         for i in range(H):
#             for j in range(W):
#                 cost_volume[:, i, j] = (feat1[:, :, i, j].unsqueeze(2).unsqueeze(3) * feat2).sum(dim=1)
#         return cost_volume.view(B, H * W, H * W)

# class UpdateBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(UpdateBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

#     def forward(self, cost_volume, feat):
#         x = torch.cat([cost_volume, feat], dim=1)
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.conv4(x)
#         return x


class EVFlowNet(nn.Module):
    def __init__(self, args):
        super(EVFlowNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        

        self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # encoder
        skip_connections = {}
        inputs = self.encoder1(inputs)
        skip_connections['skip0'] = inputs.clone()
        inputs = self.encoder2(inputs)
        skip_connections['skip1'] = inputs.clone()
        inputs = self.encoder3(inputs)
        skip_connections['skip2'] = inputs.clone()
        inputs = self.encoder4(inputs)
        skip_connections['skip3'] = inputs.clone()

        # transition
        inputs = self.resnet_block(inputs)

        # decoder
        flow_dict = {}
        inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        return flow
        

# if __name__ == "__main__":
#     from config import configs
#     import time
#     from data_loader import EventData
#     '''
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     input_ = torch.rand(8,4,256,256).cuda()
#     a = time.time()
#     output = model(input_)
#     b = time.time()
#     print(b-a)
#     print(output['flow0'].shape, output['flow1'].shape, output['flow2'].shape, output['flow3'].shape)
#     #print(model.state_dict().keys())
#     #print(model)
#     '''
#     import numpy as np
#     args = configs()
#     model = EVFlowNet(args).cuda()
#     EventDataset = EventData(args.data_path, 'train')
#     EventDataLoader = torch.utils.data.DataLoader(dataset=EventDataset, batch_size=args.batch_size, shuffle=True)
#     #model = nn.DataParallel(model)
#     #model.load_state_dict(torch.load(args.load_path+'/model18'))
#     for input_, _, _, _ in EventDataLoader:
#         input_ = input_.cuda()
#         a = time.time()
#         (model(input_))
#         b = time.time()
#         print(b-a)


class RAFTlikeNet(nn.Module):
    def __init__(self, args):
        super(RAFTlikeNet,self).__init__()
        self._args = args

        self.encoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.encoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        
        self.featencoder1 = general_conv2d(in_channels = 4, out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.featencoder2 = general_conv2d(in_channels = _BASE_CHANNELS, out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.featencoder3 = general_conv2d(in_channels = 2*_BASE_CHANNELS, out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        self.featencoder4 = general_conv2d(in_channels = 4*_BASE_CHANNELS, out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)
        
        # Cost volume and update block
        # self.cost_volume = CostVolumeLayer(8*_BASE_CHANNELS,480,640)
        self.cost_volume = CostVolumeLayer()
        self.update_block = UpdateBlock(_BASE_CHANNELS * 16)

        # self.resnet_block = nn.Sequential(*[build_resnet_block(8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm) for i in range(2)])

        self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
                        out_channels=8*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
                        out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
                        out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
                        out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)
        
        # self.decoder1 = upsample_conv2d_and_predict_flow(in_channels=16*_BASE_CHANNELS,
        #                 out_channels=4*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        # self.decoder2 = upsample_conv2d_and_predict_flow(in_channels=8*_BASE_CHANNELS+2,
        #                 out_channels=2*_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        # self.decoder3 = upsample_conv2d_and_predict_flow(in_channels=4*_BASE_CHANNELS+2,
        #                 out_channels=_BASE_CHANNELS, do_batch_norm=not self._args.no_batch_norm)

        # self.decoder4 = upsample_conv2d_and_predict_flow(in_channels=2*_BASE_CHANNELS+2,
        #                 out_channels=int(_BASE_CHANNELS/2), do_batch_norm=not self._args.no_batch_norm)

    def forward(self, img1,img2):
        # img1 = img[:, :4, :, :]  # First 4 channels
        # img2 = img[:, 4:, :, :]  # Second 4 channels
        # encoder
        feat1_1 = self.encoder1(img1)
        feat1_2 = self.encoder2(feat1_1)
        feat1_3 = self.encoder3(feat1_2)
        feat1_4 = self.encoder4(feat1_3)

        feat2_1 = self.encoder1(img2)
        feat2_2 = self.encoder2(feat2_1)
        feat2_3 = self.encoder3(feat2_2)
        feat2_4 = self.encoder4(feat2_3)
        
        # skip_connections = {}
        feat01_1 = self.featencoder1(img1)
        # skip_connections['skip0'] = feat01_1.clone()
        feat01_2 = self.featencoder2(feat01_1)
        # skip_connections['skip1'] = feat01_2.clone()
        feat01_3 = self.featencoder3(feat01_2)
        # skip_connections['skip2'] = feat01_3.clone()
        feat01_4 = self.featencoder4(feat01_3)
        # skip_connections['skip3'] = feat01_4.clone()
        
        # skip_connections = {}
        # inputs = self.encoder1(inputs)
        # skip_connections['skip0'] = inputs.clone()
        # inputs = self.encoder2(inputs)
        # skip_connections['skip1'] = inputs.clone()
        # inputs = self.encoder3(inputs)
        # skip_connections['skip2'] = inputs.clone()
        # inputs = self.encoder4(inputs)
        # skip_connections['skip3'] = inputs.clone()

        # # transition
        # inputs = self.resnet_block(inputs)
        
        # Cost volume and update
        cost_volume = self.cost_volume(feat1_4, feat2_4)
        inputs = self.update_block(cost_volume, feat01_4)
        

        # decoder #skip connectなしだと200で3.5くらい 213で2~3くらいから6まで跳ね上がった 変更が反映されちゃった？ 1epoch後3.5くらい　次のスタートは14とか
        flow_dict = {}
        # inputs = torch.cat([inputs, skip_connections['skip3']], dim=1)
        inputs, flow = self.decoder1(inputs)
        flow_dict['flow0'] = flow.clone()

        # inputs = torch.cat([inputs, skip_connections['skip2']], dim=1)
        inputs, flow = self.decoder2(inputs)
        flow_dict['flow1'] = flow.clone()

        # inputs = torch.cat([inputs, skip_connections['skip1']], dim=1)
        inputs, flow = self.decoder3(inputs)
        flow_dict['flow2'] = flow.clone()

        # inputs = torch.cat([inputs, skip_connections['skip0']], dim=1)
        inputs, flow = self.decoder4(inputs)
        flow_dict['flow3'] = flow.clone()

        return flow
        
# class CostVolumeLayer(nn.Module):
#     def __init__(self):
#         super(CostVolumeLayer, self).__init__()

#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.shape
#         cost_volume = torch.zeros(B, H, W, H, W, device=feat1.device)
#         for i in range(H):
#             for j in range(W):
#                 cost_volume[:, i, j] = (feat1[:, :, i, j].unsqueeze(2).unsqueeze(3) * feat2).sum(dim=1)
#         return cost_volume.view(B, H * W, H * W)

# class UpdateBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(UpdateBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

#     def forward(self, cost_volume, feat):
#         x = torch.cat([cost_volume, feat], dim=1)
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = self.conv3(x)
#         x = F.relu(x)
#         x = self.conv4(x)
#         return x
