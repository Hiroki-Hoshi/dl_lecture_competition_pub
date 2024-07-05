import torch
from torch import nn
import torch.nn.functional as F

class build_resnet_block(nn.Module):
    """
    a resnet block which includes two general_conv2d
    """
    def __init__(self, channels, layers=2, do_batch_norm=False):
        super(build_resnet_block,self).__init__()
        self._channels = channels
        self._layers = layers

        self.res_block = nn.Sequential(*[general_conv2d(in_channels=self._channels,
                                             out_channels=self._channels,
                                             strides=1,
                                             do_batch_norm=do_batch_norm) for i in range(self._layers)])

    def forward(self,input_res):
        inputs = input_res.clone()
        input_res = self.res_block(input_res)
        return input_res + inputs

class upsample_conv2d_and_predict_flow(nn.Module):
    """
    an upsample convolution layer which includes a nearest interpolate and a general_conv2d
    """
    def __init__(self, in_channels, out_channels, ksize=3, do_batch_norm=False):
        super(upsample_conv2d_and_predict_flow, self).__init__()
        self._in_channels = in_channels
        self._out_channels = out_channels
        self._ksize = ksize
        self._do_batch_norm = do_batch_norm

        self.general_conv2d = general_conv2d(in_channels=self._in_channels,
                                             out_channels=self._out_channels,
                                             ksize=self._ksize,
                                             strides=1,
                                             do_batch_norm=self._do_batch_norm,
                                             padding=0)
        
        self.pad = nn.ReflectionPad2d(padding=(int((self._ksize-1)/2), int((self._ksize-1)/2),
                                        int((self._ksize-1)/2), int((self._ksize-1)/2)))

        self.predict_flow = general_conv2d(in_channels=self._out_channels,
                                           out_channels=2,
                                           ksize=1,
                                           strides=1,
                                           padding=0,
                                           activation='tanh')

    def forward(self, conv):
        shape = conv.shape
        conv = nn.functional.interpolate(conv,size=[shape[2]*2,shape[3]*2],mode='nearest')
        conv = self.pad(conv)
        conv = self.general_conv2d(conv)

        flow = self.predict_flow(conv) * 256.
        
        return torch.cat([conv,flow.clone()], dim=1), flow
        

def general_conv2d(in_channels,out_channels, ksize=3, strides=2, padding=1, do_batch_norm=False, activation='relu'):
    """
    a general convolution layer which includes a conv2d, a relu and a batch_normalize
    """
    if activation == 'relu':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.ReLU(inplace=True)
            )
    elif activation == 'tanh':
        if do_batch_norm:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh(),
                nn.BatchNorm2d(out_channels,eps=1e-5,momentum=0.99)
            )
        else:
            conv2d = nn.Sequential(
                nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = ksize,
                        stride=strides,padding=padding),
                nn.Tanh()
            )
    return conv2d

# class CostVolumeLayer(nn.Module):
#     def __init__(self, max_displacement=4):
#         super(CostVolumeLayer, self).__init__()
#         self.max_displacement = max_displacement

#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.shape

#         # Pad feat2 appropriately
#         padded_feat2 = F.pad(feat2, (self.max_displacement, self.max_displacement, self.max_displacement, self.max_displacement))

#         # Initialize cost volume
#         cost_volume = torch.zeros(B,1,H,W, device=feat1.device)

#         # Compute correlation volume
#         for i in range(2*self.max_displacement + 1):
#             for j in range(2*self.max_displacement + 1):
#                 shifted_feat2 = padded_feat2[:, :, i:i + H, j:j + W]
#                 correlation = torch.sum(feat1 * shifted_feat2, dim=1, keepdim=True)
#                 print(correlation.shape,cost_volume.shape,B,C,H,W,correlation)
#                 cost_volume[:, :, i, j] = correlation
#         cost_volume = cost_volume.view(B, -1, H, W)  # (B, D*D, H, W)
#         return cost_volume
   
# class CostVolumeLayer(nn.Module):
#     def __init__(self,max_displacement=4):
#         super(CostVolumeLayer, self).__init__()
#         self.max_displacement = max_displacement

#     def forward(self, feat1, feat2):
#         B, C, H, W = feat1.shape
#         cost_volume = torch.zeros(B, H, W, H, W, device=feat1.device)
#         for i in range(H):
#             for j in range(W):
#                 # patch1 = feat1[:, :, i, j].unsqueeze(2).unsqueeze(3)  # [B, C, 1, 1]
#                 # patch2 = feat2[:, :, max(0, i - self.max_displacement):min(H, i + self.max_displacement + 1),
#                 #               max(0, j - self.max_displacement):min(W, j + self.max_displacement + 1)]  # [B, C, D, D]
#                 # patch2 = patch2.permute(0, 2, 3, 1)  # [B, D, D, C]
#                 # correlation = torch.matmul(patch1, patch2)  # [B, 1, 1, D*D]
#                 # cost_volume[:, i, j] = correlation.view(B, -1)
#                 patch1 = feat1[:, :, i, j].unsqueeze(2).unsqueeze(3)  # (B, C, 1, 1)
#                 patch2 = feat2  # (B, C, H, W)
#                 correlation = (patch1 * patch2).sum(dim=1)  # (B, H, W)
#                 cost_volume[:, i, j] = correlation
#         return cost_volume.view(B, H * W, H * W)

class CostVolumeLayer(nn.Module): #超妥協
    def __init__(self):
        super(CostVolumeLayer, self).__init__()
        self.convlay1 = general_conv2d(in_channels = 8*64,out_channels = 8*64, ksize=3, strides = 1, do_batch_norm=True, activation='relu')
        # self.convlay2 = general_conv2d(in_channels = 8*64,out_channels = 8*64, ksize=3, strides = 1, do_batch_norm=True, activation='relu')

    def forward(self, feat1, feat2):
        # B, C, H, W = feat1.shape
        cost_volume = feat2 - feat1
        cost_volume = self.convlay1(cost_volume)
        # cost_volume = self.convlay2(cost_volume)
        # feat1_flat = feat1.view(B,C*H*W)
        # feat2_flat = feat2.view(B,C*H*W)

        return cost_volume
    # def __init__(self, C, H, W):
    #     super(CostVolumeLayer, self).__init__()
    #     self.linear = nn.Linear(C * H * W * 2, C * H * W)

    # def forward(self, feat1, feat2):
    #     B, C, H, W = feat1.shape
        
    #     # フラット化して結合
    #     feat1_flat = feat1.view(B, -1)
    #     feat2_flat = feat2.view(B, -1)
    #     combined = torch.cat((feat1_flat, feat2_flat), dim=1)
        
    #     # nn.Linearを通す
    #     transformed = self.linear(combined)
        
    #     # 元の形状に戻す
    #     cost_volume = transformed.view(B, C, H, W)

    #     return cost_volume

class UpdateBlock(nn.Module):
    def __init__(self, in_channels):
        super(UpdateBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)

    def forward(self, cost_volume, feat):
        x = torch.cat([cost_volume, feat], dim=1)
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = self.conv4(x)
        return x

