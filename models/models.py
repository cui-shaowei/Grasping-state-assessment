import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from models.layers import ConvOffset2D
import torch.nn.functional as F
import torchvision.models as models
from LSTHM.LSTHM import *
from resnet import *
import numpy
# import resnet
from mypath import Path
from models.convlstm import ConvLSTM

class MARN(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, cell_size,in_size,hybrid_in_size,num_atts,num_classes, dropout_fc):
        super(MARN, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.cell_size=cell_size
        self.in_size=in_size
        self.hybrid_in_size=hybrid_in_size
        self.fc_0 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.fc_1 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.LSTHM_T=LSTHM(self.cell_size,self.in_size,self.hybrid_in_size)
        self.LSTHM_V = LSTHM(self.cell_size, self.in_size, self.hybrid_in_size)
        self.MAB=MAB(fc_early_dim,fc_early_dim,fc_early_dim,fc_early_dim,self.hybrid_in_size, num_atts)
        self.fc_late_0 = nn.Linear(self.in_size*2, self.in_size)
        self.fc_late_1 = nn.Linear(self.in_size, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_visual, x_tactile):

        x_visual=self.fc_0(x_visual)
        x_tactile = self.fc_1(x_tactile)
        c_vi=torch.zeros(x_visual.size(0),self.cell_size).cuda()
        h_vi=torch.zeros(x_visual.size(0),self.cell_size).cuda()
        z_i=torch.zeros(x_visual.size(0),self.hybrid_in_size).cuda()

        c_ti = torch.zeros(x_visual.size(0), self.cell_size).cuda()
        h_ti = torch.zeros(x_visual.size(0), self.cell_size).cuda()
        # z_ti = torch.zeros(x_visual.size(0), self.hybrid_in_size).cuda()

        for i in range(self.T):
            c_vi,h_vi=self.LSTHM_V(x_visual[:,i,:],c_vi,h_vi,z_i)
            c_ti,h_ti=self.LSTHM_T(x_tactile[:,i,:],c_ti,h_ti,z_i)
            h_i=[h_vi,h_ti]
            z_i=self.MAB(h_i)

        x=self.fc_late_0(torch.cat((h_vi,h_ti),-1))
        x = self.dropout_fc(self.fc_late_1(x))
        return x

#

    # def __init(self,):
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride,
                     padding=0, bias=False)

def deconv(in_planes, out_planes, stride=2):
    """ Deconvolutional layer """
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size=4, stride=stride,
                              padding=1, bias=False)


class resblock(nn.Module):
    """Residual block"""

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(resblock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        # out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        # out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        # out = self.relu(out)

        return out


class Encoder(nn.Module):
    """ Encoder """

    def __init__(self):
        super(Encoder, self).__init__()
        self.conv1 = conv3x3(3, 32)
        self.conv2 = nn.Sequential(
            resblock(32, 32),
            resblock(32, 32),
            conv3x3(32, 64, 2)
        )
        self.conv3 = nn.Sequential(
            resblock(64, 64),
            resblock(64, 64),
            conv3x3(64, 128, 2)
        )
        self.conv4 = nn.Sequential(
            resblock(128, 128),
            resblock(128, 128)
        )

    def forward(self, x):
        l1 = self.conv1(x)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        out = self.conv4(l3)
        return out


class Decoder(nn.Module):
    """ Decoder """

    def __init__(self):
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(
            resblock(128, 128),
            resblock(128, 128),
            deconv(128, 64)
        )
        self.conv2 = nn.Sequential(
            resblock(64, 64),
            resblock(64, 64),
            deconv(64, 32)
        )
        self.conv3 = nn.Sequential(
            resblock(32, 32),
            resblock(32, 32)
        )
        self.conv4 = conv3x3(32, 3)

    def forward(self, x):
        l1 = self.conv1(x)
        l2 = self.conv2(l1)
        l3 = self.conv3(l2)
        out = self.conv4(l3)
        return out


class DMPHN(nn.Module):
    """ Deep Multi-Patch Hierarchical Network """

    def __init__(self):
        super(DMPHN, self).__init__()
        self.encoder1 = Encoder()
        self.decoder1 = Decoder()
        self.encoder2 = Encoder()
        self.decoder2 = Decoder()
        self.encoder3 = Encoder()
        self.decoder3 = Decoder()
        self.encoder4 = Encoder()
        self.decoder4 = Decoder()

    def split(self, fmap, width, height):
        fmap_w = fmap.shape[3]
        fmap_h = fmap.shape[2]
        tile_w = int(fmap_w / width)
        tile_h = int(fmap_h / height)
        fmap_out = [[fmap[:, :, tile_h * h:tile_h * (h + 1), tile_w * w:tile_w * (w + 1)] for w in range(width)] for h
                    in range(height)]
        return fmap_out

    def merge(self, fmap, worh='w'):
        tilenum_w = len(fmap[0])
        tilenum_h = len(fmap)
        if worh == 'w':
            width = int(tilenum_w / 2)
            height = tilenum_h
            fmap_out = [[torch.cat((fmap[h][w * 2], fmap[h][w * 2 + 1]), 3) for w in range(width)] for h in
                        range(height)]
        elif worh == 'h':
            width = tilenum_w
            height = int(tilenum_h / 2)
            fmap_out = [[torch.cat((fmap[h * 2][w], fmap[h * 2 + 1][w]), 2) for w in range(width)] for h in
                        range(height)]
        return fmap_out

    def forward(self, image):
        image2 = self.split(image, 2, 1)
        image3 = self.split(image, 2, 2)
        image4 = self.split(image, 4, 2)
        # Level 4
        inter4 = [[self.encoder4(image4[h][w]) for w in range(4)] for h in range(2)]
        interCat4 = self.merge(inter4, worh='w')
        out4 = [[self.decoder4(interCat4[h][w]) for w in range(2)] for h in range(2)]
        # Level 3
        in3 = [[image3[h][w] + out4[h][w] for w in range(2)] for h in range(2)]
        inter3 = [[self.encoder3(in3[h][w]) + interCat4[h][w] for w in range(2)] for h in range(2)]
        interCat3 = self.merge(inter3, worh='h')
        out3 = [[self.decoder3(interCat3[h][w]) for w in range(2)] for h in range(1)]
        # Level 2
        in2 = [[image2[h][w] + out3[h][w] for w in range(2)] for h in range(1)]
        inter2 = [[self.encoder2(in2[h][w]) + interCat3[h][w] for w in range(2)] for h in range(1)]
        interCat2 = self.merge(inter2, worh='w')
        out2 = self.decoder2(interCat2[0][0])
        # Level 1
        in1 = image + out2
        inter1 = self.encoder1(in1) + interCat2[0][0]
        out1 = self.decoder1(inter1)

        return out1

class Self_Attn(nn.Module):
    """ Self attention Layer"""

    def __init__(self, in_dim, activation):
        super(Self_Attn, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)  #

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        m_batchsize, C, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # B X CX(N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)  # B X C x (*W*H)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)  # BX (N) X (N)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)  # B X C X N

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, width, height)

        out = self.gamma * out + x
        return out, attention

class DRModule(nn.Module):
    "Deformable residule module"

    def __init__(self,in_dim):
        super(DRModule,self).__init__()
        self.deformconv2d=ConvOffset2D(in_dim)

    def forward(self, x):
        x =self.deformconv2d(x)+x
        return x


class SARN(nn.Module):
    "Spatially-Adaptive Residual Network"

    def __init__(self):
        super(SARN,self).__init__()
        self.conv1_1=conv3x3(3,32)
        self.conv1_2=resblock(32,32)
        self.conv1_3=resblock(32,32)
        self.conv1_4=resblock(32,32)
        self.conv2_1=conv3x3(32,64,2)
        self.conv2_2=resblock(64,64)
        self.conv2_3=resblock(64,64)
        self.conv2_4=resblock(64,64)
        self.conv_3_1=conv3x3(64,128,2)
        self.sa=Self_Attn(in_dim=128,activation="softmax")
        self.drm0=DRModule(128)
        self.drm1 = DRModule(128)
        self.drm2 = DRModule(128)
        self.drm3 = DRModule(128)
        self.drm4 = DRModule(128)
        self.drm5 = DRModule(128)
        self.deconv3_1=deconv(128,64)
        self.deconv2_4=resblock(64,64)
        self.deconv2_3=resblock(64,64)
        self.deconv2_2=resblock(64,64)
        self.deconv2_1=deconv(64,32)
        self.deconv1_4=resblock(32,32)
        self.deconv1_3=resblock(32,32)
        self.deconv1_2=resblock(32,32)
        self.deconv1_1=conv3x3(32,3)

    def forward(self, x_input):
        # print(x_input)
        x_1_4=self.conv1_4(self.conv1_3(self.conv1_2(self.conv1_1(x_input))))
        # print(x.shape)
        x_2_4=self.conv2_4(self.conv2_3(self.conv2_2(self.conv2_1(x_1_4))))
        x_3_1=self.conv_3_1(x_2_4)
        # print(x.shape)
        x_sa,_=self.sa(x_3_1)
        # print(x.shape)
        x_drm=self.drm5(self.drm4(self.drm3(self.drm2(self.drm1(self.drm0(x_3_1))))))
        # print(x.shape)
        x_3_2=self.deconv3_1(x_drm)
        x_3_2=x_2_4+x_3_2
        x_2_1=self.deconv2_1(self.deconv2_2(self.deconv2_3(self.deconv2_4(x_3_2))))
        # print(x.shape)
        x=x_2_1+x_1_4
        x=self.deconv1_1(self.deconv1_2(self.deconv1_3(self.deconv1_4(x))))
        # print(x.shape)
        x =x_input+x

        return x

class EarlyFusion(nn.Module):
    "slip detection with combined tactile and visual information"


    def __init__(self,preTrain,fc_early_dim,LSTM_layers,LSTM_units,LSTM_dropout,num_classes,dropout_fc):
        super(EarlyFusion,self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim=2048
        if preTrain == "vgg":
            self.preTrain_dim=4096

        self.fc_early=nn.Linear(self.preTrain_dim*2,fc_early_dim)
        self.LSTM=nn.LSTM(input_size=fc_early_dim,
            dropout=LSTM_dropout,
            hidden_size=LSTM_units,
            num_layers=LSTM_layers,
            batch_first=True)
        self.fc_late=nn.Linear(LSTM_units,num_classes)
        self.dropout_fc=nn.Dropout(dropout_fc)

    def forward(self, x_visual,x_tactile):
        x=torch.cat((x_visual, x_tactile), -1)
        x=self.fc_early(x)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x, None)
        x=self.dropout_fc(self.fc_late(RNN_out[:,-1,:]))
        return x

class EarlyFusionTA(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(EarlyFusionTA, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.fc_early = nn.Linear(self.preTrain_dim * 2, fc_early_dim)
        self.LSTM = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=LSTM_units,
                            num_layers=LSTM_layers,
                            batch_first=True)
        self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units*2)
        self.att_fc2 = nn.Linear(LSTM_units*2, T)
        self.att_dropout = nn.Dropout(dropout_fc)
        self.TA_fc=nn.Linear(LSTM_units*self.T,T)
        self.fc_late = nn.Linear(LSTM_units, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, x_visual, x_tactile):
        x = torch.cat((x_visual, x_tactile), -1)
        x = self.fc_early(x)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x, None)
        # print(RNN_out.shape)
        # RNN_out_cat=RNN_out.view(160,512)
        # RNN_out.is_contiguous()
        # RNN_out_cat=RNN_out.contiguous().view(RNN_out.size(0),RNN_out.size(1)*RNN_out.size(2)).contiguous()
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #                       dim=1)  # attention mechanism
        # attended = torch.zeros(RNN_out.size(0), RNN_out.size(2)).cuda()
        #
        # for i in range(RNN_out.size(1)):
        #     attended += attention[:, i].reshape(RNN_out.size(0), 1) * RNN_out[:, i, :]
        x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
                       RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
                              dim=1)  # attention mechanism
        attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()

        for i in range(RNN_out.size(1)):
            attended += attention[:, i].reshape(x.size(0), 1) * RNN_out[:, i, :]
        x = self.dropout_fc(self.fc_late(attended))
        return x

class EarlyFusionTA0(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(EarlyFusionTA0, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.fc_early = nn.Linear(self.preTrain_dim * 2, fc_early_dim)
        self.att_early = nn.Sequential(
            nn.Linear(self.preTrain_dim * 2, self.preTrain_dim * 2 // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.preTrain_dim * 2 // 16, self.preTrain_dim * 2, bias=False),
            nn.Sigmoid()
        )
        self.LSTM = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=LSTM_units,
                            num_layers=LSTM_layers,
                            batch_first=True)
        self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units*2)
        self.att_fc2 = nn.Linear(LSTM_units*2, T)
        self.att_dropout = nn.Dropout(dropout_fc)
        self.TA_fc=nn.Linear(LSTM_units*self.T,T)
        self.fc_late = nn.Linear(LSTM_units, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, x_visual, x_tactile):
        x = torch.cat((x_visual, x_tactile), -1)
        x = self.fc_early(x)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x, None)
        # print(RNN_out.shape)
        # RNN_out_cat=RNN_out.view(160,512)
        # RNN_out.is_contiguous()
        # RNN_out_cat=RNN_out.contiguous().view(RNN_out.size(0),RNN_out.size(1)*RNN_out.size(2)).contiguous()
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #                       dim=1)  # attention mechanism
        # attended = torch.zeros(RNN_out.size(0), RNN_out.size(2)).cuda()
        #
        # for i in range(RNN_out.size(1)):
        #     attended += attention[:, i].reshape(RNN_out.size(0), 1) * RNN_out[:, i, :]
        x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
                       RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
                              dim=1)  # attention mechanism
        attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()
        index=torch.max(attention,1)[1]
        for batch_idx in range(RNN_out.size(0)):
            attended[batch_idx,:]=RNN_out[batch_idx,index[batch_idx],:]
        x = self.dropout_fc(self.fc_late(attended))
        return x

class AttEarlyFusion(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(AttEarlyFusion, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.fc_early = nn.Linear(self.preTrain_dim * 2, fc_early_dim)
        self.LSTM = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=LSTM_units,
                            num_layers=LSTM_layers,
                            batch_first=True)
        self.att_early = nn.Sequential(
            nn.Linear(fc_early_dim, fc_early_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fc_early_dim, fc_early_dim, bias=False),
            nn.Sigmoid()
        )
        # self.TA_fc=nn.Linear(LSTM_units*self.T,T)
        self.fc_late = nn.Linear(LSTM_units, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, x_visual, x_tactile):
        x = torch.cat((x_visual, x_tactile), -1)
        x = self.fc_early(x)
        attention = self.att_early(x)
        x=x*attention

        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x, None)

        x = self.dropout_fc(self.fc_late(RNN_out[:,-1,:]))

        return x

class Resnet101Encoder(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Resnet101Encoder, self).__init__()

        self.resnet101=models.resnet101(pretrained=True)

    def forward(self, x_3d):
        cnn_embed_seq = []
        with torch.no_grad():
            for t in range(x_3d.size(1)):
                # CNNs
                # print(x_3d[:, t, :, :, :].shape)
                x = self.resnet101(x_3d[:, t, :, :, :])  # ResNet

                cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # print(cnn_embed_seq.shape)
        return cnn_embed_seq

class VTFfeatures(nn.Module):
    def __init__(self,visual_cnn_out_dim=(7,7,512),tactile_cnn_out_dim=(7,7,512)):
        super(VTFfeatures,self).__init__()
        self.self_att=Self_Attn(in_dim=visual_cnn_out_dim[2]+tactile_cnn_out_dim[2],activation="softmax")

    def forward(self, x_img_fea_map,x_tac_fea_map):
        #viusal-tactile modal fusion fearuremap
        x_modal_fusion=torch.zeros(x_tac_fea_map.size(0),x_tac_fea_map.size(1),x_img_fea_map.size(2)+x_tac_fea_map.size(2),x_img_fea_map.size(3)*x_tac_fea_map.size(3),x_img_fea_map.size(4)*x_tac_fea_map.size(4)).cuda()
        for i in range(x_modal_fusion.size(3)):
            for j in range(x_modal_fusion.size(4)):
                x_modal_fusion[:,:,:,i,j]=torch.cat((x_tac_fea_map[:,:,:,i//x_img_fea_map.size(3),j//x_img_fea_map.size(4)],x_img_fea_map[:,:,:,i%x_img_fea_map.size(3),j%x_img_fea_map.size(4)]),dim=2)

        att_embed_seq = []
        for t in range(x_tac_fea_map.size(1)):
            # CNNs
            out,attention=self.self_att(x_modal_fusion[:,t,:,:,:])

            # x = x.view(x.size(0), -1)           # flatten the output of conv

            # # FC layers
            # x = F.relu(self.fc1(x))
            # # x = F.dropout(x, p=self.drop_p, training=self.training)
            # x = F.relu(self.fc2(x))
            # x = F.dropout(x, p=self.drop_p, training=self.training)
            # x = self.fc3(x)
            att_embed_seq.append(out)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        att_embed_seq = torch.stack(att_embed_seq, dim=0).transpose_(0, 1)
        return  att_embed_seq

class VTFSA_LSTM(nn.Module):
    def __init__(self,visual_cnn_out_dim=(7,7,512),tactile_cnn_out_dim=(7,7,512),lstm_hidden_layers = 2,
                       lstm_hidden_nodes = 64,
                      dropout_p_lstm=0.2,
                       dropout_p_fc=0.5,encoder_fc_dim=64,fc_hidden_dim=64,num_classes=2):
        super(VTFSA_LSTM,self).__init__()
        # self.resnet_visual=resnet18(pretrained=False)
        # self.resnet_tactile=resnet18(pretrained=False)
        self.VTFfeatures=VTFfeatures(visual_cnn_out_dim=visual_cnn_out_dim,tactile_cnn_out_dim=tactile_cnn_out_dim)
        # self.self_att=Self_Attn(in_dim=visual_cnn_out_dim[2]+tactile_cnn_out_dim[2],activation=)
        # self.convlstm=ConvLSTM(input_size=(cnn_out_dim[0]*cnn_out_dim[0],cnn_out_dim[1]*cnn_out_dim[1]), input_dim=cnn_out_dim[2]*2, hidden_dim=convlstm_hidden_layers, kernel_size=convlstm_kernel, num_layers=convlstm_hidden_layers,drop_out=dropout_p_lstm,batch_first = True, bias = True, return_all_layers = False)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.encoder_fc=nn.Linear(visual_cnn_out_dim[2]+tactile_cnn_out_dim[2],encoder_fc_dim)
        self.RNN_input_size=encoder_fc_dim
        self.drop_p_rnn=dropout_p_lstm
        self.h_RNN=lstm_hidden_nodes

        self.LSTM = nn.LSTM(
            input_size=self.RNN_input_size,
            dropout=self.drop_p_rnn,
            hidden_size=self.h_RNN,
            num_layers=lstm_hidden_layers,
            batch_first=True,  # input & output will has batch size as 1s dimension. e.g. (batch, time_step, input_size)
        )

        self.fc1 = nn.Linear(lstm_hidden_nodes, fc_hidden_dim)
        self.fc2 = nn.Linear(fc_hidden_dim, num_classes)
        self.drop_out_fc=dropout_p_fc

    def forward(self, x_img,x_tac):
        # cnn_embed_seq_visual = []
        # cnn_embed_seq_tactile = []
        # for t in range(x_img.size(1)):
        #     # CNNs
        #     # print(x_3d[:, t, :, :, :].shape)
        #     x_visual = self.resnet_visual(x_img[:, t, :, :, :])  # ResNet
        #     x_tactile=self.resnet_tactile(x_tac[:, t, :, :, :])
        #     cnn_embed_seq_visual.append(x_visual)
        #     cnn_embed_seq_tactile.append(x_tactile)
        # # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        # cnn_embed_seq_visual = torch.stack(cnn_embed_seq_visual, dim=0).transpose_(0, 1)
        # cnn_embed_seq_tactile = torch.stack(cnn_embed_seq_tactile, dim=0).transpose_(0, 1)
        # # print(cnn_embed_seq_tactile.shape)
        # # x_img=self.resnet_visual(x_img)
        # # x_tac=self.resnet_tactile(x_tac)
        x_modal=self.VTFfeatures(x_img,x_tac)
        # x_att=self.self_att(x_modal)
        cnn_embed_seq_fusion=[]
        for t in range(x_img.size(1)):
            # CNNs
            # print(x_3d[:, t, :, :, :].shape)
            x_att_temp=self.avgpool(x_modal[:,t,:,:,:])
            x_att_temp=x_att_temp.view(x_att_temp.size(0), -1)
            cnn_embed_seq_fusion.append(x_att_temp)
            # cnn_embed_seq_tactile.append(x_tactile)
            # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        x = torch.stack(cnn_embed_seq_fusion, dim=0).transpose_(0, 1)
        # x = self.avgpool(x_modal)
        # x.view(x.size(0), -1)
        x=self.encoder_fc(x)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x, None)

        x = self.fc1(RNN_out[:, -1, :])  # choose RNN_out at the last time step
        x = F.relu(x)
        x = F.dropout(x, p=self.drop_out_fc, training=self.training)
        x = self.fc2(x)

        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class AttEarlyFusionTA(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(AttEarlyFusionTA, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        # self.fc_early = nn.Linear(self.preTrain_dim * 2, fc_early_dim)
        self.att_early = nn.Sequential(
            nn.Linear(self.preTrain_dim*2, self.preTrain_dim*2// 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(self.preTrain_dim*2// 16, self.preTrain_dim*2, bias=False),
            nn.Sigmoid()
        )
        self.fc_early = nn.Linear(self.preTrain_dim * 2, fc_early_dim)
        self.LSTM = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=LSTM_units,
                            num_layers=LSTM_layers,
                            batch_first=True)
        self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units * 2)
        self.att_fc2 = nn.Linear(LSTM_units * 2, T)
        self.att_dropout = nn.Dropout(dropout_fc)
        self.TA_fc = nn.Linear(LSTM_units * self.T, T)
        self.fc_late = nn.Linear(LSTM_units, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, x_visual, x_tactile):
        x = torch.cat((x_visual, x_tactile), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #              dim=1)  # attention mechanism
        # x=x*attention
        x = x*self.att_early(x)
        x=self.fc_early(x)
        self.LSTM.flatten_parameters()
        RNN_out, (h_n, h_c) = self.LSTM(x, None)

        x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
                       RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
                              dim=1)  # attention mechanism
        attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()

        for i in range(RNN_out.size(1)):
            attended += attention[:, i].reshape(x.size(0), 1) * RNN_out[:, i, :]
        x = self.dropout_fc(self.fc_late(attended))
        return x

class LateFusion(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(LateFusion, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        # self.fc_0 = nn.Linear(self.preTrain_dim, fc_early_dim)
        # self.fc_1 = nn.Linear(self.preTrain_dim, fc_early_dim)
        # self.att_early = nn.Sequential(
        #     nn.Linear(self.preTrain_dim*2, self.preTrain_dim*2// 16, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.preTrain_dim*2// 16, self.preTrain_dim*2, bias=False),
        #     nn.Sigmoid()
        # )
        # self.fc_early = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.LSTM0 = nn.LSTM(input_size=self.preTrain_dim,
                            dropout=LSTM_dropout,
                            hidden_size=self.preTrain_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)

        self.LSTM1 = nn.LSTM(input_size=self.preTrain_dim,
                            dropout=LSTM_dropout,
                            hidden_size=self.preTrain_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)

        # self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units * 2)
        # self.att_fc2 = nn.Linear(LSTM_units * 2, T)
        # self.att_dropout = nn.Dropout(dropout_fc)
        # self.TA_fc = nn.Linear(LSTM_units * self.T, T)
        self.fc_0 = nn.Linear(self.preTrain_dim*2, fc_early_dim)
        self.fc_late = nn.Linear(fc_early_dim, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, x_visual, x_tactile):
        # x = torch.cat((x_visual, x_tactile), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #              dim=1)  # attention mechanism
        # x=x*attention
        # x = x*self.att_early(x)
        # x=self.fc_early(x)
        # x_visual=self.fc_0(x_visual)
        # x_tactile=self.fc_1(x_tactile)
        self.LSTM0.flatten_parameters()
        RNN_out0, (h_n, h_c) = self.LSTM0(x_visual, None)
        self.LSTM1.flatten_parameters()
        RNN_out1, (h_n, h_c) = self.LSTM1(x_tactile, None)
        x=torch.cat((RNN_out0[:,-1,:], RNN_out1[:,-1,:]), -1)
        # x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
        #                RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #                       dim=1)  # attention mechanism
        # attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()

        # for i in range(RNN_out.size(1)):
        #     attended += attention[:, i].reshape(x.size(0), 1) * RNN_out[:, i, :]
        x=self.fc_0(x)
        x = self.dropout_fc(self.fc_late(x))
        return x

class ModalFN(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(ModalFN, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.fc_0 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.fc_1 = nn.Linear(self.preTrain_dim, fc_early_dim)
        # self.att_early = nn.Sequential(
        #     nn.Linear(self.preTrain_dim*2, self.preTrain_dim*2// 16, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.preTrain_dim*2// 16, self.preTrain_dim*2, bias=False),
        #     nn.Sigmoid()
        # )
        # self.fc_early = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.LSTM0 = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=fc_early_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)

        self.LSTM1 = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=fc_early_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=fc_early_dim*2,
                             dropout=LSTM_dropout,
                             hidden_size=fc_early_dim*2,
                             num_layers=LSTM_layers,
                             batch_first=True)
        # self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units * 2)
        # self.att_fc2 = nn.Linear(LSTM_units * 2, T)
        # self.att_dropout = nn.Dropout(dropout_fc)
        # self.TA_fc = nn.Linear(LSTM_units * self.T, T)
        self.fc_late_0 = nn.Linear(fc_early_dim*4, fc_early_dim)
        self.fc_late_1 = nn.Linear(fc_early_dim, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)

    def forward(self, x_visual, x_tactile):
        # x = torch.cat((x_visual, x_tactile), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #              dim=1)  # attention mechanism
        # x=x*attention
        # x = x*self.att_early(x)
        # x=self.fc_early(x)
        x_visual=self.fc_0(x_visual)
        x_tactile=self.fc_1(x_tactile)
        x_fusion=torch.cat((x_visual, x_tactile), -1)
        self.LSTM0.flatten_parameters()
        RNN_out0, (h_n, h_c) = self.LSTM0(x_visual, None)
        self.LSTM1.flatten_parameters()
        RNN_out1, (h_n, h_c) = self.LSTM1(x_tactile, None)
        self.LSTM2.flatten_parameters()
        RNN_out2, (h_n, h_c) = self.LSTM2(x_fusion, None)
        x=torch.cat((RNN_out0[:,-1,:], RNN_out1[:,-1,:],RNN_out2[:,-1,:]), -1)
        # x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
        #                RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #                       dim=1)  # attention mechanism
        # attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()

        # for i in range(RNN_out.size(1)):
        #     attended += attention[:, i].reshape(x.size(0), 1) * RNN_out[:, i, :]
        x=self.fc_late_0(x)
        x = self.dropout_fc(self.fc_late_1(x))
        return x

class ModalFNAtt(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(ModalFNAtt, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.fc_0 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.fc_1 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.fc_early_dim=fc_early_dim
        self.query_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.key_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.value_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)

        self.query_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.key_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.value_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        # self.att_early = nn.Sequential(
        #     nn.Linear(self.preTrain_dim*2, self.preTrain_dim*2// 16, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.preTrain_dim*2// 16, self.preTrain_dim*2, bias=False),
        #     nn.Sigmoid()
        # )
        # self.fc_early = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.LSTM0 = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=fc_early_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)

        self.LSTM1 = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=fc_early_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=fc_early_dim*2,
                             dropout=LSTM_dropout,
                             hidden_size=fc_early_dim*2,
                             num_layers=LSTM_layers,
                             batch_first=True)
        # self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units * 2)
        # self.att_fc2 = nn.Linear(LSTM_units * 2, T)
        # self.att_dropout = nn.Dropout(dropout_fc)
        # self.TA_fc = nn.Linear(LSTM_units * self.T, T)
        self.fc_late_0 = nn.Linear(fc_early_dim*4, fc_early_dim)
        self.fc_late_1 = nn.Linear(fc_early_dim, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_visual, x_tactile):
        # x = torch.cat((x_visual, x_tactile), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #              dim=1)  # attention mechanism
        # x=x*attention
        # x = x*self.att_early(x)
        # x=self.fc_early(x)
        # print(x_visual.size)
        m_batchsize,T=x_visual.size(0),x_visual.size(1)
        out_t=torch.zeros(m_batchsize,T,self.fc_early_dim).cuda()
        out_v = torch.zeros(m_batchsize, T, self.fc_early_dim).cuda()
        x_visual=self.fc_0(x_visual)
        x_tactile = self.fc_1(x_tactile)
        for i in range(x_visual.size(1)):
            x_visual_temp=torch.unsqueeze(torch.unsqueeze(x_visual[:,i,:],2),3)
            x_tactile_temp=torch.unsqueeze(torch.unsqueeze(x_tactile[:,i,:],2),3)
            proj_query_v = self.query_v(x_visual_temp).view(m_batchsize, -1, 1)  # B X CX(N)
            proj_key_v = self.key_v(x_visual_temp).view(m_batchsize, -1, 1).permute(0, 2, 1) # B X C x (*W*H)
            energy_v = torch.bmm(proj_query_v, proj_key_v)  # transpose check
            # print(proj_query_v.shape)
            # print(proj_key_v.shape)
            # print('energy:',energy_v.shape)
            attention_v = self.softmax(energy_v)  # BX (N) X (N)
            proj_value_v = self.value_v(x_visual_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)  # B X C X N

            proj_query_t = self.query_t(x_tactile_temp).view(m_batchsize, -1, 1)  # B X CX(N)
            proj_key_t = self.key_t(x_tactile_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)  # B X C x (*W*H)
            energy_t = torch.bmm(proj_query_t, proj_key_t)  # transpose check
            attention_t = self.softmax(energy_t)  # BX (N) X (N)
            proj_value_t = self.value_t(x_tactile_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)
            # print(proj_value_t.shape)
            # print(attention_v.permute(0, 2, 1).shape)
            out_v_temp = torch.bmm(proj_value_t, attention_v.permute(0, 2, 1))
            # print(torch.squeeze(out_v_temp).shape)
            # print(torch.squeeze(x_visual_temp).shape)
            out_v[:,i,:] = torch.squeeze(out_v_temp)+torch.squeeze(x_visual_temp)

            out_t_temp = torch.bmm(proj_value_v, attention_t.permute(0, 2, 1))
            out_t[:,i,:] = torch.squeeze(out_t_temp) + torch.squeeze(x_tactile_temp)

        x_fusion=torch.cat((out_t, out_v), -1)
        self.LSTM0.flatten_parameters()
        RNN_out0, (h_n, h_c) = self.LSTM0(x_visual, None)
        self.LSTM1.flatten_parameters()
        RNN_out1, (h_n, h_c) = self.LSTM1(x_tactile, None)
        self.LSTM2.flatten_parameters()
        RNN_out2, (h_n, h_c) = self.LSTM2(x_fusion, None)
        x=torch.cat((RNN_out0[:,-1,:], RNN_out1[:,-1,:],RNN_out2[:,-1,:]), -1)
        # x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
        #                RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #                       dim=1)  # attention mechanism
        # attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()

        # for i in range(RNN_out.size(1)):
        #     attended += attention[:, i].reshape(x.size(0), 1) * RNN_out[:, i, :]
        x=self.fc_late_0(x)
        x = self.dropout_fc(self.fc_late_1(x))
        return x

class ModalFNAtt1(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(ModalFNAtt1, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.fc_0 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.fc_1 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.fc_early_dim=fc_early_dim
        self.query_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.key_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.value_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)

        self.query_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.key_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        self.value_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        # self.att_early = nn.Sequential(
        #     nn.Linear(self.preTrain_dim*2, self.preTrain_dim*2// 16, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.preTrain_dim*2// 16, self.preTrain_dim*2, bias=False),
        #     nn.Sigmoid()
        # )
        # self.fc_early = nn.Linear(self.preTrain_dim, fc_early_dim)
        # self.LSTM0 = nn.LSTM(input_size=fc_early_dim,
        #                     dropout=LSTM_dropout,
        #                     hidden_size=fc_early_dim,
        #                     num_layers=LSTM_layers,
        #                     batch_first=True)
        #
        # self.LSTM1 = nn.LSTM(input_size=fc_early_dim,
        #                     dropout=LSTM_dropout,
        #                     hidden_size=fc_early_dim,
        #                     num_layers=LSTM_layers,
        #                     batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=fc_early_dim*2,
                             dropout=LSTM_dropout,
                             hidden_size=fc_early_dim*2,
                             num_layers=LSTM_layers,
                             batch_first=True)
        # self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units * 2)
        # self.att_fc2 = nn.Linear(LSTM_units * 2, T)
        # self.att_dropout = nn.Dropout(dropout_fc)
        # self.TA_fc = nn.Linear(LSTM_units * self.T, T)
        self.fc_late_0 = nn.Linear(fc_early_dim*2, fc_early_dim)
        self.fc_late_1 = nn.Linear(fc_early_dim, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_visual, x_tactile):
        # x = torch.cat((x_visual, x_tactile), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #              dim=1)  # attention mechanism
        # x=x*attention
        # x = x*self.att_early(x)
        # x=self.fc_early(x)
        # print(x_visual.size)
        m_batchsize,T=x_visual.size(0),x_visual.size(1)
        out_t=torch.zeros(m_batchsize,T,self.fc_early_dim).cuda()
        out_v = torch.zeros(m_batchsize, T, self.fc_early_dim).cuda()
        x_visual=self.fc_0(x_visual)
        x_tactile = self.fc_1(x_tactile)
        for i in range(x_visual.size(1)):
            x_visual_temp=torch.unsqueeze(torch.unsqueeze(x_visual[:,i,:],2),3)
            x_tactile_temp=torch.unsqueeze(torch.unsqueeze(x_tactile[:,i,:],2),3)
            proj_query_v = self.query_v(x_visual_temp).view(m_batchsize, -1, 1)  # B X CX(N)
            proj_key_v = self.key_v(x_visual_temp).view(m_batchsize, -1, 1).permute(0, 2, 1) # B X C x (*W*H)
            energy_v = torch.bmm(proj_query_v, proj_key_v)  # transpose check
            # print(proj_query_v.shape)
            # print(proj_key_v.shape)
            # print('energy:',energy_v.shape)
            attention_v = self.softmax(energy_v)  # BX (N) X (N)
            proj_value_v = self.value_v(x_visual_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)  # B X C X N

            proj_query_t = self.query_t(x_tactile_temp).view(m_batchsize, -1, 1)  # B X CX(N)
            proj_key_t = self.key_t(x_tactile_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)  # B X C x (*W*H)
            energy_t = torch.bmm(proj_query_t, proj_key_t)  # transpose check
            attention_t = self.softmax(energy_t)  # BX (N) X (N)
            proj_value_t = self.value_t(x_tactile_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)
            # print(proj_value_t.shape)
            # print(attention_v.permute(0, 2, 1).shape)
            out_v_temp = torch.bmm(proj_value_t, attention_v.permute(0, 2, 1))
            # print(torch.squeeze(out_v_temp).shape)
            # print(torch.squeeze(x_visual_temp).shape)
            out_v[:,i,:] = torch.squeeze(out_v_temp)+torch.squeeze(x_visual_temp)

            out_t_temp = torch.bmm(proj_value_v, attention_t.permute(0, 2, 1))
            out_t[:,i,:] = torch.squeeze(out_t_temp) + torch.squeeze(x_tactile_temp)

        x_fusion=torch.cat((out_t, out_v), -1)
        # self.LSTM0.flatten_parameters()
        # RNN_out0, (h_n, h_c) = self.LSTM0(x_visual, None)
        # self.LSTM1.flatten_parameters()
        # RNN_out1, (h_n, h_c) = self.LSTM1(x_tactile, None)
        self.LSTM2.flatten_parameters()
        RNN_out2, (h_n, h_c) = self.LSTM2(x_fusion, None)
        # x=torch.cat((RNN_out0[:,-1,:], RNN_out1[:,-1,:],RNN_out2[:,-1,:]), -1)
        # x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
        #                RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #                       dim=1)  # attention mechanism
        # attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()

        # for i in range(RNN_out.size(1)):
        #     attended += attention[:, i].reshape(x.size(0), 1) * RNN_out[:, i, :]
        x=self.fc_late_0(RNN_out2[:,-1,:])
        x = self.dropout_fc(self.fc_late_1(x))
        return x

class ModalFNAtt0(nn.Module):

    def __init__(self, preTrain, fc_early_dim,T, LSTM_layers, LSTM_units, LSTM_dropout, num_classes, dropout_fc):
        super(ModalFNAtt0, self).__init__()
        if preTrain == "resnet":
            self.preTrain_dim = 2048
        if preTrain == "vgg":
            self.preTrain_dim = 4096
        self.T=T
        self.fc_0 = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.fc_1 = nn.Linear(self.preTrain_dim, fc_early_dim)
        # self.fc_2 = nn.Linear(self.preTrain_dim*2, fc_early_dim)
        self.att_early = nn.Sequential(
            nn.Linear(fc_early_dim * 2, fc_early_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(fc_early_dim, fc_early_dim*2, bias=False),
            nn.Sigmoid()
        )
        # self.fc_early_dim=fc_early_dim
        # self.query_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        # self.key_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        # self.value_v = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        #
        # self.query_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        # self.key_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        # self.value_t = nn.Conv2d(in_channels=fc_early_dim, out_channels=fc_early_dim, kernel_size=1)
        # # self.att_early = nn.Sequential(
        #     nn.Linear(self.preTrain_dim*2, self.preTrain_dim*2// 16, bias=False),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(self.preTrain_dim*2// 16, self.preTrain_dim*2, bias=False),
        #     nn.Sigmoid()
        # )
        # self.fc_early = nn.Linear(self.preTrain_dim, fc_early_dim)
        self.LSTM0 = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=fc_early_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)

        self.LSTM1 = nn.LSTM(input_size=fc_early_dim,
                            dropout=LSTM_dropout,
                            hidden_size=fc_early_dim,
                            num_layers=LSTM_layers,
                            batch_first=True)
        self.LSTM2 = nn.LSTM(input_size=fc_early_dim*2,
                             dropout=LSTM_dropout,
                             hidden_size=fc_early_dim*2,
                             num_layers=LSTM_layers,
                             batch_first=True)
        # self.att_fc1 = nn.Linear(LSTM_units * T, LSTM_units * 2)
        # self.att_fc2 = nn.Linear(LSTM_units * 2, T)
        # self.att_dropout = nn.Dropout(dropout_fc)
        # self.TA_fc = nn.Linear(LSTM_units * self.T, T)
        self.fc_late_0 = nn.Linear(fc_early_dim*4, fc_early_dim)
        self.fc_late_1 = nn.Linear(fc_early_dim, num_classes)
        self.dropout_fc = nn.Dropout(dropout_fc)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x_visual, x_tactile):
        # x = torch.cat((x_visual, x_tactile), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #              dim=1)  # attention mechanism
        # x=x*attention
        # x = x*self.att_early(x)
        # x=self.fc_early(x)
        # print(x_visual.size)
        # m_batchsize,T=x_visual.size(0),x_visual.size(1)
        # out_t=torch.zeros(m_batchsize,T,self.fc_early_dim).cuda()
        # out_v = torch.zeros(m_batchsize, T, self.fc_early_dim).cuda()
        # x_fusion = self.fc_2(torch.cat((x_visual, x_tactile), -1))
        x_visual=self.fc_0(x_visual)
        # print(x_visual.shape)
        x_tactile = self.fc_1(x_tactile)
        x_fusion=torch.zeros(x_tactile.size(0),x_tactile.size(1),x_tactile.size(2)*2).cuda()
        # for i in range(x_visual.size(1)):
        #     x_visual_temp=torch.unsqueeze(torch.unsqueeze(x_visual[:,i,:],2),3)
        #     x_tactile_temp=torch.unsqueeze(torch.unsqueeze(x_tactile[:,i,:],2),3)
        #     proj_query_v = self.query_v(x_visual_temp).view(m_batchsize, -1, 1)  # B X CX(N)
        #     proj_key_v = self.key_v(x_visual_temp).view(m_batchsize, -1, 1).permute(0, 2, 1) # B X C x (*W*H)
        #     energy_v = torch.bmm(proj_query_v, proj_key_v)  # transpose check
        #     # print(proj_query_v.shape)
        #     # print(proj_key_v.shape)
        #     # print('energy:',energy_v.shape)
        #     attention_v = self.softmax(energy_v)  # BX (N) X (N)
        #     proj_value_v = self.value_v(x_visual_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)  # B X C X N
        #
        #     proj_query_t = self.query_t(x_tactile_temp).view(m_batchsize, -1, 1)  # B X CX(N)
        #     proj_key_t = self.key_t(x_tactile_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)  # B X C x (*W*H)
        #     energy_t = torch.bmm(proj_query_t, proj_key_t)  # transpose check
        #     attention_t = self.softmax(energy_t)  # BX (N) X (N)
        #     proj_value_t = self.value_t(x_tactile_temp).view(m_batchsize, -1, 1).permute(0, 2, 1)
        #     # print(proj_value_t.shape)
        #     # print(attention_v.permute(0, 2, 1).shape)
        #     out_v_temp = torch.bmm(proj_value_t, attention_v.permute(0, 2, 1))
        #     # print(torch.squeeze(out_v_temp).shape)
        #     # print(torch.squeeze(x_visual_temp).shape)
        #     out_v[:,i,:] = torch.squeeze(out_v_temp)+torch.squeeze(x_visual_temp)
        #
        #     out_t_temp = torch.bmm(proj_value_v, attention_t.permute(0, 2, 1))
        #     out_t[:,i,:] = torch.squeeze(out_t_temp) + torch.squeeze(x_tactile_temp)
        #
        # x_fusion=torch.cat((out_t, out_v), -1)
        self.LSTM0.flatten_parameters()
        RNN_out0, (h_n, h_c) = self.LSTM0(x_visual, None)
        self.LSTM1.flatten_parameters()
        RNN_out1, (h_n, h_c) = self.LSTM1(x_tactile, None)
        for i in range(x_tactile.size(1)):
            fusion_temp=torch.cat((RNN_out0[:,i,:],RNN_out1[:,i,:]),-1)
            fusion_temp=fusion_temp*self.att_early(fusion_temp)
            x_fusion[:,i,:]=fusion_temp

        self.LSTM2.flatten_parameters()
        RNN_out2, (h_n, h_c) = self.LSTM2(x_fusion, None)
        x=torch.cat((RNN_out0[:,-1,:], RNN_out1[:,-1,:],RNN_out2[:,-1,:]), -1)
        # x = torch.cat((RNN_out[:, 0, :], RNN_out[:, 1, :], RNN_out[:, 2, :], RNN_out[:, 3, :], RNN_out[:, 4, :],
        #                RNN_out[:, 5, :], RNN_out[:, 6, :], RNN_out[:, 7, :]), -1)
        # attention = F.softmax(self.att_fc2(self.att_dropout(F.relu(self.att_fc1(x)))),
        #                       dim=1)  # attention mechanism
        # attended = torch.zeros(x.size(0), RNN_out.size(2)).cuda()

        # for i in range(RNN_out.size(1)):
        #     attended += attention[:, i].reshape(x.size(0), 1) * RNN_out[:, i, :]
        x=self.fc_late_0(x)
        x = self.dropout_fc(self.fc_late_1(x))
        return x
class Resnet18Encoder(nn.Module):
    def __init__(self):
        """Load the pretrained ResNet-152 and replace top fc layer."""
        super(Resnet18Encoder, self).__init__()

        self.resnet18=resnet18(pretrained=True)

    def forward(self, x_3d):
        cnn_embed_seq = []
        with torch.no_grad():
            for t in range(x_3d.size(1)):
                # CNNs
                # print(x_3d[:, t, :, :, :].shape)
                x = self.resnet18(x_3d[:, t, :, :, :])  # ResNet

                cnn_embed_seq.append(x)

        # swap time and sample dim such that (sample dim, time dim, CNN latent dim)
        cnn_embed_seq = torch.stack(cnn_embed_seq, dim=0).transpose_(0, 1)
        # cnn_embed_seq: shape=(batch, time_step, input_size)
        # print(cnn_embed_seq.shape)
        # print(cnn_embed_seq.shape)
        return cnn_embed_seq
def conv3D_output_size(img_size, padding, kernel_size, stride):
    # compute output shape of conv3D
    outshape = (np.floor((img_size[0] + 2 * padding[0] - (kernel_size[0] - 1) - 1) / stride[0] + 1).astype(int),
                np.floor((img_size[1] + 2 * padding[1] - (kernel_size[1] - 1) - 1) / stride[1] + 1).astype(int),
                np.floor((img_size[2] + 2 * padding[2] - (kernel_size[2] - 1) - 1) / stride[2] + 1).astype(int))
    return outshape

class CNN3D(nn.Module):
    def __init__(self, t_dim=120, img_x=90, img_y=120, drop_p=0.2, fc_hidden1=256,ch1=32,ch2=48):#, fc_hidden2=128, num_classes=50):
        super(CNN3D, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1= int(fc_hidden1)
        self.drop_p = drop_p
        # self.num_classesclasses = num_classes
        self.ch1, self.ch2 = ch1,ch2
        self.k1, self.k2 = (3, 3, 3), (3, 3, 3)  # 3d kernel size
        self.s1, self.s2 = (1, 1, 1), (1, 1, 1)  # 3d strides
        self.pd1, self.pd2 = (0, 0, 0), (0, 0, 0)  # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        # print(self.conv1_outshape)
        self.conv2_outshape = conv3D_output_size(self.conv1_outshape, self.pd2, self.k2, self.s2)
        # print(self.conv2_outshape)
        # print(type(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2]))
        self.conv1 = nn.Conv3d(in_channels=1, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.conv2 = nn.Conv3d(in_channels=self.ch1, out_channels=self.ch2, kernel_size=self.k2, stride=self.s2,
                               padding=self.pd2)
        self.bn2 = nn.BatchNorm3d(self.ch2)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        self.pool = nn.MaxPool3d(2)
        self.fc_dim = int(self.ch2 * self.conv2_outshape[0] * self.conv2_outshape[1] * self.conv2_outshape[2])
        # print(type(self.fc_hidden1))
        self.fc1 = nn.Linear(self.fc_dim,self.fc_hidden1)  # fully connected hidden layer
        # self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        # self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # Conv 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        # x = self.fc3(x)

        return x

class CNN3D1(nn.Module):
    def __init__(self, t_dim=30, img_x=4, img_y=4, drop_p=0.2, fc_hidden1=64,ch1=8):#, fc_hidden2=128, num_classes=50):
        super(CNN3D1, self).__init__()

        # set video dimension
        self.t_dim = t_dim
        self.img_x = img_x
        self.img_y = img_y
        # fully connected layer hidden nodes
        self.fc_hidden1= int(fc_hidden1)
        self.drop_p = drop_p
        # self.num_classesclasses = num_classes
        self.ch1= ch1
        self.k1 = (3, 3, 3)  # 3d kernel size
        self.s1 = (1, 1, 1)  # 3d strides
        self.pd1= (0, 0, 0) # 3d padding

        # compute conv1 & conv2 output shape
        self.conv1_outshape = conv3D_output_size((self.t_dim, self.img_x, self.img_y), self.pd1, self.k1, self.s1)
        # print(self.conv1_outshape)
        self.conv1 = nn.Conv3d(in_channels=3, out_channels=self.ch1, kernel_size=self.k1, stride=self.s1,
                               padding=self.pd1)
        self.bn1 = nn.BatchNorm3d(self.ch1)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout3d(self.drop_p)
        # self.pool = nn.MaxPool3d(2)
        self.fc_dim = int(self.ch1 * self.conv1_outshape[0] * self.conv1_outshape[1] * self.conv1_outshape[2])
        # print(type(self.fc_hidden1))
        self.fc1 = nn.Linear(self.fc_dim,self.fc_hidden1)  # fully connected hidden layer
        # self.fc2 = nn.Linear(self.fc_hidden1, self.fc_hidden2)
        # self.fc3 = nn.Linear(self.fc_hidden2, self.num_classes)  # fully connected layer, output = multi-classes

    def forward(self, x_3d):
        # Conv 1
        x = self.conv1(x_3d)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.drop(x)
        # # Conv 2
        #         # x = self.conv2(x)
        #         # x = self.bn2(x)
        #         # x = self.relu(x)
        #         # x = self.drop(x)
        # FC 1 and 2
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        # x = self.fc3(x)

        return x

class C3D(nn.Module):
     def __init__(self, v_dim=15, img_xv=256, img_yv=256, drop_p_v=0.2, fc_hidden_v=256, ch1_v=32,ch2_v=48,ch1_t=8,ch2_t=12,t_dim=30, img_xt=4, img_yt=4, drop_p_t=0.2, fc_hidden_t=64,fc_hidden_1=128,num_classes=3):#, fc_hidden2=128, num_classes=50):
        super(C3D, self).__init__()
        self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.tactile_c3d=CNN3D1(t_dim=t_dim, img_x=img_xt, img_y=img_yt, drop_p=drop_p_t, fc_hidden1=fc_hidden_t,ch1=ch1_t,ch2=ch2_t)
        self.fc1 = nn.Linear(fc_hidden_v+fc_hidden_t, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p=drop_p_v

     def forward(self,x_3d_v,x_3d_t):
         # print(x_3d_t.shape,x_3d_v.shape)
         x_v=self.visual_c3d(x_3d_v)
         x_t=self.tactile_c3d(x_3d_t)
         x=torch.cat((x_v,x_t),-1)
         x=F.relu(self.fc1(x))
         x = F.dropout(x, p=self.drop_p, training=self.training)
         x = self.fc2(x)

         return x

class ResC3D(nn.Module):
     def __init__(self,img_size=224,drop_p_v=0.2,visual_dim=1000,ch1_t=8,t_dim=30, img_xt=4, img_yt=4, drop_p_t=0.2, fc_hidden_t=64,fc_hidden_1=128,num_classes=3):#, fc_hidden2=128, num_classes=50):
        super(ResC3D, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.resnet101 = resnet101(pretrained=True)
        self.tactile_c3d=CNN3D1(t_dim=t_dim, img_x=img_xt, img_y=img_yt, drop_p=drop_p_t, fc_hidden1=fc_hidden_t,ch1=ch1_t)
        self.fc1 = nn.Linear(visual_dim+fc_hidden_t, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p=drop_p_v

     def forward(self,x_3d_v,x_3d_t):
         # print(x_3d_t.shape,x_3d_v.shape)
         x_v=self.resnet101(x_3d_v)
         x_t=self.tactile_c3d(x_3d_t)
         # print(x_v.shape,x_t.shape)
         x=torch.cat((x_v,x_t),-1)
         x=F.relu(self.fc1(x))
         x = F.dropout(x, p=self.drop_p, training=self.training)
         x = self.fc2(x)

         return x


class Resnet_v(nn.Module):
    def __init__(self, img_size=224,visual_dim=2048):  # , fc_hidden2=128, num_classes=50):
        super(Resnet_v, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.resnet101 = resnet101(pretrained=True)

    def forward(self, x_3d_v):
        # print(x_3d_t.shape,x_3d_v.shape)
        x = self.resnet101(x_3d_v)
        # x = F.relu(self.fc1(x))
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        # x = self.fc2(x)

        return x

class Resvisual(nn.Module):
    def __init__(self, img_size=224, drop_p_v=0.2, visual_dim=2048,fc_hidden_1=128, num_classes=3):  # , fc_hidden2=128, num_classes=50):
        super(Resvisual, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.resnet101 = resnet101(pretrained=True)
        self.fc1 = nn.Linear(visual_dim , fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v):
        # print(x_3d_t.shape,x_3d_v.shape)
        x = self.resnet101(x_3d_v)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x


class ResC3Dh5(nn.Module):
    def __init__(self, visual_dim=1000, ch1_t=8, t_dim=6, img_xt=4, img_yt=4, drop_p_t=0.2,
                 fc_hidden_v=256,fc_hidden_t=64, fc_hidden_1=128, num_classes=3):  # , fc_hidden2=128, num_classes=50):
        super(ResC3Dh5, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        # self.resnet101 = resnet101(pretrained=True)
        self.tactile_c3d = C3D_UCF(num_classes=3,pretrained=False)
        self.fc1 = nn.Linear(visual_dim+fc_hidden_t, fc_hidden_1)
        # self.fc2=nn.Linear(fc_hidden_v+fc_hidden_t, fc_hidden_1)
        self.fc2= nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_t

    def forward(self, x_3d_v, x_3d_t):
        # print(x_3d_t.shape,x_3d_v.shape)
        # x_v = self.fc1(x_3d_v)
        x_t = self.tactile_c3d(x_3d_t)
        # print(x_v.shape,x_t.shape)
        x = torch.cat((x_3d_v, x_t), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

class ResC3Dh52(nn.Module):
    def __init__(self, visual_dim=1000, ch1_t=8, t_dim=6, img_xt=4, img_yt=4, drop_p_t=0.2,
                 fc_hidden_v=256,fc_hidden_t=64, fc_hidden_1=128, num_classes=3):  # , fc_hidden2=128, num_classes=50):
        super(ResC3Dh52, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        # self.resnet101 = resnet101(pretrained=True)
        self.tactile_c3d = C3D_tactile(pretrained=False)
        self.fc1 = nn.Linear(visual_dim+fc_hidden_t, fc_hidden_1)
        # self.fc2=nn.Linear(fc_hidden_v+fc_hidden_t, fc_hidden_1)
        self.fc2= nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_t

    def forward(self, x_3d_v, x_3d_t):
        # print(x_3d_t.shape,x_3d_v.shape)
        # x_v = self.fc1(x_3d_v)
        x_t = self.tactile_c3d(x_3d_t)
        # print(x_v.shape,x_t.shape)
        x = torch.cat((x_3d_v, x_t), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x


class C3D_tactile(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, pretrained=False,length=10,fc_dim=32):
        super(C3D_tactile, self).__init__()
        self.length=length
        self.fc_dim=fc_dim
        if length == 12:
            self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            #
            self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2,1, 1),padding=(1,0,0))

            self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        if length == 14:
            self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),padding=(1,0,0))
            #
            self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2,1, 1))

            self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        if length == 16:
            self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            #
            self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

            self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        if length == 10:
            self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            #
            self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2,1, 1))
        if length == 8:
            self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            #
            self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(1,1, 1))

        if length == 6:
            self.conv1 = nn.Conv3d(3, 8, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv2 = nn.Conv3d(8, 16, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
            #
            # self.conv3 = nn.Conv3d(16, 32, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            # self.pool3 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
            # self.conv4 = nn.Conv3d(32, 64, kernel_size=(3, 1, 1), padding=(1, 0, 0))
            # self.pool4 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))
        # self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #
        # self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool4 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        #
        # self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        # self.pool5 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 1, 1))
        #
        # self.fc6 = nn.Linear(8192, 4096)
        # self.fc7 = nn.Linear(4096, 4096)
        # self.fc8 = nn.Linear(4096, num_classes)
        #
        # self.dropout = nn.Dropout(p=0.5)
        #
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        if self.length == 6:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            # x = x.view(x.size(0), -1)
            #
            # # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
        if self.length == 8:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            # x = x.view(x.size(0), -1)
            #
            # # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            x = self.relu(self.conv3(x))
            # x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
        elif self.length ==10 :
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            # x = x.view(x.size(0), -1)
            #
            # # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            #
            x = self.relu(self.conv3(x))
            # x = self.relu(self.conv3b(x))
            x = self.pool3(x)
        elif self.length == 12 or self.length == 14 or self.length ==16:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            # x = x.view(x.size(0), -1)
            #
            # # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            #
            x = self.relu(self.conv3(x))
            # x = self.relu(self.conv3b(x))
            x = self.pool3(x)

            x = self.relu(self.conv4(x))
            # x = self.relu(self.conv3b(x))
            x = self.pool4(x)
        # print(x.shape)
        #
        # x = self.relu(self.conv4(x))
        # # x = self.relu(self.conv4b(x))
        # x = self.pool4(x)
        # print(x.shape)
        #
        # x = self.relu(self.conv5a(x))
        # x = self.relu(self.conv5b(x))
        # x = self.pool5(x)
        #
        x = x.view(-1, self.fc_dim)
        # x = self.relu(self.fc6(x))
        # x = self.dropout(x)
        # x = self.relu(self.fc7(x))
        # x = self.dropout(x)
        #
        # logits = self.fc8(x)

        return x

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
class C3D_visual(nn.Module):
    """
    The C3D network.
    """

    def __init__(self, pretrained=False,length=5,img_size=112):
        super(C3D_visual, self).__init__()
        self.img_size=img_size
        if length == 5:
            if img_size == 112:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1,2, 2), stride=(1, 2, 2))

                self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)
            elif img_size == 32:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)
            elif img_size == 64:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)
            elif img_size ==224:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv6a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv6b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)

            elif img_size == 512:
                self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

                self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv6a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv6b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool6 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.conv7a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.conv7b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
                self.pool7 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

                self.fc6 = nn.Linear(8192, 4096)
                self.fc7 = nn.Linear(4096, 4096)

        if length == 6:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2),padding=(1,0,0))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2,2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 7:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2,2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 8:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(2,2, 2), stride=(2, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))
            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 3:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)
        if length == 4:
            self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv3a = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv3b = nn.Conv3d(256, 256, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool3 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))

            self.conv4a = nn.Conv3d(256, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv4b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool4 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

            self.conv5a = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.conv5b = nn.Conv3d(512, 512, kernel_size=(3, 3, 3), padding=(1, 1, 1))
            self.pool5 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2), padding=(0, 1, 1))

            self.fc6 = nn.Linear(8192, 4096)
            self.fc7 = nn.Linear(4096, 4096)

        # self.fc8 = nn.Linear(4096, num_classes)

        self.dropout = nn.Dropout(p=0.5)
        #
        self.relu = nn.ReLU()

        self.__init_weight()

        if pretrained:
            self.__load_pretrained_weights()

    def forward(self, x):
        if self.img_size == 32:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            # x = self.relu(self.conv4a(x))
            # x = self.relu(self.conv4b(x))
            # x = self.pool4(x)
            # # print(x.shape)
            # x = self.relu(self.conv5a(x))
            # x = self.relu(self.conv5b(x))
            # x = self.pool5(x)
            # print(x.shape)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size ==64:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            # x = self.relu(self.conv5a(x))
            # x = self.relu(self.conv5b(x))
            # x = self.pool5(x)
            # print(x.shape)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size ==112:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            # print(x.shape)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size ==224:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            # print(x.shape)
            x = self.relu(self.conv6a(x))
            x = self.relu(self.conv6b(x))
            x = self.pool6(x)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        elif self.img_size == 512:
            x = self.relu(self.conv1(x))
            x = self.pool1(x)
            # print(x.shape)
            x = self.relu(self.conv2(x))
            x = self.pool2(x)
            # print(x.shape)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            x = self.pool3(x)
            # print(x.shape)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))
            x = self.pool4(x)
            # print(x.shape)
            x = self.relu(self.conv5a(x))
            x = self.relu(self.conv5b(x))
            x = self.pool5(x)
            # print(x.shape)
            x = self.relu(self.conv6a(x))
            x = self.relu(self.conv6b(x))
            x = self.pool6(x)
            x = self.relu(self.conv7a(x))
            x = self.relu(self.conv7b(x))
            x = self.pool7(x)
            x = x.view(-1, 8192)
            x = self.relu(self.fc6(x))
            x = self.dropout(x)
            x = self.relu(self.fc7(x))
            logits = self.dropout(x)
        # logits = self.fc8(x)

        return logits

    def __load_pretrained_weights(self):
        """Initialiaze network."""
        corresp_name = {
                        # Conv1
                        "features.0.weight": "conv1.weight",
                        "features.0.bias": "conv1.bias",
                        # Conv2
                        "features.3.weight": "conv2.weight",
                        "features.3.bias": "conv2.bias",
                        # Conv3a
                        "features.6.weight": "conv3a.weight",
                        "features.6.bias": "conv3a.bias",
                        # Conv3b
                        "features.8.weight": "conv3b.weight",
                        "features.8.bias": "conv3b.bias",
                        # Conv4a
                        "features.11.weight": "conv4a.weight",
                        "features.11.bias": "conv4a.bias",
                        # Conv4b
                        "features.13.weight": "conv4b.weight",
                        "features.13.bias": "conv4b.bias",
                        # Conv5a
                        "features.16.weight": "conv5a.weight",
                        "features.16.bias": "conv5a.bias",
                         # Conv5b
                        "features.18.weight": "conv5b.weight",
                        "features.18.bias": "conv5b.bias",
                        # fc6
                        "classifier.0.weight": "fc6.weight",
                        "classifier.0.bias": "fc6.bias",
                        # fc7
                        "classifier.3.weight": "fc7.weight",
                        "classifier.3.bias": "fc7.bias",
                        }

        p_dict = torch.load(Path.model_dir())
        s_dict = self.state_dict()
        for name in p_dict:
            if name not in corresp_name:
                continue
            s_dict[corresp_name[name]] = p_dict[name]
        self.load_state_dict(s_dict)

    def __init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class C3D2(nn.Module):
    def __init__(self, drop_p_v=0.2, img_size=112, visual_dim=4096, visual_length=5,fc_hidden_t=32, tactile_length=10,fc_hidden_1=128, num_classes=3):  # , fc_hidden2=128, num_classes=50):
        super(C3D2, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.visual_c3d = C3D_visual(pretrained=False,length=visual_length,img_size=img_size)
        self.tactile_c3d = C3D_tactile(pretrained=False,length=tactile_length,fc_dim=fc_hidden_t)
        self.fc1 = nn.Linear(visual_dim + fc_hidden_t, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v, x_3d_t):
        # print(x_3d_t.shape,x_3d_v.shape)
        x_v = self.visual_c3d(x_3d_v)
        x_t = self.tactile_c3d(x_3d_t)
        # print(x_v.shape,x_t.shape)
        x = torch.cat((x_v, x_t), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

class C3D_ConvLSTM(nn.Module):
    def __init__(self, drop_p_v=0.2, visual_dim=4096, fc_hidden_t=64, fc_hidden_1=128, num_classes=3):  # , fc_hidden2=128, num_classes=50):
        super(C3D_ConvLSTM, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        # self.visual_c3d = C3D_visual(pretrained=False)
        self.tactile_convlstm = ConvLSTM((4,4), 3, 16, (3,3),1,0.2,batch_first = True, bias = True, return_all_layers = False)
        # self.fc1 = nn.Linear(visual_dim + fc_hidden_t, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v, x_3d_t):
        # print(x_3d_t.shape,x_3d_v.shape)
        # x_v = self.visual_c3d(x_3d_v)
        RNN_out, (h_n, h_c)  = self.tactile_convlstm(x_3d_t)
        x_t=x = RNN_out[:, -1, :].view(-1, 256)
        # print(x_t.shape)
        # print(x_v.shape,x_t.shape)
        x = torch.cat((x_3d_v, x_t), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)

        return x

class C3D_visual_only(nn.Module):
    def __init__(self, drop_p_v=0.2, visual_dim=4096, fc_hidden_1=128, num_classes=3):  # , fc_hidden2=128, num_classes=50):
        super(C3D_visual_only, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.visual_c3d = C3D_visual(pretrained=False,length=5)
        # self.tactile_convlstm = ConvLSTM((4,4), 3, 16, (3,3),1,0.2,batch_first = True, bias = True, return_all_layers = False)
        self.fc1 = nn.Linear(visual_dim, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v):
        # print(x_3d_t.shape,x_3d_v.shape)
        x_v = self.visual_c3d(x_3d_v)
        # RNN_out, (h_n, h_c)  = self.tactile_convlstm(x_3d_t)
        # x_t=x = RNN_out[:, -1, :].view(-1, 256)
        # print(x_t.shape)
        # print(x_v.shape,x_t.shape)
        # x = torch.cat((x_3d_v, x_t), -1)
        x = F.relu(self.fc1(x_v))
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x

class C3D_tactile_only(nn.Module):
    def __init__(self, drop_p_v=0.2, tactile_dim=32, fc_hidden_1=32, num_classes=3):  # , fc_hidden2=128, num_classes=50):
        super(C3D_tactile_only, self).__init__()
        # self.visual_c3d=CNN3D(t_dim=v_dim, img_x=img_xv, img_y=img_yv, drop_p=drop_p_v, fc_hidden1=fc_hidden_v,ch1=ch1_v,ch2=ch2_v)
        self.tactile_c3d = C3D_tactile(pretrained=False,length=10)
        # self.tactile_convlstm = ConvLSTM((4,4), 3, 16, (3,3),1,0.2,batch_first = True, bias = True, return_all_layers = False)
        self.fc1 = nn.Linear(tactile_dim, fc_hidden_1)
        self.fc2 = nn.Linear(fc_hidden_1, num_classes)
        self.drop_p = drop_p_v

    def forward(self, x_3d_v):
        # print(x_3d_t.shape,x_3d_v.shape)
        x_v = self.tactile_c3d(x_3d_v)
        # RNN_out, (h_n, h_c)  = self.tactile_convlstm(x_3d_t)
        # x_t=x = RNN_out[:, -1, :].view(-1, 256)
        # print(x_t.shape)
        # print(x_v.shape,x_t.shape)
        # x = torch.cat((x_3d_v, x_t), -1)
        x = F.relu(self.fc1(x_v))
        # x = F.dropout(x, p=self.drop_p, training=self.training)
        x = self.fc2(x)
        return x