import torch
import torch.nn as nn

## Extract feature only from LSTM
class extract_Tensor(nn.Module):
    def forward(self,x):
        # Output shape (batch, features, hidden)
        tensor, _ = x
        # Reshape shape (batch, hidden)
        return tensor

class CNN_1D(nn.Module):
    def __init__(self, in_channel, fs=100):
        super(CNN_1D, self).__init__()
        drate = 0.5        
        self.RELU = nn.ReLU()  
        self.features1 = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=int(fs/2), stride=int(fs/16), bias=False, padding=int(fs/4)),
            nn.BatchNorm1d(64),
            self.RELU,
            nn.MaxPool1d(kernel_size=8, stride=8, padding=4),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.MaxPool1d(kernel_size=4, stride=4),
            nn.Dropout(drate)        
        )

        self.features2 = nn.Sequential(
            nn.Conv1d(in_channel, 64, kernel_size=int(fs*4), stride=int(fs/2), bias=False, padding=int(fs*2)),
            nn.BatchNorm1d(64),
            self.RELU,
            nn.MaxPool1d(kernel_size=4, stride=4, padding=2),
            nn.Dropout(drate),

            nn.Conv1d(64, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.Conv1d(128, 128, kernel_size=7, stride=1, bias=False, padding=3),
            nn.BatchNorm1d(128),
            self.RELU,

            nn.MaxPool1d(kernel_size=2, stride=2)
        )
        self.dropout = nn.Dropout(drate)

    def forward(self, x):
        x1 = self.features1(x)
        # print(x1.shape)
        x2 = self.features2(x)
        # print(x2.shape)
        x_concat = torch.cat((x1, x2), dim=2)
        x_concat = self.dropout(x_concat)
        return x_concat



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None,
                 *, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, 1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.se = SELayer(planes, reduction)
        self.downsample = downsample
        self.stride = stride
        

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, pool_size=2):
        super(ResidualBlock, self).__init__()


        self.activation = nn.LeakyReLU()
        self.batchnorm = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1,  padding=1 , bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        self.conv3 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        
#        self.shortcut = nn.Sequential()
        self.maxpool = nn.MaxPool1d(kernel_size=pool_size, stride=pool_size, padding=0) 

        self.shortcut = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding=1, bias=False)
        )
    
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.batchnorm(out)
        out = self.activation(out) 
        out = self.dropout(out)
        out = self.conv3(out)
        out = self.activation(out)
        out = self.maxpool(out)
        if self.shortcut:
            residual = self.shortcut(x)
            residual = self.maxpool(residual) 
        out += residual
        out = self.activation(out) 
        
        return out

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, num_layer = 3, drate = 0.5):
        super(DilatedConvBlock, self).__init__()
        ## Shape이 변하면 안되므로, same padding 맞춰주는 작업 진행
        ## padding = dilation * (kernel - stride) / 2
        
        # self.dconv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding = 2 * 3, dilation=2)
        # self.dconv3 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding = 4 * 3, dilation=4)
        # self.dconv4 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding = 8 * 3, dilation=8)
        # self.dconv5 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding = 16 * 3, dilation=16)
        # self.dconv6 = nn.Conv1d(out_channels, out_channels, kernel_size=7, padding = 32 * 3, dilation=32)
        self.num_layer = num_layer
        
        self.dconv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding = 3, dilation=1)
        self.dropout = nn.Dropout(p=drate)
        self.activation = nn.ReLU()
        if self.num_layer > 1:
            self.dconv = self._make_layer(out_channels=out_channels, num_layer=num_layer)

    def _make_layer(self, out_channels, num_layer = 3):
        layers = []
        for i in range(1, num_layer):
            dilation = 2**i
            layers.append(nn.Conv1d(out_channels, out_channels, kernel_size=7, padding = dilation*3, dilation=dilation))
            layers.append(self.activation)
            layers.append(self.dropout)
        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.dconv1(x)
        x = self.activation(x) 
        x = self.dropout(x)
        if self.num_layer > 1:
            x = self.dconv(x)
        return x
        # x = self.dconv2(x)
        # x = self.activation(x) 
        # x = self.dropout(x)

        # x = self.dconv3(x)
        # x = self.activation(x) 
        # x = self.dropout(x)

        # x = self.dconv4(x)
        # x = self.activation(x) 
        # x = self.dropout(x)
        
        # x = self.dconv5(x)
        # x = self.activation(x) 
        # x = self.dropout(x)

        # x = self.dconv6(x)
        # x = self.activation(x) 
        # x = self.dropout(x)
        # # x = self.activation(x) 
        # return x


if __name__=="__main__":
    fs = 25
    inputs = torch.rand((5,1,fs*30))
    print(inputs.shape)
    #model = CNN_1D(in_channel=1, fs=fs)
    TCN = DilatedConvBlock(in_channels=1, out_channels=128, drate = 0.2)

    #model = ResNetLayer(in_channels=1, out_channels=256)
    #print(model(inputs).shape)
    print(TCN(inputs).shape)