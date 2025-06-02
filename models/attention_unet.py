import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels)
        
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = self.shortcut(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += residual
        return F.relu(out)

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(AttentionGate, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        return x * psi

class AttentionUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[32, 64, 128, 256]):
        super(AttentionUNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.attention_gates = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder
        in_feat = in_channels
        for feat in features:
            self.encoder.append(ResidualBlock(in_feat, feat))
            in_feat = feat

        # Bottleneck
        self.bottleneck = ResidualBlock(features[-1], features[-1] * 2)

        # Decoder
        features = features[::-1]  # Reverse the features list for decoder
        for idx, feat in enumerate(features):
            # Upsampling and concatenation
            self.decoder.append(
                nn.ConvTranspose2d(
                    feat * 2 if idx == 0 else feat * 2,
                    feat,
                    kernel_size=2,
                    stride=2
                )
            )
            # Attention Gate
            self.attention_gates.append(
                AttentionGate(F_g=feat, F_l=feat, F_int=max(feat//4, 8))
            )
            # Decoder block after concatenation
            self.decoder.append(
                ResidualBlock(feat * 2, feat)
            )

        # Final convolution
        self.final_conv = nn.Conv2d(features[-1], out_channels, kernel_size=1)

    def forward(self, x):
        # Encoder
        skip_connections = []
        for encoder in self.encoder:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]  # Reverse for decoder
        for idx in range(0, len(self.decoder), 2):
            # Upsample
            x = self.decoder[idx](x)
            skip = skip_connections[idx//2]

            # Apply attention gate
            attention_skip = self.attention_gates[idx//2](x, skip)
            
            # Concatenate
            x = torch.cat((attention_skip, x), dim=1)
            
            # Apply decoder block
            x = self.decoder[idx+1](x)

        return self.final_conv(x)

if __name__ == "__main__":
    # Test the model
    model = AttentionUNet()
    x = torch.randn((2, 3, 256, 256))
    print(f"Input shape: {x.shape}")
    output = model(x)
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}") 