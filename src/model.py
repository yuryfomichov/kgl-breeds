import torch.nn as nn
import math
import torchvision.models as models

class BreedsModel(nn.Module):
    def __init__(self, num_classes=120):
        super(BreedsModel, self).__init__()
        vgg = models.vgg19(pretrained=True)
        self.features = vgg.features
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, 512),
            nn.ReLU(True),
            nn.BatchNorm1d(512),
            nn.Linear(512, num_classes),
        )
        self._require_grad_false()
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            # if isinstance(m, nn.Conv2d):
            #     n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            #     m.weight.data.normal_(0, math.sqrt(2. / n))
            #     if m.bias is not None:
            #         m.bias.data.zero_()
            # elif isinstance(m, nn.BatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            if isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def _require_grad_false(self):
        for p in self.features.parameters():
            p.requires_grad = False