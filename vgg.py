import torch
from rgb_hsv import RGB_HSV
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
class VGG(torch.nn.Module):
    def __init__(self):
        super(VGG,self).__init__()
        self.convertor = RGB_HSV()
        self.VGG16 = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
    def hsv_rgb(self,input):
        return self.convertor.hsv_to_rgb(input)

    def rgb_hsv(self,input):
        return self.convertor.rgb_to_hsv(input)

    def forward(self, input, delta):#delta --> hue's change(dimension = (1,3,32,32))
        b = torch.arange(input.size(0)).long().to(device)
        color0 = torch.LongTensor([0 for i in range(input.size(0))]).to(device)
        color1 = torch.LongTensor([1 for i in range(input.size(0))]).to(device)
        color2 = torch.LongTensor([2 for i in range(input.size(0))]).to(device)

        out = self.rgb_hsv(input)

        _a = delta[b, color0].reshape(-1, 1)
        _b = torch.ones(32 * 32).to(device)
        _d = torch.kron(_a, _b).reshape(-1, 32, 32).to(device)
        out[b, color0] += _d[b]

        _a = delta[b, color1].reshape(-1, 1)
        _b = torch.ones(32 * 32).to(device)
        _d = torch.kron(_a, _b).reshape(-1, 32, 32).to(device)
        out[b, color1] += _d[b]

        _a = delta[b, color2].reshape(-1, 1)
        _b = torch.ones(32 * 32).to(device)
        _d = torch.kron(_a, _b).reshape(-1, 32, 32).to(device)
        out[b, color2] += _d[b]


        # out = self.hsv_rgb(out+eps)
        # print(out)
        out = self.hsv_rgb(out)
        # out = input
        
        out[b, color0] -= 0.4914
        out[b, color0] /= 0.2023
        
        out[b, color1] -= 0.4822
        out[b, color1] /= 0.1994

        out[b, color2] -= 0.4465
        out[b, color2] /= 0.2010
        
        out = self.VGG16(out)
        return out
