import torch
from rgb_hsv import RGB_HSV
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
class FGSM():
    def __init__(self,model,eps): #eps is delta's limit
        self.model = model
        self.eps = eps
        self.criterion = torch.nn.CrossEntropyLoss()
        self.convertor = RGB_HSV()
    def get_delta(self,hsv_images,labels,delta):
        
        delta.requires_grad = True

        output = self.model(hsv_images,delta)
        loss = self.criterion(output,labels)
        grad = torch.autograd.grad(loss, delta,
               retain_graph=False, create_graph=False,allow_unused = True)[0]
        delta = self.eps * grad.sign()
        return delta

    def add_delta(self, hsv_images, delta):
        b = torch.arange(hsv_images.size(0)).long().to(device)
        color0 = torch.LongTensor([0 for i in range(hsv_images.size(0))]).to(device)
        color1 = torch.LongTensor([1 for i in range(hsv_images.size(0))]).to(device)
        color2 = torch.LongTensor([2 for i in range(hsv_images.size(0))]).to(device)

        _a = delta[b, color0].reshape(-1, 1)
        _b = torch.ones(32 * 32).to(device)
        _d = torch.kron(_a, _b).reshape(-1, 32, 32).to(device)
        hsv_images[b, color0] += _d[b]

        _a = delta[b, color1].reshape(-1, 1)
        _b = torch.ones(32 * 32).to(device)
        _d = torch.kron(_a, _b).reshape(-1, 32, 32).to(device)
        hsv_images[b, color1] += _d[b]

        _a = delta[b, color2].reshape(-1, 1)
        _b = torch.ones(32 * 32).to(device)
        _d = torch.kron(_a, _b).reshape(-1, 32, 32).to(device)
        hsv_images[b, color2] += _d[b]

        return hsv_images

    def forward(self,images,labels,delta):
        # get delta
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        delta = delta.clone().detach().to(device)
        new_delta = self.get_delta(images, labels, delta)
        # image to hsv -> hsv + delta -> rgb
        img = self.convertor.rgb_to_hsv(images)
        img = self.add_delta(img, new_delta)
        img = self.convertor.hsv_to_rgb(img)
        return img.detach()


