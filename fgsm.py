import torch
from convertor import Convertor

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

class FGSM():
    def __init__(self, model, eps):
        self.model = model
        self.eps = eps
        self.criterion = torch.nn.CrossEntropyLoss()
        self.convertor = Convertor()
        pass

    def gen_adv(self, images, labels):
        # detach the sources
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        delta = torch.zeros(images.size(0), 3).to(device)
        delta.requires_grad = True
        
        # convert rgb to hsv, apply delta and convert back
        hsv_images = self.convertor.rgb_to_hsv(images)
        hsv_images = self.convertor.apply_delta(hsv_images, delta)
        images = self.convertor.hsv_to_rgb(hsv_images)

        # go through the model
        images = self.convertor.normalize(images)
        outputs = self.model(images)
        
        # get the new delta
        loss = self.criterion(outputs, labels)
        grad = torch.autograd.grad(loss, delta,
               retain_graph=False, create_graph=False)[0]
        delta = self.eps * grad.sign()

        adv_images = self.convertor.apply_delta(hsv_images, delta)
        adv_images = self.convertor.hsv_to_rgb(adv_images).detach()

        return adv_images



