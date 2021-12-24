import torch
from convertor import Convertor

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

class PGD():
    def __init__(self, model, eps, steps=40, alpha=0.02):
        self.model = model
        self.eps = eps
        self.criterion = torch.nn.CrossEntropyLoss()
        self.convertor = Convertor()
        self.steps = steps
        self.alpha = alpha

    def gen_adv(self, images, labels):
        # detach the sources
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)
        delta = torch.zeros(images.size(0), 3).to(device)
        
        # convert rgb to hsv, apply delta and convert back
        ori_hsv_images = self.convertor.rgb_to_hsv(images)

        # for loop
        for _ in range(self.steps):
            delta.requires_grad = True
            hsv_images = ori_hsv_images.clone().detach().to(device)
            hsv_images = self.convertor.apply_delta(hsv_images, delta)
            images = self.convertor.hsv_to_rgb(hsv_images)
            images = self.convertor.normalize(images)
            outputs = self.model(images)
            
            # get the new delta
            loss = self.criterion(outputs, labels)
            grad = torch.autograd.grad(loss, delta,
                   retain_graph=False, create_graph=False)[0]

            delta = delta.detach() + self.alpha * grad.sign()
            delta = torch.clamp(delta, min=-self.eps, max=self.eps)

        adv_images = self.convertor.apply_delta(ori_hsv_images, delta)
        adv_images = self.convertor.hsv_to_rgb(adv_images).detach()

        return adv_images



