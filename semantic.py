import torch
from convertor import Convertor
import random

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

class Semantic():
    def __init__(self, model, N=1000):
        self.model = model
        self.N = N
        self.convertor = Convertor()

    def gen_adv(self, images, labels):
        # detach the sources
        images = images.clone().detach().to(device)
        labels = labels.clone().detach().to(device)

        outputs = self.model(images)
        _, ori_predicted = torch.max(outputs.data, 1)

        delta = torch.zeros(images.size(0), 3).to(device)
            
        ori_hsv_images = self.convertor.rgb_to_hsv(images)

        test = 0
        flag = torch.zeros(images.size(0)).to(device).int()

        adv_delta = torch.zeros(images.size(0), 3).to(device)
        
        for i in range(self.N):
            test = i

            d_h = random.uniform(0, 1)
            d_s = random.uniform(-i / self.N, i / self.N)

            delta[:, 0] = d_h
            delta[:, 1] = d_s
        
            # convert rgb to hsv, apply delta and convert back
            hsv_images = self.convertor.apply_delta(ori_hsv_images, delta)
            images = self.convertor.hsv_to_rgb(hsv_images)

            # go through the model
            images = self.convertor.normalize(images)
            outputs = self.model(images)

            _, predicted = torch.max(outputs.data, 1)
            diff = (ori_predicted != predicted).int()

            # flag = 0 & diff = 1
            choose = torch.bitwise_and((1 - flag), diff)

            condition_choose = torch.kron(choose, torch.ones(3).to(device).int()).reshape(images.size(0), -1)

            adv_delta = torch.where(condition_choose == 1, delta, adv_delta)

            flag = torch.bitwise_or(choose, flag)

            if flag.sum() == images.size(0):
                break

        # print(test)

        adv_images = self.convertor.apply_delta(ori_hsv_images, adv_delta)
        adv_images = self.convertor.hsv_to_rgb(adv_images).detach()

        return adv_images



