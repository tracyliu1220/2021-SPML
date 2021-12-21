import torch
import torchvision
import torchvision.transforms as transforms
from fgsm import FGSM
from convertor import Convertor

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# ==== parameters ====
eps = 0.1
batch_size = 4

# ==== model ====
model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar10_vgg16_bn", pretrained=True)
model = model.to(device)
model.eval()

# ==== attack method ====
attack = FGSM(model, eps)

# ==== convertor ====
convertor = Convertor()

# ==== loaders ====
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)

correct = 0
total = 0

for i,data in enumerate(testloader):
    if i == 500:
        break
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    # generate adversarial images
    adv_imgs = attack.gen_adv(images,labels)

    # go through the target model
    outputs = model(convertor.normalize(adv_imgs))
    
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
