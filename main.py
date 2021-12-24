import torch
import torchvision
import torchvision.transforms as transforms
from fgsm import FGSM
from convertor import Convertor
from torchvision.utils import save_image

device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")

# ==== parameters ====
eps = torch.Tensor([0.3, 0.15, 0.2]).to(device)
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
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=True, num_workers=2)

correct = 0
total = 0

for i,data in enumerate(testloader):
    # if i == 500:
    #     break
    images, labels = data
    images, labels = images.to(device), labels.to(device)

    save_image(images[0], 'test.png')
    if i == 0:
        print(labels[0])

    # generate adversarial images
    adv_imgs = attack.gen_adv(images,labels)
    if i == 0:
        save_image(adv_imgs[0], 'test_adv.png')

    # go through the target model
    if i == 0:
        outputs = model(convertor.normalize(images))
        print('origin:', torch.max(outputs.data, 1)[1][0])
    outputs = model(convertor.normalize(adv_imgs))
    if i == 0:
        print('adv:', torch.max(outputs.data, 1)[1][0])
    
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
