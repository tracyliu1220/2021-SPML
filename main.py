from vgg import VGG
import torch
import torchvision
import torchvision.transforms as transforms
from fgsm import FGSM
device = torch.device("cuda" if (torch.cuda.is_available()) else "cpu")
eps = 0.1
model = VGG()
model = model.to(device)
model.eval()
attack = FGSM(model,eps)
transform = transforms.Compose(
    [transforms.ToTensor()])
     # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

batch_size = 4

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,shuffle=False, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,shuffle=False, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
# dataiter = iter(trainloader)
# images, labels = dataiter.next()

# out = model(images)
# print(out)
correct = 0
total = 0
criterion = torch.nn.CrossEntropyLoss()
# since we're not training, we don't need to calculate the gradients for our outputs
# with torch.no_grad():
for i,data in enumerate(testloader):
    if i == 500:
        break
    images, labels = data
    images, labels = images.to(device),labels.to(device)
    delta = torch.zeros(images.size(0),3).to(device)
    adv_imgs = attack.forward(images,labels,delta)
    # delta = torch.full((images.size(0),3),0.1)
    # calculate outputs by running images through the network
    # delta.requires_grad = True
    outputs = model(adv_imgs,delta)
    loss = criterion(outputs,labels)
    # loss.backward()
    # optimizer.step()
    # grad = torch.autograd.grad(loss, delta,
    #                                retain_graph=False, create_graph=False,allow_unused = True)[0]
    # print(grad)
    # the class with the highest energy is what we choose as prediction
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))
