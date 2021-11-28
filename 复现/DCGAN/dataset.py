from torchvision import transforms, datasets

facesdataset = datasets.ImageFolder(root="./data/faces", transform=transforms.ToTensor())