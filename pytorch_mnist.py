import torch
from torchvision import transforms

if torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')

# define transformations
data_transform = transforms.Compose([transforms])

