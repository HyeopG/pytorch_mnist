from torch import nn, optim
from torch.utils.data import dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms

import torch.nn.init

# GPU, CPU 사용
device = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

# parameters
learning_rate = 0.0001
training_epochs = 15
batch_size = 100

# MNIST dataset
mnist_train = dsets.MNIST(root="MNIST_data/",
                          train=True,
                          transform=transforms.ToTensor(),
                          download=True)

mnist_test = dsets.MNIST(root="MNIST_data/",
                         train=False,
                         transform=transforms.ToTensor(),
                         download=True)

data_loader = torch.utils.data.DataLoader(dataset=mnist_train,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)


class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.layer1 = nn.Linear(784, 300, bias=True)  # 입력층(784) -> 은닉1층(300)
        self.layer2 = nn.Linear(300, 100, bias=True)  # 은닉1층(300) -> 은닉2층(100)
        self.layer3 = nn.Linear(100, 10, bias=True)  # 은닉2층(100) -> 출력층(10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.view(-1, 28*28)

        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        out = self.layer3(out)
        return out


model = MLP().to(device)  # MLP모델
criterion = nn.MSELoss()  # loss 구하기

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train model
total_batch = len(data_loader)
model.train()  # set the model to train mode
print("Learning Started")

def targetUpdate(target, batch):
    for b in range(batch):
        tar = [0 for i in range(10)]
        print(target[b])
        tar[target[b]] = 1
        target[b] = tar


for epoch in range(training_epochs):
    avg_loss = 0

    for img, label in data_loader:
        # image is already size of (28x28), no reshape
        # label is not one-hot encoded
        img = img.to(device)

        label = label.to(device)
        label = label.to(torch.float32)
        #targetUpdate(label, batch_size)

        optimizer.zero_grad()   # 이걸 빼먹으면 학습이 안됌
        hypothesis = model(img)
        # hypothesis = torch.max(hypothesis, dim=1)[1]
        # hypothesis = hypothesis.to(torch.float32)
        print(hypothesis)
        print(label)
        loss = criterion(hypothesis, label.view(-1, 1))
        print(loss)
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch

    print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_loss))

print('Learning Finished!')
