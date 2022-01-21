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
        out = self.layer1(x)
        out = self.sigmoid(out)
        out = self.layer2(out)
        out = self.sigmoid(out)
        out = self.layer3(out)
        return out


# target이 숫자로 되어있어서 10개의 노드로 변환 ex) 8 -> 0000000010
def NumberToTarget(target, batch):
    target = target.tolist()
    for b in range(batch):
        tar = [0 for i in range(10)]
        tar[target[b]] = 1
        target[b] = tar

    return torch.tensor(target)


model = MLP().to(device)  # MLP모델
criterion = nn.MSELoss()  # loss 구하기

optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# train model
total_batch = len(data_loader)
model.train()  # set the model to train mode
print("Learning Started")

for epoch in range(training_epochs):
    avg_loss = 0

    for img, label in data_loader:
        # img 설정
        img = img.view(-1, 28*28).to(device)

        # target 설정
        label = NumberToTarget(label, batch_size)
        label = label.to(device)
        label = label.to(torch.float32)

        optimizer.zero_grad()   # 이걸 빼먹으면 학습이 안됌
        hypothesis = model(img)
        loss = criterion(hypothesis, label)
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch

    print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_loss))

print('Learning Finished!')

# test model using test sets
model.eval()
with torch.no_grad():  # test set으로 데이터를 다룰 때에는 gradient를 주면 안된다.
    X_test = mnist_test.data.view(-1, 28 * 28).float().to(device)
    Y_test = mnist_test.targets.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, dim=1) == Y_test  # 결과값이랑 실제값이랑 같은지 확인
    accuracy = correct_prediction.float().mean()  # 평균으로 전체 정확도 확인
    print('accuracy:', accuracy.item())