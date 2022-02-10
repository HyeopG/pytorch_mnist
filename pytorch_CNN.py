from torch import nn, optim
from torch.utils.data import dataset
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

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

test_loader = torch.utils.data.DataLoader(dataset=mnist_test,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          drop_last=True)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        )

        self.fc1 = torch.nn.Linear(3*3*128, 625, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        self.layer4 = torch.nn.Sequential(
            self.fc1,
            torch.nn.ReLU())
        self.fc2 = torch.nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        plt_count=1
        plt_count = Draw(x, 2, 1, plt_count, 0)
        plt_count = Draw(x, 1, 1, plt_count, 1)
        out = self.layer1(x)
        plt_count = Draw(out, 2, 32, plt_count, 0)
        plt_count = Draw(out, 1, 32, plt_count, 1)
        out = self.layer2(out)
        plt_count = Draw(out, 2, 64, plt_count, 0)
        plt_count = Draw(out, 1, 64, plt_count, 1)
        out = self.layer3(out)
        plt_count = Draw(out, 2, 128, plt_count, 0)
        plt_count = Draw(out, 1, 128, plt_count, 1)
        out = out.view(out.size(0), -1)
        out = self.layer4(out)
        out = self.fc2(out)
        plt.show()
        return out

# draw Instance
x_draw = 4
y_draw = 3
# image 출력
def Draw(image, num, size, plt_count, order=1):     # image = 이미지, num = 몇개를, plt_count = 어디에, x = x축 크기, y = y축 크기, order = 오름내림차순
    image = image.detach().cpu().numpy()
    x = image.shape[2]
    y = image.shape[3]
    image = image.reshape(-1, x, y, 1)
    draw = []
    for i in range(num):
        if order == 0:    # 앞부터
            count = size * i
            draw.append(image[count])
        else:           # 뒤부터
            count = size * (i+1) - (size-1)
            draw.append(image[-count])

    for i in range(num):
        plt.subplot(x_draw, y_draw, plt_count)        # 3 3 총 9개중 plt_count번째에 출력
        plt_count += 1
        plt.imshow(np.reshape(draw[i], [x, y]), cmap="gray")

    return plt_count

# target이 숫자로 되어있어서 10개의 노드로 변환 ex) 8 -> 0000000010
def NumberToTarget(target, batch):
    target = target.tolist()
    for b in range(batch):
        tar = [0 for i in range(10)]
        tar[target[b]] = 1
        target[b] = tar

    return torch.tensor(target)


model = CNN().to(device)  # MLP모델
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
        img = img.to(device)

        # target 설정
        label = NumberToTarget(label, batch_size)
        label = label.to(device)
        label = label.to(torch.float32)

        optimizer.zero_grad()
        hypothesis = model(img)
        loss = criterion(hypothesis, label)
        loss.backward()
        optimizer.step()

        avg_loss += loss / total_batch

    print('[Epoch: {:>4}] loss = {:>.9}'.format(epoch + 1, avg_loss))

    if avg_loss < 0.01:
        break

print('Learning Finished!')

# test model using test sets
model.eval()
total_batch = len(test_loader)
with torch.no_grad():  # test set으로 데이터를 다룰 때에는 gradient를 주면 안된다.
    avg_accuracy = 0
    for X_test, Y_test in test_loader:
        X_test = X_test.to(device)
        Y_test = Y_test.to(device)

        prediction = model(X_test)
        correct_prediction = torch.argmax(prediction, dim=1) == Y_test  # 결과값이랑 실제값이랑 같은지 확인
        accuracy = correct_prediction.float().mean()  # 평균으로 전체 정확도 확인
        avg_accuracy += accuracy / total_batch

    print('accuracy: {:>.9}'.format(avg_accuracy))


# # weight Save
# PATH = './weights/'
#
# torch.save(model, PATH + 'model.pt')  # 전체 모델 저장
# torch.save(model.state_dict(), PATH + 'model_state_dict.pt')  # 모델 객체의 state_dict 저장
# torch.save({
#     'model': model.state_dict(),
#     'optimizer': optimizer.state_dict()
# }, PATH + 'all.tar')  # 여러 가지 값 저장, 학습 중 진행 상황 저장을 위해 epoch, loss 값 등 일반 scalar값 저장 가능

# ------------------------------------------------------------------------------------------------------------------

# # weight Load
# PATH = './weights/'
#
# model = torch.load(PATH + 'model.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
# model.load_state_dict(torch.load(PATH + 'model_state_dict.pt'))  # state_dict를 불러 온 후, 모델에 저장
#
# checkpoint = torch.load(PATH + 'all.tar')  # dict 불러오기
# model.load_state_dict(checkpoint['model'])
# optimizer.load_state_dict(checkpoint['optimizer'])