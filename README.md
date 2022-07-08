# CNN 분석하기
***
### 분석할 이미지
![image](https://user-images.githubusercontent.com/87799990/177898815-78c53e61-02ed-47f5-91cc-39e8a80bf743.png)
***
### 1번 Conv2d의 필터들 (32개)
![image](https://user-images.githubusercontent.com/87799990/177898993-58acf3b4-c489-4448-a66f-7dd4279add00.png)
***
### 1번 Conv2d를 진행한 결과물
![image](https://user-images.githubusercontent.com/87799990/177899054-350c5a8e-ca01-4ebf-bfcd-070368f0b53c.png)
***
### 1번 PoolMax2d (28,28) -> (14,14)
![image](https://user-images.githubusercontent.com/87799990/177899106-dc64cde2-b0ce-40e2-bc20-a6ead1a052ce.png)
***
### 2번 Conv2d의 필터들 (64개)
![image](https://user-images.githubusercontent.com/87799990/177899129-51c13171-5ccd-47d1-85a4-7a1ec3eec81d.png)
***
### 2번 Conv2d를 진행한 결과물
![image](https://user-images.githubusercontent.com/87799990/177899171-81b07ea8-2fca-4025-a6b8-0079bf4f44ac.png)
***
### 2번 PoolMax2d (14,14) -> (7,7)
![image](https://user-images.githubusercontent.com/87799990/177899203-90004034-b18e-4a51-a48d-bb7609e79381.png)
***
### 3번 Conv2d 필터 (128개)
![image](https://user-images.githubusercontent.com/87799990/177899246-020f3da4-2d43-46e7-8555-31ed5f66f2a3.png)
***
### 3번 Conv2d를 진행한 결과물
![image](https://user-images.githubusercontent.com/87799990/177899262-005ba8a7-eff6-4742-8461-275a0a023acf.png)
***
### 3번 PoolMax2d (7,7) -> (3,3)
![image](https://user-images.githubusercontent.com/87799990/177899309-2e1e42e5-a18e-43e7-b9d9-2574e658a569.png)
***
## 필터
![image](https://user-images.githubusercontent.com/87799990/177899458-73ce8db7-fc97-49b6-be0c-40289490862e.png)
***
## 위의 필터를 Pooling 없이 진행한 이미지
![image](https://user-images.githubusercontent.com/87799990/177899493-3056d569-f6e5-4a52-9d8d-655210065c48.png)

*** 
# Pytorch 딥러닝 함수 효율조사
### (MSE + Adam), (MSE + SGD), (CrossEntropy + Adam), (CrossEntropy + SGD)
![image](https://user-images.githubusercontent.com/87799990/177899801-327cf44c-7b3c-4024-afa5-d28109b43c80.png)
***
## Loss를 그래프로 나타낸 값
### 4가지 전부
![image](https://user-images.githubusercontent.com/87799990/177899866-ca36a82f-f698-4171-83b6-e36a645c28ce.png)
***
### 4가지 중 가장 높게 있는 Cross + SGD를 제외한 그래프
![image](https://user-images.githubusercontent.com/87799990/177899907-2be9f174-bc45-436c-9abf-13df1a3dc357.png)
# 결론
## 학습이 잘 되는 순서 : (MSE+Adam) -> (Cross + Adam) -> (MSE+SGD) -> (Cross + SGD)
## 가장 극적인 그래프 곡선 : (Cross + Adam) -> (MSE+Adam) -> (Cross + SGD) -> (MSE+SGD)
