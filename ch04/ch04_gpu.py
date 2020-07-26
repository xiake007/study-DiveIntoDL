import torch
import torch.nn as nn

def test_gpu_info():
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.current_device())
    print(torch.cuda.get_device_name())

def test_tensor():
    x=torch.tensor([1,2,3])
    print(x)
    x=x.cuda()
    print(x)
    print(x.device)

def test_create_gpu_tensor():
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # print(device)
    x=torch.tensor([1,2,3],device=device)
    print(x)
    x=torch.tensor([1,2,3]).to(device)
    print(x)
    y=x**2
    print(y)

if __name__=='__main__':
    # test_gpu_info()
    # test_tensor()
    test_create_gpu_tensor()

