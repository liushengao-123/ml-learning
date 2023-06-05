import torch
import torch.nn as nn
# from torchsummary import summary

class Module1(nn.Module):
    def __init__(self):
        super(Module1, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(3, 48, 11, 4, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(48, 128, 5, 1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),
        )
    def forward(self,x):
        return self.layers(x)

class Module2(nn.Module):
    def __init__(self):
        super(Module2, self).__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(256, 192, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 192, 3, 1, padding=1),
            nn.ReLU(),
            nn.Conv2d(192, 128, 3, 1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2,2)
        )
    def forward(self,x):
        return self.layers(x)

class Module3(nn.Module):
    def __init__(self):
        super(Module3, self).__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(9216,4096),
            nn.ReLU(),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Linear(4096,1000)
        )

    def forward(self,x):
        return self.layers(x)



if __name__ == '__main__':
    device_cpu = torch.device("cpu")
    device_gpu = torch.device("cuda:0")

    x = torch.randn(8,3,224,224)
    mm11 = Module1().to(device_gpu)
    mm12 = Module1().to(device_cpu)
    x11_out = mm11(x.to(device_gpu))
    x12_out = mm12(x.to(device_cpu))
    # print(x11_out)
    # print(x12_out.shape)
    x2 = torch.concat([x11_out.to(device_gpu), x12_out.to(device_gpu)], dim=1)
    # print(x2.shape)
    mm21 = Module2().to(device_gpu)
    mm22 = Module2().to(device_cpu)
    x21_out = mm21(x2.to(device_gpu))
    x22_out = mm22(x2.to(device_cpu))
    # print(x21_out.shape)
    # print(x22_out.shape)

    x3 = torch.concat([x21_out.to(device_gpu), x22_out.to(device_gpu)], dim=1)
    # x3 = x3.view(-1, 9216)
    # print(x3.shape)
    mm3 = Module3().to(device_gpu)
    x3_out = mm3(x3.to(device_gpu))

    print(x3_out.shape)

    # print(summary(mm11, input_size=(3, 224, 224), device="cuda"))
    # print(summary(mm12, input_size=(3, 224, 224), device="cpu"))
    # print(summary(mm21, input_size=(256, 13, 13), device="cuda"))
    # print(summary(mm22, input_size=(256, 13, 13), device="cpu"))
    # print(summary(mm3, input_size=(9216, ), device="cuda"))


###################################################################
###################################################################
###################################################################

# class MyAlexNet(nn.Module):
#     def __init__(self):
#         super(MyAlexNet, self).__init__()
#         self.layers = nn.Sequential(
#             nn.Conv2d(3,96,11,4,padding=0),
#             nn.ReLU(),
#             nn.LocalResponseNorm(5),
#             nn.MaxPool2d(2,2),
#             nn.Conv2d(96,256,5,1,padding=2),
#             nn.ReLU(),
#             nn.LocalResponseNorm(5),
#             nn.MaxPool2d(2,2),
#             nn.Conv2d(256,384,3,1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(384,384,3,1,padding=1),
#             nn.ReLU(),
#             nn.Conv2d(384,256,3,1,padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2,2),
#             nn.Flatten(),
#             nn.Linear(256*6*6,4096),
#             nn.ReLU(),
#             nn.Linear(4096,4096),
#             nn.ReLU(),
#             nn.Linear(4096,1000)
#         )
#
#     def forward(self, x):
#         return self.layers(x)
#
#
# if __name__ == '__main__':
#     #GPU方式
#     device_gpu=torch.device("cuda:0")
#     x = torch.randn(8,3,224,224).to(device_gpu)
#     module = MyAlexNet().to(device_gpu)
#     x_out = module(x)
#     print(summary(module,input_size=(3,224,224)))

    #CPU方式
    # x = torch.randn(8,3,224,224)
    # module = MyAlexNet()
    # x_out = module(x)
    # print(summary(module,input_size=(3,224,224),device="cpu"))

############################################################