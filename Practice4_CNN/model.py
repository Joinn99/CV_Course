import torch.nn as nn

# CNN Classifier for MNIST classification
class ConvClassifier(nn.Module):
    def __init__(self):
        super(ConvClassifier, self).__init__()
        self.conv1 = nn.Sequential( # 1 * 28 * 28 -> 16 * 14 * 14
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(num_features=16),
            nn.LeakyReLU(negative_slope=5e-3),
            nn.MaxPool2d(kernel_size=2)
        ).cuda()
        self.conv2 = nn.Sequential( # 16 * 14 * 14 -> 32 * 7 * 7
            nn.Conv2d(in_channels=16, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.LeakyReLU(negative_slope=5e-3),
            nn.MaxPool2d(kernel_size=2)
        ).cuda()
        self.fcn = nn.Sequential(
            nn.Linear(in_features=64*7*7, out_features=392),
            nn.LeakyReLU(negative_slope=2e-2),
            nn.Linear(in_features=392, out_features=10),
        ).cuda()


    def forward(self, input_x):
        input_x = self.conv1(input_x)
        input_x = self.conv2(input_x)
        input_x = input_x.view(input_x.size(0), -1)
        output = self.fcn(input_x)
        return output
