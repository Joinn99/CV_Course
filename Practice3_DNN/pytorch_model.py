import torch
import torch.utils.data
import numpy as np
from sklearn.datasets import fetch_openml

torch.multiprocessing.set_start_method("spawn")

class MNISTClassifier:
    def __init__(self):
        vec_x, vec_y = fetch_openml('mnist_784', version=1, return_X_y=True)
        label_y = np.eye(10)[vec_y.astype(int)]
        self.train_tensor = torch.from_numpy(
            vec_x[:60000, :] / 256).float().cuda()
        self.train_label = torch.from_numpy(
            label_y[:60000].astype(int)).float().cuda()
        self.eval_tensor = torch.from_numpy(
            vec_x[60000:, :] / 256).float().cuda()
        self.eval_label = torch.from_numpy(
            label_y[60000:].astype(int)).float().cuda()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(784, 392),
            torch.nn.Sigmoid(),
            torch.nn.Linear(392, 196),
            torch.nn.Sigmoid(),
            torch.nn.Linear(196, 49),
            torch.nn.Sigmoid(),
            torch.nn.Linear(49, 10),
            torch.nn.Softmax(dim=None)).cuda()
        self.torch_dataset = torch.utils.data.TensorDataset(
            self.train_tensor, self.train_label)
        self.loader = torch.utils.data.DataLoader(
            dataset=self.torch_dataset,         # torch TensorDataset format
            batch_size=60,                      # mini batch size
            shuffle=True,
            num_workers=0,
        )
        print("Initizlized.")

    def train(self, epoch_range):
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.002)
        # optimizer = torch.optim.ASGD(self.net.parameters(), lr=0.01)
        train_lossfunc = torch.nn.MSELoss()
        for epoch in range(epoch_range):
            for step, (batch_x, batch_y) in enumerate(self.loader):
                pred = self.net(batch_x)
                loss = train_lossfunc(pred, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if not step % 10:
                    self.net.eval()
                    predict_label = self.net(self.eval_tensor)
                    eval_index = torch.max(self.eval_label, 1)[1].squeeze()
                    predict_index = torch.max(predict_label, 1)[1].squeeze()
                    accuracy = torch.sum(
                        eval_index == predict_index).float() / predict_index.size(0)
                    print('Epoch {:2d}'.format(epoch) + ' | Step {:3d}'.format(step), end=' | ')
                    print('Acc: {:.4f}'.format(accuracy) + ' | Loss: {:.4f}'.format(loss.float()))
                    self.net.train()
    def predict(self):
        pass


if __name__ == "__main__":
    CLF = MNISTClassifier()
    CLF.train(1)
    