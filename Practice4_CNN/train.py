import torch
import torchvision
from PIL import Image, ImageDraw
import model

# Parameters
TRAIN = False
LR = 5e-6
EPOCH = 2
BATCH_SIZE = 300

# Load MNIST datasets
TRAIN_DATA = torchvision.datasets.MNIST(
    root='./Practice4_CNN/',
    train=True,                                     # this is training data
    transform=torchvision.transforms.ToTensor(),
    # torch.FloatTensor of shape (C x H x W) and normalize in the range [0.0, 1.0]
    download=False,
)

TEST_DATA = torchvision.datasets.MNIST(
    root='./Practice4_CNN/',
    train=False,
)  # shape from (2000, 28, 28) to (2000, 1, 28, 28), value in range(0,1)
TEST_X = (torch.unsqueeze(TEST_DATA.data, dim=1).type(
    torch.FloatTensor) / 255.).cuda()
TEST_Y = TEST_DATA.targets.cuda()

TRAIN_LOADER = torch.utils.data.DataLoader(
    dataset=TRAIN_DATA, batch_size=BATCH_SIZE, shuffle=True)

# Train function


def train(use_model=False):
    clf = model.ConvClassifier().cuda()
    if use_model:
        clf.load_state_dict(torch.load('./Practice4_CNN/Model/params.pkl'))
    optimizer = torch.optim.Adam(clf.parameters(), lr=LR)
    lossfunc = torch.nn.CrossEntropyLoss()
    for epoch in range(EPOCH):
        for step, (batch_x, batch_y) in enumerate(TRAIN_LOADER):
            output = clf(batch_x.cuda())
            loss = lossfunc(output, batch_y.cuda())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if not (step + 1) % 50:
                clf.eval()
                accu = calculate_accuracy(clf=clf)
                print('Epoch {:2d}'.format(epoch) +
                      ' | Step {:3d}'.format(step), end=' | ')
                print('Acc: {:.4f}'.format(accu) +
                      ' | Loss: {:.4f}'.format(loss.float()))
                clf.train()
    torch.save(clf.state_dict(), './Practice4_CNN/Model/params.pkl')


def calculate_accuracy(clf):
    pred_y = torch.zeros_like(TEST_Y)
    for index in range(20):
        output = clf(TEST_X[index * 500:(index + 1) * 500])
        pred_y[index * 500:(index + 1) *
               500] = torch.max(output, dim=1)[1].squeeze()
    accuracy = torch.sum(pred_y == TEST_Y).float() / TEST_Y.size(0)
    return accuracy

# Find the unmatched results


def find_incorrect(clf):
    pred_y = torch.zeros_like(TEST_Y)
    for index in range(20):
        output = clf(TEST_X[index * 500:(index + 1) * 500])
        pred_y[index * 500:(index + 1) *
               500] = torch.max(output, dim=1)[1].squeeze()
    incorrect = torch.nonzero(pred_y != TEST_Y).squeeze()
    accuracy = torch.sum(pred_y == TEST_Y).float() / TEST_Y.size(0)
    print('Model accuracy on MNIST: {:.4f}'.format(accuracy))

    matrix_incorrect = torch.zeros((10, 10)).type(torch.int64)
    for index in incorrect:
        matrix_incorrect[pred_y[index], TEST_Y[index]] += 1
        print('MNIST Index: {:4d}'.format(index) +
              ' | Predict: {:1d}'.format(pred_y[index]) +
              ' | Ground truth: {:1d}'.format(TEST_Y[index]))
    print('\nIncorrect index matrix: (PRED_IND, TRUE_IND)')
    print(matrix_incorrect)
    vec_err_pred = torch.sum(matrix_incorrect, dim=1)
    img = Image.new(
        TEST_DATA[0][0].mode, (36 * (torch.max(vec_err_pred).item() + 1), 360))
    draw = ImageDraw.Draw(img)
    for index in range(10):
        draw.text((12, 36 * index + 12),
                  '{:1d}'.format(index), fill=255)

    for index in incorrect:
        img.paste(TEST_DATA[index][0], (36 *
                                        vec_err_pred[pred_y[index]] + 4, 36 * pred_y[index] + 4))
        draw.text((36 * vec_err_pred[pred_y[index]], 36 * pred_y[index]),
                  '{:1d}'.format(TEST_Y[index]), fill=255)
        vec_err_pred[pred_y[index]] -= 1
    img.save('./Practice4_CNN/incorrect.png')
    img.show()


if __name__ == "__main__":
    if TRAIN:
        train(use_model=True)
    else:
        CLF = model.ConvClassifier().cuda()
        CLF.load_state_dict(torch.load('./Practice4_CNN/Model/params.pkl'))
        CLF.eval()
        find_incorrect(CLF)
