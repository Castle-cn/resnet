import os
from utils.args import get_args
from utils.setup import setup
from utils.utils import get_loader
from torch import nn
import torch
from tqdm import tqdm


class Model():
    def __init__(self, args):

        if torch.cuda.is_available():
            self.device = 'cuda'
        elif torch.backends.mps.is_available():
            self.device = 'mps'  # 苹果电脑加速
        else:
            self.device = 'cpu'

        self.train_loader, self.test_loader = get_loader(args)
        print('data has been loaded over!')

        self.model = setup(args).to(self.device)
        # self.lr = args.lr
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        self.epochs = args.epoch
        self.loss_fn = nn.CrossEntropyLoss()

        # 模型保存路径
        self.save_model_path = args.save_model_path

    def train(self):
        num_batches = len(self.train_loader)
        self.model.train()
        with tqdm(total=num_batches) as pbar:
            for batch, (X, y) in enumerate(self.train_loader):
                X, y = X.to(self.device), y.to(self.device)
                # Compute prediction error
                pred = self.model(X)
                loss = self.loss_fn(pred, y)
                # Backpropagation
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                pbar.update(1)
            loss, current = loss.item(), batch * len(X)
            tqdm.write(f"train loss: {loss:>7f}")

    def test(self):
        size = len(self.test_loader.dataset)
        num_batches = len(self.test_loader)
        self.model.eval()
        test_loss, acc = 0, 0
        with torch.no_grad():
            with tqdm(total=num_batches) as pbar:
                for X, y in self.test_loader:
                    X, y = X.to(self.device), y.to(self.device)
                    pred = self.model(X)
                    test_loss += self.loss_fn(pred, y).item()
                    acc += (pred.argmax(1) == y).type(torch.float).sum().item()
                    pbar.update(1)
        test_loss /= num_batches
        acc /= size
        tqdm.write(f"Test Accuracy: {(100 * acc):>0.1f}%, Avg loss: {test_loss:>8f} \n")
        return acc

    def run(self):
        if not os.path.exists(self.save_model_path):
            os.mkdir(self.save_model_path)

        best_acc = 0
        for t in range(self.epochs):
            print(f"----------Epoch {t + 1} ---------")
            self.train()
            acc = self.test()
            if acc > best_acc:
                best_acc = acc
                torch.save(self.model.state_dict(), os.path.join(self.save_model_path, 'model_weights.pth'))
        print("Done!")


def main():
    # 获取命令行参数
    args = get_args()
    model = Model(args)
    model.run()


if __name__ == '__main__':
    main()
