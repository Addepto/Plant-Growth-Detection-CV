import logging
import os

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import cli
from kws_logging import config_logger
from net import dataset

_logger = logging.getLogger('kws')


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 9)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(network, train_data_loader, device):
    _logger.info('Starting training')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    for epoch in range(5):
        running_loss = 0.0

        for i, data in enumerate(train_data_loader, 0):
            image, label = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = network(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 30 == 29:  # print every 2000 mini-batches
                _logger.info('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0
    _logger.info('Finished training')


def run():
    parser = cli.create_parser()
    args = parser.parse_args()
    config_logger(args.output_dir, 'kws')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    _logger.info('Creating net')
    network = Net()
    network.to(device)

    _logger.info('Creating dataset')
    train_dataset = dataset.PlantDataset(args.output_dir, args.plants_names, args.growth_stages)
    _logger.info('Creating data loader')
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=2)

    train(network, train_loader, device)
    path = os.path.join(args.output_dir, 'net.pth')
    _logger.info(f'Saving model to: {path}')


if __name__ == '__main__':
    run()
