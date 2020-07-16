import logging
import os

# it seems that there is a bug in pytorch/sklearn, and sklearn modules needs to be imported before torch
from net import dataset
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import cli
from kws_logging import config_logger

_logger = logging.getLogger('kws')


class Net(nn.Module):
    def __init__(self, growth=False):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 141 * 213, 120)
        self.fc2 = nn.Linear(120, 84)
        self.output_classes = 2 if growth else 9
        self.fc3 = nn.Linear(84, self.output_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.contiguous().view(-1, 16 * 141 * 213)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def train(network, train_data_loader, device):
    _logger.info('Starting training')

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(network.parameters(), lr=0.001)

    for epoch in range(10):
        running_loss = 0.0

        for i, data in enumerate(train_data_loader, 0):
            image, label = data[0].float().to(device), data[1].to(device)
            image = image.permute(0, 3, 1, 2)

            optimizer.zero_grad()

            outputs = network(image)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 5 == 4:  # print every 5 mini-batches
                _logger.info('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / 5))
                running_loss = 0.0
    _logger.info('Finished training')


def test(network, test_data_loader, device):
    total = 0
    correct = 0
    with torch.no_grad():
        for data in test_data_loader:
            image, label = data[0].float().to(device), data[1].to(device)
            image = image.permute(0, 3, 1, 2)

            outputs = network(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = correct / total * 100.0
    _logger.info(f'Accuracy on {total} images is : {accuracy}')


def run():
    parser = cli.create_parser()
    args = parser.parse_args()
    config_logger(args.output_dir, 'kws')

    use_growth = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_file_name = 'net.pth'

    _logger.info('Creating net')
    network = Net(use_growth)
    network.to(device)

    _logger.info('Creating dataset')
    train_plant_dataset = dataset.PlantDataset(args.output_dir, args.plants_names, args.growth_stages,
                                               use_growth=use_growth)
    test_plant_dataset = dataset.PlantDataset(args.output_dir, args.plants_names, args.growth_stages,
                                              train=False, use_growth=use_growth)
    _logger.info('Creating data loader')
    train_loader = DataLoader(train_plant_dataset, batch_size=4, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_plant_dataset, batch_size=4, shuffle=True, num_workers=2)

    path = os.path.join(args.output_dir, output_file_name) if not args.network_model_path else args.network_model_path
    if args.network_model_path:
        _logger.info('Loading network from file {path}'.format(path=args.network_model_path))
        network.load_state_dict(torch.load(path))
    else:
        train(network, train_loader, device)
        _logger.info(f'Saving model to: {path}')
        torch.save(network.state_dict(), path)

    test(network, test_loader, device)


if __name__ == '__main__':
    run()
