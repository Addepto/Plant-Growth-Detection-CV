import logging
import os

import copy
import cv2

# it seems that there is a bug in pytorch/sklearn, and sklearn modules needs to be imported before torch
from net import dataset
import torch
from torchvision import models
from torch import nn, optim
from torch.utils.data.dataloader import DataLoader

import cli
from kws_logging import config_logger

_logger = logging.getLogger('kws')


class Model:
    def __init__(self, use_growth, device):
        self.model = models.resnet18(pretrained=True)
        for param in self.model.parameters():
            param.requires_grad = False
        features = self.model.fc.in_features
        self.output_classes = 9 if use_growth else 2
        self.model.fc = nn.Linear(features, self.output_classes)
        self.model.to(device)

    def get_model(self):
        return self.model


def train(network, train_data_loader, test_data_loader, train_size, test_size, device):
    _logger.info('Starting training')

    dataloaders = {
        'train': train_data_loader,
        'validation': test_data_loader
    }

    dataset_sizes = {
        'train': train_size,
        'validation': test_size
    }

    criterion = nn.CrossEntropyLoss()
    # for 2 classes lr = 0.05
    optimizer = optim.Adam(network.parameters(), lr=0.005, amsgrad=True)
    # optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
    exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    best_model = copy.deepcopy(network.state_dict())
    best_accuracy = 0.0

    for epoch in range(40):
        for phase in ['train', 'validation']:
            if phase == 'train':
                network.train()
            else:
                network.eval()

            running_loss = 0.0
            runnning_corrects = 0

            for i, data in enumerate(dataloaders[phase], 0):
                image, label = data[0].float().to(device), data[1].to(device)
                image = image.permute(0, 3, 1, 2)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = network(image)
                    _, predicted = torch.max(outputs, 1)
                    loss = criterion(outputs, label)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * data[0].size(0)
                runnning_corrects += (predicted == label).sum().item()
                if i % 10 == 9:
                    _logger.info('[%d, %5d] loss: %.7f' % (epoch + 1, i + 1, running_loss / 10))
            if phase == 'train':
                exp_lr_scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_accuracy = float(runnning_corrects) / dataset_sizes[phase]
            _logger.info(f'Phase: {phase} Epoch loss: {epoch_loss}, Epoch accuracy: {epoch_accuracy}')

            if phase == 'validation' and epoch_accuracy > best_accuracy:
                best_accuracy = epoch_accuracy
                best_model = copy.deepcopy(network.state_dict())
    _logger.info('Finished training')
    _logger.info(f'Best accuracy: {best_accuracy}')

    network.load_state_dict(best_model)


def test(network, test_data_loader, device):
    total = 0
    correct = 0
    counter = 0
    with torch.no_grad():
        for data in test_data_loader:
            image, label, label_names = data[0].float().to(device), data[1].to(device), data[2]
            image = image.permute(0, 3, 1, 2)

            outputs = network(image)
            _, predicted = torch.max(outputs.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()

            image = image.permute(0, 2, 3, 1).to('cpu')
            label = label.to('cpu')
            for i in range(image.shape[0]):
                img = image[i, :, :, :].numpy()
                lab = label_names[i]
                pred = label_names[predicted[i]]
                cv2.putText(img, f'Predicted: {pred}. Actual: {lab}', (0, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, 255)
                cv2.imwrite(r'D:\Praca\Addepto\kws\results\res\{name}.png'.format(name=str(counter)), img)
                counter += 1

    accuracy = correct / total * 100.0
    _logger.info(f'Accuracy on {total} images is : {accuracy}')


def run():
    parser = cli.create_parser()
    args = parser.parse_args()
    config_logger(args.output_dir, 'kws')

    use_growth = True

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    output_file_name = 'net.pth'

    _logger.info('Creating net')
    network = Model(use_growth, device)

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
        network.get_model().load_state_dict(torch.load(path))
    else:
        train(network.get_model(), train_loader, test_loader, train_size=len(train_plant_dataset),
              test_size=len(test_plant_dataset), device=device)
        _logger.info(f'Saving model to: {path}')
        torch.save(network.get_model().state_dict(), path)

    test(network.get_model(), test_loader, device)


if __name__ == '__main__':
    run()
