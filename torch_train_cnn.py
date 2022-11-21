import torch
from torch import nn
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import time

from utils import get_splited_data, get_preds


class Gun_classifier(nn.Module):
    def __init__(self):
        super(Gun_classifier, self).__init__()

        self.features = nn.Sequential(
            # input channels, output channels, kernel size, padding
            nn.Conv2d(3, 64, (3, 3), padding='same'),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, (3, 3)),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 128, (3, 3)),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, (3, 3)),
            nn.MaxPool2d((3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 256, (3, 3)),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.glob_pool = nn.AdaptiveAvgPool2d((1, 1)) # Global average pooling

        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 5), # n_classes
        )


    def forward(self, x):
      x = self.features(x)
      x = self.glob_pool(x).reshape(-1, 256)
      x = self.classifier(x)
      return x


def train(train_data, device, optimizer, model, loss_func, valid_data, epochs, path_to_save, scheduler):
    best_metric = 0

    for epoch in range(1, epochs + 1):
        model.train() # set mode

        lrs=[]

        lr_update = epoch % 2 == 0
        with tqdm(train_data, unit="batch") as tepoch:
            for data, target in tepoch:
                tepoch.set_description(f"Epoch {epoch}/{epochs}")

                images, labels = data.to(device), target.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + loss + backward + optimize
                outputs = model.forward(images)
                loss = loss_func(outputs, labels)
                loss.backward()
                optimizer.step()

                lr=optimizer.param_groups[0]["lr"]

        # Get metrics after an epoch
        preds, valid_labels = get_preds(model, valid_data, device)
        f1 = f1_score(preds, valid_labels, average='weighted')
        accuracy = accuracy_score(preds, valid_labels)

        print(f'Valid accuracy: {round(accuracy, 2)}, valid f1: {round(f1, 2)}')

        # Save best model
        if f1 > best_metric:
            best_metric = f1
            torch.save(model.state_dict(), os.path.join(path_to_save, 'model.pt'))
            # torch.save(model, os.path.join(path_to_save, 'model.pt'))

        if lr_update:
            lrs.append(lr)
            scheduler.step()


def infer(model_path, valid_data, test_data, device):
    model = Gun_classifier()
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint)
    model.to(device)

    start_time = time.perf_counter()
    valid_accuracy = round(accuracy_score(*get_preds(model, valid_data, device)), 2)
    test_accuracy = round(accuracy_score(*get_preds(model, test_data, device)), 2)
    work_time = time.perf_counter() - start_time

    print(f'Took {work_time} seconds to proced {len(valid_data) + len(test_data)} images')
    print(f'{work_time / (len(valid_data) + len(test_data))} seconds per image')

    print(f'{valid_accuracy = }, {test_accuracy = }')


def main():
    path_to_folder = Path('/Users/argosaakyan/Data/dis_arm/classification/crops')
    path_to_save = Path('/Users/argosaakyan/Data/dis_arm/classification/torch_model')
    device = torch.device('mps')

    classes = ['big_gun', 'empty', 'phone', 'small_gun', 'umbrella']
    im_size = 178, 178
    valid_part = 0.15
    test_part = 0.05
    batch_size = 30
    epochs = 15

    torch.manual_seed(42)

    train_data, valid_data, test_data = get_splited_data(path_to_folder, valid_part, test_part,
                                                         classes, im_size, batch_size)

    model = Gun_classifier().to(device) # build the model
    loss_func = nn.CrossEntropyLoss() # init loss function (combined with final activation)

    lmbda = lambda epoch: 0.75
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lmbda)

    train(train_data, device, optimizer, model, loss_func, valid_data, epochs, path_to_save, scheduler)

    infer(path_to_save / 'model_prod.pt', valid_data, test_data, device)


if __name__ == '__main__':
    main()
