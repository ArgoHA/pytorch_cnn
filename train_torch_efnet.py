# This model was trained in colab with accuracy on test dataset 98%
import torch
from torch import nn
import os
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score
from pathlib import Path
import torchvision.models as models
import time

from utils import get_splited_data, get_preds


def build_model(num_classes, device):
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT,
        )
    model.classifier[1] = nn.Linear(in_features=1280, out_features=num_classes)
    return model.to(device)


def train(train_data, device, optimizer, model, loss_func, valid_data, epochs, path_to_save):
    best_metric = 0

    for epoch in range(1, epochs + 1):
        model.train() # set mode

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


def infer(model_path, valid_data, test_data, device, classes):
    model = build_model(len(classes), device)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint)

    time_start = time.perf_counter()

    valid_accuracy = round(accuracy_score(*get_preds(model, valid_data, device)), 2)
    test_accuracy = round(accuracy_score(*get_preds(model, test_data, device)), 2)

    print('time:', round(time.perf_counter() - time_start, 3))

    print(f'valid_accuracy = {valid_accuracy}, test_accuracy = {test_accuracy}')


def main():
    path_to_folder = Path('')
    path_to_save = Path('')
    device = torch.device('mps') # cuda if you use nvidia gpu

    classes = ['small_gun', 'big_gun', 'phone', 'umbrella', 'empty']
    im_size = 256, 256
    valid_part = 0.15
    test_part = 0.05
    batch_size = 30
    epochs = 15

    torch.manual_seed(42)

    train_data, valid_data, test_data = get_splited_data(path_to_folder, valid_part, test_part,
                                                         classes, im_size, batch_size)

    model = build_model(len(classes), device) # build the model
    loss_func = nn.CrossEntropyLoss() # init loss function (combined with final activation)
    optimizer = torch.optim.Adam(model.parameters())

    train(train_data, device, optimizer, model, loss_func, valid_data, epochs, path_to_save)
    infer(path_to_save / 'model.pt', valid_data, test_data, device, classes)


if __name__ == '__main__':
    main()
