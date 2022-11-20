import os
from torchvision import datasets, transforms
import shutil
import torch
import random


# Count images for train valid test split
def get_sets_amount(valid_x, test_x, path_to_folder):
    count_images = 0

    folders = [x for x in os.listdir(path_to_folder) if not x.startswith(".")]
    for folder in folders:
        path = os.path.join(path_to_folder, folder)
        for image in os.listdir(path):
            image_path = os.path.join(path, image)
            if os.path.isfile(image_path) and not image.startswith("."):
                count_images += 1

    valid_amount = int(count_images * valid_x)
    test_amount = int(count_images * test_x)
    train_amount = count_images - valid_amount - test_amount

    return train_amount, valid_amount, test_amount


# Split images by folders
def create_sets_folders(path_to_folder, valid_part, test_part, classes):
    train_amount, valid_amount, test_amount = get_sets_amount(valid_part, test_part, path_to_folder)
    print(f'Train images: {train_amount}\nValid images: {valid_amount}\nTest images: {test_amount}')

    os.chdir(path_to_folder)
    if os.path.isdir('train') is False:

        os.mkdir('valid')
        os.mkdir('test')

        for name in classes:
            shutil.copytree(f'{name}', f'train/{name}')
            os.mkdir(f'valid/{name}')
            os.mkdir(f'test/{name}')

            valid_samples = random.sample(os.listdir(f'train/{name}'), round(valid_amount / len(classes)))
            for j in valid_samples:
                shutil.move(f'train/{name}/{j}', f'valid/{name}')

            test_samples = random.sample(os.listdir(f'train/{name}'), round(test_amount / len(classes)))
            for k in test_samples:
                shutil.move(f'train/{name}/{k}', f'test/{name}')

        print('Created train, valid and test directories')


# Load images to Torch and preprocess them
def load_data(path, im_size, batch_size):
    transform = transforms.Compose([transforms.Resize(im_size),
                                    transforms.ToTensor()])
    dataset = datasets.ImageFolder(path, transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader


def get_splited_data(path_to_folder, valid_part, test_part, classes, im_size, batch_size):
    create_sets_folders(path_to_folder, valid_part, test_part, classes)

    train_data = load_data(os.path.join(path_to_folder, 'train'), im_size, batch_size)
    valid_data = load_data(os.path.join(path_to_folder, 'valid'), im_size, batch_size)
    test_data = load_data(os.path.join(path_to_folder, 'test'), im_size, batch_size)

    return train_data, valid_data, test_data


# get model predictions
def get_preds(model, testing_data, device):
    val_preds = []
    val_labels = []
    model.eval() # set mode

    with torch.no_grad():
        for data, target in testing_data:
            images, labels = data.to(device), target.to(device)
            outputs = model.forward(images)
            val_preds.extend(torch.max(outputs.data, 1).indices.tolist())
            val_labels.extend(labels.tolist())

    return val_preds, val_labels
