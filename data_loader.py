from torch.utils import data
from torchvision import transforms as T
from PIL import Image
import os
import numpy as np
import pickle
from glob import glob

DEBUG=False

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_h:offset_h + size, offset_w:offset_w + size]


class MedicalData(data.Dataset):

    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.datasetA = []
        self.datasetB = []
        self.preprocess()

        if mode == 'train':
            self.num_images = len(self.datasetA) + len(self.datasetB)
        else:
            self.num_images = max(len(self.datasetA), len(self.datasetB))

    def preprocess(self):
        if self.mode in ['train'] :
            pos = glob(os.path.join(self.image_dir, 'train', 'pos', '*png'))
            neg = glob(os.path.join(self.image_dir, 'train', 'neg', '*png'))
            neg_mixed = glob(os.path.join(self.image_dir, 'train', 'neg_mixed', '*png'))

            self.datasetA = pos + neg_mixed
            self.datasetB = neg
        else:
            self.datasetA = glob(os.path.join(self.image_dir, 'test', 'pos', '*png'))
            self.datasetB = glob(os.path.join(self.image_dir, 'test', 'neg', '*png'))

        print('Finished preprocessing the dataset...')

    def __getitem__(self, index):
        datasetA = self.datasetA
        datasetB = self.datasetB
        
        filenameA = datasetA[index%len(datasetA)]
        filenameB = datasetB[index%len(datasetB)]

        if self.mode in ['train']:
            imageA = Image.open(filenameA).convert("RGB")
            imageB = Image.open(filenameB).convert("RGB")
        else:
            imageA = Image.open(filenameA).convert("RGB")
            imageB = Image.open(filenameB).convert("RGB")

        imageA = np.array(imageA)
        imageB = np.array(imageB)

        if DEBUG: print("Original image size: ", imageA.shape, imageB.shape, "min: ", np.min(imageA), np.min(imageB), "max: ", np.max(imageA), np.max(imageB), "type: ", imageA.dtype, imageB.dtype)

        imageA = np.pad(imageA, [(5, 5), (5, 5), (0, 0)], mode='constant', constant_values=0)
        imageA = central_crop(imageA)

        imageB = np.pad(imageB, [(5, 5), (5, 5), (0, 0)], mode='constant', constant_values=0)
        imageB = central_crop(imageB)

        if DEBUG: print("Processed image size: ", imageA.shape, imageB.shape, "min: ", np.min(imageA), np.min(imageB), "max: ", np.max(imageA), np.max(imageB), "type: ", imageA.dtype, imageB.dtype)

        imageA = Image.fromarray(imageA.astype(np.uint8))
        imageB = Image.fromarray(imageB.astype(np.uint8))

        return self.transform(imageA), self.transform(imageB)

    def __len__(self):
        """Return the number of images."""
        return self.num_images



class TestValidInductive(data.Dataset):
    """Dataset class for the inductive testing."""

    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.datasetA = []
        self.datasetB = []
        self.preprocess()

        if "ano" in self.mode:
            self.num_images = len(self.datasetA)
        elif "hea" in self.mode:
            self.num_images = len(self.datasetB)

    def preprocess(self):
        self.datasetA = glob(os.path.join(self.image_dir, 'test', 'pos', '*png'))
        self.datasetB = glob(os.path.join(self.image_dir, 'test', 'neg', '*png'))

        print(f'Finished preprocessing the dataset for {self.mode} ...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if "ano" in self.mode:
            dataset = self.datasetA
        else:
            dataset = self.datasetB

        filename = dataset[index%len(dataset)]
        image = Image.open(filename).convert("RGB")
        image = np.array(image)

        if DEBUG: print("Original image size: ", image.shape, "min: ", np.min(image), "max: ", np.max(image), "type: ", image.dtype)

        image = np.pad(image, [(5, 5), (5, 5), (0, 0)], mode='constant', constant_values=0)
        image = central_crop(image)

        if DEBUG: print("Processed image size: ", image.shape, "min: ", np.min(image), "max: ", np.max(image), "type: ", image.dtype)

        image = Image.fromarray(image.astype(np.uint8))

        return filename, self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images



class TestValidTransductive(data.Dataset):
    """Dataset class for the transductive testing."""

    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.datasetA = []
        self.datasetB = []
        self.preprocess()

        if "ano" in self.mode:
            self.num_images = len(self.datasetA)
        elif "hea" in self.mode:
            self.num_images = len(self.datasetB)

    def preprocess(self):
        self.datasetA = glob(os.path.join(self.image_dir, 'test', 'pos', '*png'))
        self.datasetB = glob(os.path.join(self.image_dir, 'test', 'neg', '*png'))

        print(f'Finished preprocessing the dataset for {self.mode} ...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if "ano" in self.mode:
            dataset = self.datasetA
        else:
            dataset = self.datasetB

        filename = dataset[index%len(dataset)]
        image = Image.open(filename).convert("RGB")
        image = np.array(image)

        if DEBUG: print("Original image size: ", image.shape, "min: ", np.min(image), "max: ", np.max(image), "type: ", image.dtype)

        image = np.pad(image, [(5, 5), (5, 5), (0, 0)], mode='constant', constant_values=0)
        image = central_crop(image)

        if DEBUG: print("Processed image size: ", image.shape, "min: ", np.min(image), "max: ", np.max(image), "type: ", image.dtype)

        image = Image.fromarray(image.astype(np.uint8))

        return filename, self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images


class testAUCp(data.Dataset):
    """Dataset class for the AUCp calculation."""

    def __init__(self, image_dir, transform, mode):
        self.image_dir = image_dir
        self.transform = transform
        self.mode = mode
        self.datasetA = []
        self.datasetB = []
        self.preprocess()

        if "ano" in self.mode:
            self.num_images = len(self.datasetA)
        elif "hea" in self.mode:
            self.num_images = len(self.datasetB)

    def preprocess(self):
        self.datasetA = glob(os.path.join(self.image_dir, 'test', 'pos', '*png'))
        self.datasetB = glob(os.path.join(self.image_dir, 'test', 'neg', '*png'))

        print(f'Finished preprocessing the dataset for {self.mode} ...')

    def __getitem__(self, index):
        """Return one image and its corresponding attribute label."""
        if "ano" in self.mode:
            dataset = self.datasetA
        else:
            dataset = self.datasetB

        filename = dataset[index%len(dataset)]
        image = Image.open(filename).convert("RGB")
        image = np.array(image)

        if DEBUG: print("Original image size: ", image.shape, "min: ", np.min(image), "max: ", np.max(image), "type: ", image.dtype)

        image = np.pad(image, [(5, 5), (5, 5), (0, 0)], mode='constant', constant_values=0)
        image = central_crop(image)

        if DEBUG: print("Processed image size: ", image.shape, "min: ", np.min(image), "max: ", np.max(image), "type: ", image.dtype)

        image = Image.fromarray(image.astype(np.uint8))

        return filename, self.transform(image)

    def __len__(self):
        """Return the number of images."""
        return self.num_images



def get_loader(image_dir, image_size=192, batch_size=1, dataset='MedicalData', mode='train', num_workers=1):
    """Build and return a data loader."""
    transform = []
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.Resize(image_size))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    if dataset == 'MedicalData':
        dataset = MedicalData(image_dir, transform, mode)
    elif dataset == 'TestValidInductive':
        dataset = TestValidInductive(image_dir, transform, mode)
    elif dataset == 'TestValidTransductive':
        dataset = TestValidTransductive(image_dir, transform, mode)
    elif dataset == 'testAUCp':
        dataset = testAUCp(image_dir, transform, mode)
    else:
        print("Dataset not found!")
        exit()

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  shuffle=(mode=='train'),
                                  num_workers=num_workers)
    return data_loader
