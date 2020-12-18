from PIL import Image
import os
import os.path
import numpy as np
import pickle
import torch
from torchvision import datasets, transforms
import torch.nn.functional as F
import pandas as pd

# from .vision import VisionDataset
from torch.utils.data import Dataset
# from .utils import check_integrity, download_and_extract_archive


class Cifar10Dataset(Dataset):
    """`CIFAR10 <https://www.cs.toronto.edu/~kriz/cifar.html>`_ Dataset.
    Args:
        root (string): Root directory of dataset where directory
            ``cifar-10-batches-py`` exists or will be saved to if download is set to True.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
    """
    # base_folder = 'cifar-10-batches-py'
    # base_folder = 'asymmetric'+str(0.1)
    url = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
    filename = "cifar-10-python.tar.gz"
    tgz_md5 = 'c58f30108f718f92721af3b95e74349a'
    train_list = [
        ['data_batch_1', 'c99cafc152244af753f735de768cd75f'],
        ['data_batch_2', 'd4bba439e000b95fd0a9bffe97cbabec'],
        ['data_batch_3', '54ebc095f3ab1f0389bbae665268c751'],
        ['data_batch_4', '634d18415352ddfa80567beed471001a'],
        ['data_batch_5', '482c414d41f54cd18b22e5b47cb7c3cb'],
    ]

    test_list = [
        ['test_batch', '40351d587109b95175f43aff81a1287e'],
    ]
    meta = {
        'filename': 'batches.meta',
        'key': 'label_names',
        'md5': '5ff9c542aee3614f3951f8cda6e48888',
    }

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False,noise = None, rate = 0.0):

        # super(Cifar10Dataset, self).__init__(root, transform=transform,
        #                               target_transform=target_transform)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # training set or test set
        self.root = root
        self.base_folder = 'clean0.0'#noise+str(rate)##########################################################

        # if download:
        #     self.download()

        # if not self._check_integrity():
        #     raise RuntimeError('Dataset not found or corrupted.' +
        #                        ' You can use download=True to download it')

        if self.train:
            downloaded_list = self.train_list
        else:
            downloaded_list = self.test_list

        self.data = []
        self.targets = []

        # now load the picked numpy arrays
        for file_name, checksum in downloaded_list:
            file_path = os.path.join(self.root, self.base_folder, file_name)
            with open(file_path, 'rb') as f:
                entry = pickle.load(f, encoding='latin1')
                self.data.append(entry['data'])
                if 'labels' in entry:
                    self.targets.extend(entry['labels'])
                else:
                    self.targets.extend(entry['fine_labels'])
        
        if(self.train):
            
            self.targets = self.targets[0:45000]

            #########################################################################################################
            # # kratao N deigmata
            # # auto an thelo na ta dimiourgiso, ego ta exo idi kanei kai apla ta kano load
            # #df = pd.DataFrame(list(zip(idx,self.train_labels)), columns =['idx','target']) 
            # #n  = np.array(df)
            # idx = list(range(45000))

            # arr = np.column_stack((idx , self.targets))
            # np.random.shuffle(arr)#ginete in place

            # #dimourgo dataframe
            # df = pd.DataFrame({'idx': arr[:, 0], 'target': arr[:, 1]})
            # df.sort_values(by=['target'],inplace=True)

            # # kratao ta N deigmata
            # N = 40000
            # keeped_indexes = []
            # for k in range(10):
            #     keeped_indexes.extend(df['idx'].loc[df['target'] == k].head(np.ceil(N/10).astype(int)).tolist())

            # npArray = np.array(keeped_indexes)
            # np.random.shuffle(npArray)#ginetai in place

            # # to ksanakano lista
            # self.keeped_indexes = npArray.tolist()

            npArray = np.load("5000.npy")
            self.keeped_indexes = npArray.tolist()
            #########################################################################################################

            #gia na diavazei mono ta 45000 san train set
            #self.targets = self.targets[0:45000]
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data[0:45000]# an mpei pio prin den douleuei giati to self.data exei 5 arrays ton 10000 stoixeion

            # kratao mono ta indexes pou dialeksa pio prin
            # temp_data =  [self.data[x] for x in self.keeped_indexes]
            self.data = self.data[self.keeped_indexes]
            # self.data = temp_data
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC

            temp_targets = [self.targets[x] for x in self.keeped_indexes]
            self.targets = temp_targets


            # Adding noise
            #add_noise
            if(noise == 'symmetric'):
                for label in range(len(self.data)):
                    if np.random.random()< rate:
                        self.targets[label] = np.random.randint(0,10)
                        #print(label)
                        # self.count += 1
            elif(noise == 'asymmetric'):
                for label in range(len(self.data)):
                    if np.random.random() < rate:
                        if self.targets[label] == 9:
                            self.targets[label] = 1
                        elif self.targets[label] == 2:
                            self.targets[label] = 0
                        elif self.targets[label] == 4:
                            self.targets[label] = 7
                        elif self.targets[label] == 3:
                            self.targets[label] = 5
                        elif self.targets[label] == 5:
                            self.targets[label] = 3
            
        else:
            self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
            self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC


        self._load_meta()

    def _load_meta(self):
        path = os.path.join(self.root, self.base_folder, self.meta['filename'])
        # if not check_integrity(path, self.meta['md5']):
        #     raise RuntimeError('Dataset metadata file not found or corrupted.' +
        #                        ' You can use download=True to download it')
        with open(path, 'rb') as infile:
            data = pickle.load(infile, encoding='latin1')
            self.classes = data[self.meta['key']]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        #prosoxi to transform sto training set epistrefei tin idia eikona 2 fores simfona me to paper
        return img, target, index

    def __len__(self):
        return len(self.data)

    # def _check_integrity(self):
    #     root = self.root
    #     for fentry in (self.train_list + self.test_list):
    #         filename, md5 = fentry[0], fentry[1]
    #         fpath = os.path.join(root, self.base_folder, filename)
    #         if not check_integrity(fpath, md5):
    #             return False
    #     return True

    # def download(self):
    #     if self._check_integrity():
    #         print('Files already downloaded and verified')
    #         return
    #     download_and_extract_archive(self.url, self.root, filename=self.filename, md5=self.tgz_md5)

    # def extra_repr(self):
    #     return "Split: {}".format("Train" if self.train is True else "Test")


class RandomTranslateWithReflect:
    '''
    Translate image randomly

    Translate vertically and horizontally by n pixels where
    n is integer drawn uniformly independently for each axis
    from [-max_translation, max_translation].

    Fill the uncovered blank area with reflect padding.
    '''

    def __init__(self, max_translation):
        self.max_translation = max_translation

    def __call__(self, old_image):
        xtranslation, ytranslation = np.random.randint(-self.max_translation,
                                                       self.max_translation + 1,
                                                       size=2)
        xpad, ypad = abs(xtranslation), abs(ytranslation)
        xsize, ysize = old_image.size

        flipped_lr = old_image.transpose(Image.FLIP_LEFT_RIGHT)
        flipped_tb = old_image.transpose(Image.FLIP_TOP_BOTTOM)
        flipped_both = old_image.transpose(Image.ROTATE_180)

        new_image = Image.new("RGB", (xsize + 2 * xpad, ysize + 2 * ypad))

        new_image.paste(old_image, (xpad, ypad))

        new_image.paste(flipped_lr, (xpad + xsize - 1, ypad))
        new_image.paste(flipped_lr, (xpad - xsize + 1, ypad))

        new_image.paste(flipped_tb, (xpad, ypad + ysize - 1))
        new_image.paste(flipped_tb, (xpad, ypad - ysize + 1))

        new_image.paste(flipped_both, (xpad - xsize + 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad - ysize + 1))
        new_image.paste(flipped_both, (xpad - xsize + 1, ypad + ysize - 1))
        new_image.paste(flipped_both, (xpad + xsize - 1, ypad + ysize - 1))

        new_image = new_image.crop((xpad - xtranslation,
                                    ypad - ytranslation,
                                    xpad + xsize - xtranslation,
                                    ypad + ysize - ytranslation))
        return new_image


class TransformsC10:
    '''
    Apply the same input transform twice, with independent randomness.
    '''

    def __init__(self):
        # flipping image along vertical axis
        self.flip_lr = transforms.RandomHorizontalFlip(p=0.5)
        # image augmentation functions
        normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
        col_jitter = transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8)
        img_jitter = transforms.RandomApply([
            RandomTranslateWithReflect(4)], p=0.8)
        rnd_gray = transforms.RandomGrayscale(p=0.25)
        # main transform for self-supervised training
        self.train_transform = transforms.Compose([
            img_jitter,
            col_jitter,
            rnd_gray,
            transforms.ToTensor(),
            normalize
        ])
        # transform for testing
        self.test_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

    def __call__(self, inp):
        inp = self.flip_lr(inp)
        out1 = self.train_transform(inp)
        out2 = self.train_transform(inp)
        return out1, out2
