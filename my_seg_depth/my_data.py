import numpy as np
import torch
import os
import os.path
import random
from PIL import Image
import torchvision.transforms as transforms
import torch.utils.data as data
import torchvision.transforms.functional as F
from options.train_options import TrainOptions

ignore_label = 255
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()

def make_dataset(path_files):
    image_paths = []
    assert os.path.isdir(path_files), '%s is not a valid directory' %path_files
    #print(path_files)


    for root, _, fnames in os.walk(path_files):
        for fname in sorted(fnames):
            #print(fname)
            if is_image(fname):
               # print('----------------')
                path = os.path.join(root,fname)
                image_paths.append(path)
    #print(len(image_paths))
    return image_paths, len(image_paths)


class CreateDataset(data.Dataset):
    def initialize(self,opt,train_or_test):
        self.opt = opt
        self.train_or_test = train_or_test
        if(train_or_test=='train'):
            img_syn_path = opt.img_source_file_train
            img_real_path = opt.img_target_file_train
            lab_syn_path = opt.lab_source_file_train
            lab_real_path = opt.lab_target_file_train
            depth_source_path = opt.depth_source_file_train
        else:
            img_syn_path = opt.img_source_file_test
            img_real_path = opt.img_target_file_test
            lab_syn_path = opt.lab_source_file_test
            lab_real_path = opt.lab_target_file_test
            depth_source_path = opt.depth_source_file_test

        self.img_syn_paths,self.img_syn_size = make_dataset(img_syn_path)
        self.img_real_paths, self.img_real_size = make_dataset(img_real_path)
        self.lab_syn_paths,self.lab_syn_size = make_dataset(lab_syn_path)
        self.lab_real_paths,self.lab_real_suze = make_dataset(lab_real_path)
        self.depth_source_paths, self.depth_source_size = make_dataset(depth_source_path)

        self.transform_augment_normalize = get_transform(opt, True, normalize=True)
        self.transform_no_augment_normalize = get_transform(opt, False, normalize=True)

        self.mask2tensor = MaskToTensor()
        self.invalid_synthia = [0, 1, 2, 3, 4, 5]
        self.invalid_cityscape = [0, 1, 2, 3, 4, 5]
        self.syn_id_to_realid = {0: 0,
                                  1: 7,
                                  2: 8,
                                  3: 11,
                                  4: 12,
                                  5: 13,
                                  6: 17,
                                  7: 19,
                                  8: 20,
                                  9: 21,
                                  10: 22,
                                  11: 23,
                                  12: 24,
                                  13: 25,
                                  14: 26,
                                  15: 27,
                                  16: 28,
                                  17: 31,
                                  18: 32,
                                  19: 33,
                                  20: 7,
                                  21: 0,
                                  22: 0,
                                  }
        self.real_id_to_trainid = {-1: ignore_label, 0: ignore_label, 1: ignore_label, 2: ignore_label,
                                   3: ignore_label, 4: ignore_label, 5: ignore_label,
                                   6: 0,
                                   7: 1,
                                   8: 2,
                                   9: 3,
                                   10: 4,
                                   11: 5,
                                   12: 6,
                                   13: 7,
                                   14: 8,
                                   15: 9,
                                   16: 10,
                                   17: 11,
                                   18: 12,
                                   19: 13,
                                   20: 14,
                                   21: 15,
                                   22: 16,
                                   23: 17,
                                   24: 18,
                                   25: 19,
                                   26: 20,
                                   27: 21,
                                   28: 22,
                                   29: 23,
                                   30: 24,
                                   31: 25,
                                   32: 26,
                                   33: 27
                                   }

    def __getitem__(self, item):
        index = random.randint(0, self.img_real_size - 1)
        img_source_path = self.img_syn_paths[item % self.img_syn_size]
        img_target_path = self.img_real_paths[index]

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')
        img_source = img_source.resize([640, 192], Image.BILINEAR)
        img_target = img_target.resize([640, 192], Image.BILINEAR)

        depth_source_path = self.depth_source_paths[item % self.depth_source_size]
        depth_source = Image.open(depth_source_path)  # .convert('RGB')
        depth_source = depth_source.resize([640, 192], Image.BILINEAR)

        lab_source_path = self.lab_syn_paths[item % self.lab_syn_size]
        lab_target_path = self.lab_real_paths[index]

        lab_source = Image.open(lab_source_path)  # .convert('RGB')
        lab_target = Image.open(lab_target_path)  # .convert('RGB')
        lab_source = lab_source.resize([640, 192], Image.NEAREST)
        lab_target = lab_target.resize([640, 192], Image.NEAREST)

        lab_source = np.array(lab_source)
        lab_source_copy = lab_source.copy()
        for k, v in self.syn_id_to_realid.items():
            lab_source_copy[lab_source == k] = v


        lab_target = np.array(lab_target)
        lab_target_copy = lab_target.copy()
        for k, v in self.real_id_to_trainid.items():
            lab_target_copy[lab_target == k] = v
            lab_source_copy[lab_source_copy == k] = v

        lab_target = Image.fromarray(lab_target_copy.astype(np.uint8))
        lab_source = Image.fromarray(lab_source_copy.astype(np.uint8))

        if (self.train_or_test == 'train'):
            img_source, lab_source,depth_source, scale = paired_transform_(self.opt, img_source, lab_source,depth_source)

        img_source = self.transform_augment_normalize(img_source)
        depth_source = self.transform_augment_normalize(depth_source)
        lab_source = self.mask2tensor(np.array(lab_source))

        target_dummy = lab_target
        if (self.train_or_test == 'train'):
            img_target, lab_target, scale = paired_transform(self.opt, img_target, lab_target)
        img_target = self.transform_augment_normalize(img_target)
        lab_target = self.mask2tensor(np.array(lab_target))

        lab_source = lab_source.unsqueeze(0)
        lab_target = lab_target.unsqueeze(0)
        depth_source = depth_source.unsqueeze(0)
        del target_dummy
        return {'img_syn': img_source, 'img_real': img_target,
                'seg_l_syn': lab_source, 'seg_l_real': lab_target,
                'dep_l_syn': depth_source,
                'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path,
                'depth_source_path': depth_source_path
                }

    def __len__(self):
        return max(self.img_syn_size, self.img_real_size)

    def name(self):
        return 'T^2Dataset'
def paired_transform(opt, image, seg):
    scale_rate = 1.0
    opt.flip=True
    opt.rotation=True
    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            seg= F.hflip(seg)

    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BILINEAR)
            seg=F.rotate(seg, degree, Image.NEAREST)

    return image, seg, scale_rate
def paired_transform_(opt, image, seg,dep):
    scale_rate = 1.0
    opt.flip=True
    opt.rotation=True
    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            seg= F.hflip(seg)
            dep = F.hflip(dep)

    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BILINEAR)
            seg=F.rotate(seg, degree, Image.NEAREST)
            dep = F.rotate(dep,degree,Image.NEAREST)
        return image, seg,dep, scale_rate

def get_transform(opt, augment,normalize):
    transforms_list = []

    # if augment:
    #     if opt.isTrain:
    #         transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    if normalize:
        transforms_list += [
            transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    else:
        transforms_list += [
            transforms.ToTensor()
        ]
    return transforms.Compose(transforms_list)

def dataloader(opt,train_or_test):
    datasets = CreateDataset()
    datasets.initialize(opt,train_or_test)
    dataset = data.DataLoader(datasets, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    return dataset

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    opt = TrainOptions().parse()
    dataset=dataloader(opt,train_or_test='train')

    for i, data in enumerate(dataset):

        img=data['lab_target'].data.numpy()
        print(img.shape)
        print(img.max())

        plt.imshow(np.squeeze(img[0,:,:,:]))
        plt.show()