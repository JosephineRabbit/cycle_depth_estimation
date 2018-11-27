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

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(path_files):
    if path_files.find('.txt') != -1:
        paths, size = make_dataset_txt(path_files)
    else:
        paths, size = make_dataset_dir(path_files)

    return paths, size

def make_dataset_txt(path_files):
    # reading txt file
    image_paths = []
    returnlist=[]
    root='/home/gwl/datasets/KITTI/others/Residual/'
    with open(path_files) as f:
        paths = f.readlines()

    for path in paths:
        path = path.strip().split()
        image_paths.extend(path)
    for path in image_paths:
        if(os.path.exists(root+path[:-4]+'.png')):
            returnlist.append(root+path[:-4]+'.png')
    return returnlist, len(returnlist)


def make_dataset_dir(dir):
    image_paths = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in os.walk(dir):
        for fname in sorted(fnames):
            if is_image_file(fname):
                path = os.path.join(root, fname)
                image_paths.append(path)

    return image_paths, len(image_paths)

class CreateDataset(data.Dataset):
    def initialize(self, opt):
        self.opt = opt
        self.opt.dataset_mode='unpaired'
        self.img_source_paths, self.img_source_size = make_dataset(opt.img_source_file)
        self.img_target_paths, self.img_target_size = make_dataset(opt.img_target_file)
        self.lab_source_paths, self.lab_source_size = make_dataset(opt.lab_source_file)


        print(len(self.img_target_paths))
        self.transform_augment = get_transform(opt, True)
        self.transform_no_augment = get_transform(opt, False)
    def __getitem__(self, item):
        index = random.randint(0, self.img_target_size - 1)
        img_source_path = self.img_source_paths[item % self.img_source_size]
        if self.opt.dataset_mode == 'paired':
            img_target_path = self.img_target_paths[item % self.img_target_size]
        elif self.opt.dataset_mode == 'unpaired':
            img_target_path = self.img_target_paths[index]
        else:
            raise ValueError('Data mode [%s] is not recognized' % self.opt.dataset_mode)

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')
        img_source = img_source.resize([640,192], Image.BICUBIC)
        img_target = img_target.resize([640,192], Image.BICUBIC)
        if self.opt.isTrain:
            lab_source_path = self.lab_source_paths[item % self.lab_source_size]
            lab_source = Image.open(lab_source_path)#.convert('RGB')
            lab_source = lab_source.resize([640,192], Image.BILINEAR)
            lab_source = np.array(lab_source)
            lab_source.clip(min=0,max=8000,out=lab_source)
            lab_source=lab_source/8000.0

            # lab_source=transforms.ToTensor()(lab_source)
            # lab_source=transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(lab_source)


            img_source, lab_source, scale = paired_transform(self.opt, img_source, lab_source)
            img_source = self.transform_augment(img_source)
            lab_source = torch.from_numpy(lab_source).unsqueeze(0).float()

            # img_target, lab_target, scale = paired_transform(self.opt, img_target, lab_target)
            img_target = self.transform_no_augment(img_target)
            # lab_target = self.transform_no_augment(lab_target)

            return {'img_source': img_source, 'img_target': img_target,
                    'lab_source': lab_source,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path
                    }

        else:
            img_source = self.transform_augment(img_source)
            img_target = self.transform_no_augment(img_target)
            return {'img_source': img_source, 'img_target': img_target,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    }

    def __len__(self):
        return max(self.img_source_size, self.img_target_size)

    def name(self):
        return 'T^2Dataset'


def dataloader(opt):
    datasets = CreateDataset()
    datasets.initialize(opt)
    dataset = data.DataLoader(datasets, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    return dataset

def paired_transform(opt, image, depth):
    scale_rate = 1.0
    opt.flip=False
    opt.rotation=False
    if opt.flip:
        n_flip = random.random()
        if n_flip > 0.5:
            image = F.hflip(image)
            depth = F.hflip(depth)

    if opt.rotation:
        n_rotation = random.random()
        if n_rotation > 0.5:
            degree = random.randrange(-500, 500)/100
            image = F.rotate(image, degree, Image.BICUBIC)
            depth = F.rotate(depth, degree, Image.BILINEAR)

    return image, depth, scale_rate


def get_transform(opt, augment):
    transforms_list = []

    if augment:
        if opt.isTrain:
            transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
    transforms_list += [
        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ]

    return transforms.Compose(transforms_list)

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    opt = TrainOptions().parse()
    dataset=dataloader(opt)
    for i, data in enumerate(dataset):
        img=data['lab_source'].data.numpy()
        print(img.shape)

        img=img.squeeze()
        plt.imshow(img)
        plt.show()
    # import shutil
    # path='/home/gwl/datasets/vKITTI/vkitti_1.3.1_rgb/'
    # savepath='/home/gwl/datasets/vKITTI/rgb_all/'
    # table1={'0001':'1','0002':'2','0006':'6','0018':'18','0020':'20','15-deg-left':'15left','15-deg-right':'15right','30-deg-left':'30left','30-deg-right':'30right'}
    # arch1=['0001','0002','0006','0018','0020']
    # arch2=['15-deg-right','15-deg-left','30-deg-right','30-deg-left']
    # for ar1 in arch1:
    #     for ar2 in arch2:
    #         filelist=os.listdir(os.path.join(path,ar1,ar2))
    #         prefix=table1[ar1]+'_'+table1[ar2]+'_'
    #         for file in filelist:
    #             savename=prefix+file
    #             readname=os.path.join(path,ar1,ar2,file)
    #             shutil.copyfile(readname,savepath+savename)