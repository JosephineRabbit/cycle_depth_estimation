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
import cv2


ignore_label = 0
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
def Canny(img):
    img_ = np.uint8(img)
    img_1 = np.zeros(img_.shape)
    img_2 = np.zeros(img_.shape)
    img_3 = np.zeros(img_.shape)
    img_4 = np.zeros(img_.shape)
    e21 = np.zeros(img_.shape) + 21.
    img_1[:-1, :] = img_[1:, :]
    img_1[img_ == e21] = 21
    img_2[1:, :] = img_[:-1, :]
    img_2[img_ == e21] = 21
    img_3[:, :-1] = img_[:, 1:]
    img_3[img_ == e21] = 21
    img_4[:, 1:] = img_[:, :-1]
    img_4[img_ == e21] = 21
    # edge = cv2.Sobel(img_,cv2.CV_8U,1,0)
    edge = np.zeros(img_.shape)
    edge_ = edge.copy()
    edge[img_ != img_1] = 1
    edge[img_ != img_2] = 1
    edge[img_ != img_4] = 1
    edge[img_ != img_3] = 1



    return edge

class CreateDataset(data.Dataset):
    def initialize(self,opt,train_or_test):
        self.opt = opt
        self.train_or_test = train_or_test
        if(train_or_test=='train'):
            img_syn_path = opt.img_source_file_train
            img_real_path = opt.img_target_file_train
            lab_syn_path = opt.lab_source_file_train
            lab_real_path = opt.lab_target_file_train
            self.lab_real_paths, self.lab_real_suze = make_dataset(lab_real_path)
            depth_source_path = opt.depth_source_file_train
        else:
            img_syn_path = opt.img_source_file_test
            img_real_path = opt.img_target_file_test
            lab_syn_path = opt.lab_source_file_test

            depth_source_path = opt.depth_source_file_test

        self.img_syn_paths,self.img_syn_size = make_dataset(img_syn_path)
        self.img_real_paths, self.img_real_size = make_dataset(img_real_path)
        self.lab_syn_paths,self.lab_syn_size = make_dataset(lab_syn_path)


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
        #print('index',index)
        img_source_path = self.img_syn_paths[item % self.img_syn_size]
        img_target_path = self.img_real_paths[index]

        img_source = Image.open(img_source_path).convert('RGB')
        img_target = Image.open(img_target_path).convert('RGB')
        img_source = img_source.resize([576, 192], Image.BILINEAR)
        img_target = img_target.resize([576, 192], Image.BILINEAR)



        depth_source_path = self.depth_source_paths[item % self.depth_source_size]
        depth_source = Image.open(depth_source_path)  # .convert('RGB')
        depth_source = depth_source.resize([576, 192], Image.BILINEAR)

        lab_source_path = self.lab_syn_paths[item % self.lab_syn_size]

        if self.train_or_test=='train':
            lab_target_path = self.lab_real_paths[index]

            lab_source = Image.open(lab_source_path)  # .convert('RGB')
            lab_target = Image.open(lab_target_path)  # .convert('RGB')



            if (self.train_or_test == 'train'):
                img_source, lab_source, depth_source, scale = paired_transform_(self.opt, img_source, lab_source,
                                                                                depth_source)

            if (self.train_or_test == 'train'):
                img_target, lab_target, scale = paired_transform(self.opt, img_target, lab_target)


            img_target = np.array(img_target)
            img_source = np.array(img_source)
            img_source = self.transform_augment_normalize(img_source)
            img_target = self.transform_augment_normalize(img_target)


            lab_source = np.array(lab_source).astype(np.float32)
            lab_source[lab_source == 0] = 7

            for k, v in self.real_id_to_trainid.items():
              #  lab_target[lab_target == k] = v
                lab_source[lab_source.copy() == k] = v

            lab_source_copy = lab_source.copy()

            lab_target = np.array(lab_target).astype(np.float32)

            lab_target  = lab_target - 6
            lab_target[lab_target<0] =0

            lab_source_edge = Canny(lab_source)

            lab_target_edge = Canny(lab_target)

          #  lab_source = lab_source_copy.resize([576, 192], Image.NEAREST)
           # lab_target = lab_target_copy.resize([576, 192], Image.NEAREST)
            lab_source_edge = cv2.resize(lab_source_edge, (576, 192))
            lab_target_edge = cv2.resize(lab_target_edge, (576, 192))


            #print('+_++++++++++', lab_target.max())
            lab_source_copy = cv2.resize(lab_source_copy, (576, 192), cv2.INTER_NEAREST)
            lab_target_copy = cv2.resize(lab_target, (576, 192), cv2.INTER_NEAREST)
            #print('-------------', lab_source_copy.max())


            lab_source = self.mask2tensor(lab_source_copy.astype(np.uint8))

            lab_target = self.mask2tensor(lab_target_copy.astype(np.uint8))



            depth_source = np.array(depth_source).astype(np.float32)

            lab_source_ = lab_source.unsqueeze(0)
            lab_target_ = lab_target.unsqueeze(0)
            del lab_target_copy, lab_source_copy, lab_target,lab_source
            depth_source_2 = depth_source.copy()
            depth_source_3 = depth_source.copy()
            depth_source_4 = depth_source.copy()
            depth_source_5 = depth_source.copy()
         #   print(depth_source.max(),depth_source.min(),'-------------')

            depth_source[depth_source > 8000] = 8000



            depth_source_2[depth_source_2> 8000] = 8000
            depth_source_2[depth_source < 5000] = 5000
          #  print('-----------2222222222222222222222',depth_source_2.min(),depth_source_2.max())
            depth_source_2= np.expand_dims((2 * (depth_source_2-depth_source_2.min())/ (depth_source_2.max()-depth_source_2.min()) - 1), 0)
            depth_source_3[depth_source > 6000] = 6000
            depth_source_3[depth_source < 3000] = 3000
           # print('-----------3333333333333333333333333333333', depth_source_3.min(), depth_source_3.max())
            depth_source_3 = np.expand_dims(
                (2 * (depth_source_3 - depth_source_3.min()) / (depth_source_3.max() - depth_source_3.min()) - 1), 0)
            depth_source_4[depth_source > 4000] = 4000
            depth_source_4[depth_source < 1000] = 1000
            #print('-----------44444444444444444444', depth_source_4.min(), depth_source_4.max())
            depth_source_4 = np.expand_dims(
                (2 * (depth_source_4 - depth_source_4.min()) / (depth_source_4.max()-depth_source_4.min()) - 1), 0)

            depth_source_5[depth_source > 2000] = 2000
            #print('-----------555555555555555555555', depth_source_5.min(), depth_source_5.max())
            depth_source_5 = np.expand_dims(
                (2 *(depth_source_5-depth_source_4.min()) / (depth_source_5.max()-depth_source_5.min()) - 1), 0)

            depth_source = np.expand_dims((2 * (depth_source-depth_source.min()) / (depth_source.max()-depth_source.min()) - 1), 0)
            #print(depth_source.min(),depth_source.max(),'+_+_=_')
            depth_labels = np.concatenate([depth_source_2,depth_source_3,depth_source_4,depth_source_5],axis=0)

            #print(depth_labels.shape)


            #print(img_target_path[-13:])

            return {'img_syn':np.array(img_source),'img_real':np.array(img_target),
                    'dep_l_syn':depth_source,'seg_l_syn':np.array(lab_source_),'depth_l_s':depth_labels,
                    'seg_l_real':np.array(lab_target_),'seg_e_real':lab_target_edge,'seg_e_syn':lab_source_edge,
                'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path, 'lab_target_paths': lab_target_path,
                    'depth_source_path': depth_source_path,'name':img_target_path[-14:]
                    }
        else:
            # lab_target_path = self.lab_real_paths[index]

            lab_source = Image.open(lab_source_path)  # .convert('RGB')
            # lab_target = Image.open(lab_target_path)  # .convert('RGB')

            lab_source = lab_source.resize([576,192], Image.NEAREST)
            # lab_target = lab_target.resize([256, 256], Image.NEAREST)


            img_target = np.array(img_target)
            img_source = np.array(img_source)
            img_source = self.transform_augment_normalize(img_source)
            img_target = self.transform_augment_normalize(img_target)

            lab_source = np.array(lab_source)
            lab_source_copy = lab_source.copy()
            # lab_target = np.array(lab_target)
            # lab_target_copy = lab_target.copy()

            lab_source_copy[lab_source_copy == 0] = 7

            for k, v in self.real_id_to_trainid.items():
                #    lab_target_copy[lab_target == k] = v
                lab_source_copy[lab_source_copy == k] = v
            lab_source_edge = Canny(lab_source)

            # lab_target_edge = Canny(lab_target)
            lab_source_edge = cv2.resize(lab_source_edge, (576, 192))
            # lab_target_edge = cv2.resize(lab_target_edge, (256, 256))

            lab_source = self.mask2tensor(lab_source_copy.astype(np.uint8))

            # lab_target = self.mask2tensor(lab_target_copy.astype(np.uint8))

            depth_source = np.array(depth_source).astype(np.float32)

            lab_source = lab_source.unsqueeze(0)
            # lab_target = lab_target.unsqueeze(0)
            depth_source_2 = depth_source.copy()
            depth_source_3 = depth_source.copy()
            depth_source_4 = depth_source.copy()
            depth_source_5 = depth_source.copy()
            #   print(depth_source.max(),depth_source.min(),'-------------')

            depth_source[depth_source > 8000] = 8000

            depth_source_2[depth_source_2 > 8000] = 8000
            depth_source_2[depth_source < 5000] = 5000
            #  print('-----------2222222222222222222222',depth_source_2.min(),depth_source_2.max())
            depth_source_2 = np.expand_dims(
                (2 * (depth_source_2 - depth_source_2.min()) / (depth_source_2.max() - depth_source_2.min()) - 1), 0)
            depth_source_3[depth_source > 6000] = 6000
            depth_source_3[depth_source < 3000] = 3000
            # print('-----------3333333333333333333333333333333', depth_source_3.min(), depth_source_3.max())
            depth_source_3 = np.expand_dims(
                (2 * (depth_source_3 - depth_source_3.min()) / (depth_source_3.max() - depth_source_3.min()) - 1), 0)
            depth_source_4[depth_source > 4000] = 4000
            depth_source_4[depth_source < 1000] = 1000
            # print('-----------44444444444444444444', depth_source_4.min(), depth_source_4.max())
            depth_source_4 = np.expand_dims(
                (2 * (depth_source_4 - depth_source_4.min()) / (depth_source_4.max() - depth_source_4.min()) - 1), 0)

            depth_source_5[depth_source > 2000] = 2000
            # print('-----------555555555555555555555', depth_source_5.min(), depth_source_5.max())
            depth_source_5 = np.expand_dims(
                (2 * depth_source_5 / depth_source_5.max() - 1), 0)

            depth_source = np.expand_dims((2 * depth_source / depth_source.max() - 1), 0)
            depth_labels = np.concatenate([depth_source_2, depth_source_3, depth_source_4, depth_source_5], axis=0)
            # print(depth_labels.shape)


            #      print(img_source_path,img_target_path,lab_source_path,lab_target_path,depth_source_path)

            print('++++++++++++++++++++',img_target_path[-14:])
            return {'img_syn': np.array(img_source), 'img_real': np.array(img_target),
                    'dep_l_syn': depth_source, 'seg_l_syn': np.array(lab_source), 'depth_l_s': depth_labels,
                    'seg_e_syn': lab_source_edge,
                    'img_source_paths': img_source_path, 'img_target_paths': img_target_path,
                    'lab_source_paths': lab_source_path,
                    'depth_source_path': depth_source_path,'f_name':img_target_path[-56:-29],'l_name':img_target_path[-24:]
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

    if augment:
         if opt.isTrain:
             transforms_list.append(transforms.ColorJitter(brightness=0.0, contrast=0.0, saturation=0.0, hue=0.0))
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
    dataset = data.DataLoader(datasets, batch_size=1, shuffle=True, num_workers=8)
    return dataset

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    opt = TrainOptions().parse()
    dataset=dataloader(opt,train_or_test='train')

    for i, data in enumerate(dataset):





        img = data['img_syn'].data.numpy()
        print('img',img.shape)
        print(img.max())


        img=data['seg_l_syn'].data.numpy()
        print('lab',img.shape)
        print(img.max())

        img = data['dep_l_syn'].data.numpy()
        print(img.shape)
        print(img.max())



        img = data['seg_e_real'].data.numpy()
        print(img.shape)
        print('seg_e', img.max())

        img = data['seg_l_real'].data.numpy()
        print('lab', img.shape)
        print(img.max())

        plt.imshow(img[0,0,:,:])
        plt.show()