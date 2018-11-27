import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from datasets.dataset_kitti import dataloader
from models import create_model
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch
if __name__ == '__main__':
    test_epoch='latest'
    writer = SummaryWriter(log_dir='./summary/cyclegan_test/test_%s'%test_epoch)
    opt = TrainOptions().parse()
    data_loader = CreateDataLoader(opt)
    dataset = dataloader(opt)
    dataset_size = len(dataset)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    model.setup(opt)
    model.load_networks(test_epoch)

    visualizer = Visualizer(opt)

    for i, data in enumerate(dataset):
        print(i)
        if(i==100):
            break
        with torch.no_grad():
            model.set_input(data)
            model.optimize_parameters(train_or_test='test')
            errors = model.get_current_losses()
        for name, error in errors.items():
            writer.add_scalar("{}train/{}".format(opt.name, name), error, i)
        images = model.get_current_visuals()
        for name, img in images.items():
            img = torch.from_numpy(img.transpose([2, 0, 1]))
            writer.add_image("{}train/{}".format(opt.name, name), img, i)


