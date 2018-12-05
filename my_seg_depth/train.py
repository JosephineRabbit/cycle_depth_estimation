import time
from options.train_options import TrainOptions
from datasets.dataset_synthia import dataloader
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import torch
#from models import
from  .networks import G_1,Discriminator,Feature_net,SEG,DEP

if __name__ == '__main__':
    opt = TrainOptions().parse()
    opt.name = 'synthia_segCycle'

    dataset_train = dataloader(opt, train_or_test='train')
    dataset_test = dataloader(opt, train_or_test='test')
    writer_train = SummaryWriter(log_dir='./summary/my_seg_depth')
    writer_test = SummaryWriter(log_dir='./summary/my_seg_depth')
    opt = TrainOptions().parse()
    opt.name = 'my_seg_depth'

    dataset_train = dataloader(opt, train_or_test='train')
    dataset_test = dataloader(opt, train_or_test='test')
    G_syn = G_1(input_nc=3, out_nc=128).cuda()
    G_real = G_1(input_nc=3, out_nc=128).cuda()
    Dis = Discriminator().cuda()
    F = Feature_net().cuda()
    Seg_net = SEG().cuda()
    Dep_net = DEP().cuda()
    visualizer = Visualizer(opt)
    total_steps = 0
    global_iter=0

    for epoch in range(1,30):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset_train):
            print(global_iter)
            if (global_iter % 200 == 0 and global_iter > 200):
                print("validation start")
                model.eval()
                for ii, data_test in enumerate(dataset_test):
                    if (ii == 50):
                        break
                    with torch.no_grad():
                        model.set_input(data_test)
                        model.optimize_parameters(train_or_test='test')
                    errors = model.get_current_losses()
                    for name, error in errors.items():
                        writer_test.add_scalar("{}test/{}".format(opt.name, name), error, global_iter + ii)
                    images = model.get_current_visuals()
                    for name, img in images.items():
                        im = torch.from_numpy(img.transpose([2, 0, 1])).squeeze(0)
                        writer_test.add_image("{}test/img_{}".format(opt.name, name), im, global_iter + ii)
                print("validation done")






