import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from datasets.dataset_synthia import dataloader
from models import create_model,create_model_seg,create_model_segCycle
from util.visualizer import Visualizer
from tensorboardX import SummaryWriter
import matplotlib.pyplot as plt
import torch
if __name__ == '__main__':
    writer_train = SummaryWriter(log_dir='./summary/synthia_segCycle')
    writer_test = SummaryWriter(log_dir='./summary/synthia_segCycle')
    opt = TrainOptions().parse()
    opt.name='synthia_segCycle'

    dataset_train = dataloader(opt,train_or_test='train')
    dataset_test = dataloader(opt,train_or_test='test')
    model = create_model_segCycle(opt)
    model.setup(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    global_iter=0
    for epoch in range(1,30):
        epoch_start_time = time.time()
        iter_data_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset_train):
            print(global_iter)
#=-------------------------------------------------
            if (global_iter % 200 == 0 and global_iter > 200):
                print("validation start")
                model.eval()
                for ii, data_test in enumerate(dataset_test):
                    if(ii==50):
                        break
                    with torch.no_grad():
                        model.set_input(data_test)
                        model.optimize_parameters(train_or_test='test')
                    errors = model.get_current_losses()
                    for name, error in errors.items():
                        writer_test.add_scalar("{}test/{}".format(opt.name, name), error, global_iter+ii)
                    images = model.get_current_visuals()
                    for name, img in images.items():
                        im = torch.from_numpy(img.transpose([2, 0, 1])).squeeze(0)
                        writer_test.add_image("{}test/img_{}".format(opt.name, name), im, global_iter+ii)
                print("validation done")
#---------------------------------------------
            global_iter+=1
            iter_start_time = time.time()
            if total_steps % opt.print_freq == 0:
                t_data = iter_start_time - iter_data_time
            visualizer.reset()
            total_steps += opt.batch_size
            epoch_iter += opt.batch_size
            model.set_input(data)
            model.optimize_parameters(train_or_test='train')
            if (global_iter % 50 == 0):
                errors = model.get_current_losses()
                for name, error in errors.items():
                    writer_train.add_scalar("{}train/{}".format(opt.name, name), error, global_iter)
                images = model.get_current_visuals()

                for name, img in images.items():
                    img=img/img.max()
                    img = torch.from_numpy(img.transpose([2, 0, 1]))
                    writer_train.add_image("{}train/img_{}".format(opt.name, name), img, global_iter)
            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save_networks('iter_%d'%total_steps)

            iter_data_time = time.time()
        if epoch % 5 == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save_networks(epoch)

        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()
