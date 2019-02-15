import numpy as np
import argparse
import os
import cv2


def compute_errors(ground_truth, predication):

    # accuracy
    predication = (predication-predication.min())/(predication.max()-predication.min())*49+1
    print(ground_truth.shape)
    threshold = np.maximum((ground_truth / predication),(predication / ground_truth))
    a1 = (threshold < 1.25 ).mean()
    a2 = (threshold < 1.25 ** 2 ).mean()
    a3 = (threshold < 1.25 ** 3 ).mean()

    #MSE
    rmse = (ground_truth - predication) ** 2
    rmse = np.sqrt(rmse.mean())

    #MSE(log)
    rmse_log = (np.log(ground_truth) - np.log(predication)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    # Abs Relative difference
    abs_rel = np.mean(np.abs(ground_truth - predication) / ground_truth)

    # Squared Relative difference
    sq_rel = np.mean(((ground_truth - predication) ** 2) / ground_truth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3



def eval_metric():
    num_samples = 1000
    abs_rel = np.zeros(num_samples, np.float32)
    sq_rel = np.zeros(num_samples, np.float32)
    rmse = np.zeros(num_samples, np.float32)
    rmse_log = np.zeros(num_samples, np.float32)
    a1 = np.zeros(num_samples, np.float32)
    a2 = np.zeros(num_samples, np.float32)
    a3 = np.zeros(num_samples, np.float32)
    gt_p = '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/data/test_gt'
    pre_p = '/home/dut-ai/Documents/temp/code/pytorch-CycleGAN-and-pix2pix/data/test_re'
    files = os.listdir(gt_p)
    files_2 = os.listdir(pre_p)
    i = 0
    for file in files_2:
        print(file)
        if file in files:
            print('1+++++++++++++++++++++++++++++++++++++++++++++++++')
            gt = cv2.imread(gt_p+'/'+file,0)

            pred = cv2.imread(pre_p+'/'+file,0)
            pred = cv2.resize(pred,(gt.shape[1],gt.shape[0]))
            pred = pred/255*80
           # gt = gt/gt.max()*80

            print(gt.max(),gt.min())
            print('pred',pred.max(),pred.min())
            j=1
            if j ==1:
                ground_depth = gt

                predicted_depth = pred

                # print(ground_depth.max(),ground_depth.min())
                # print(predicted_depth.max(),predicted_depth.min())

                # depth_predicted = (predicted_depth / 7) * 255
                # depth_predicted = Image.fromarray(depth_predicted.astype(np.uint8))
                # depth_predicted.save(os.path.join('/home/asus/lyndon/program/Image2Depth/results/predicted_depth/', str(i)+'.png'))

                # depth = (depth / 80) * 255
                # depth = Image.fromarray(depth.astype(np.uint8))
                # depth.save(os.path.join('/data/result/syn_real_result/KITTI/ground_truth/{:05d}.png'.format(t_id)))

                predicted_depth[predicted_depth < 1] = 1
                predicted_depth[predicted_depth > 80] = 80



                height, width = ground_depth.shape
                mask = np.logical_and(ground_depth > 1, ground_depth < 50)

                # crop used by Garg ECCV16

                 #   crop = np.array([0.40810811 * height,  0.99189189 * height,
                   #                      0.03594771 * width,   0.96405229 * width]).astype(np.int32)

                # crop we found by trail and error to reproduce Eigen NIPS14 results
                #elif args.eigen_crop:
                crop =np.array([0.3324324 * height,  0.91351351 * height,
                                     0.0359477 * width,   0.96405229 * width]).astype(np.int32)

                crop_mask = np.zeros(mask.shape)
                crop_mask[crop[0]:crop[1],crop[2]:crop[3]] = 1
                mask = np.logical_and(mask, crop_mask)

                abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i] = compute_errors(ground_depth[mask],predicted_depth[mask])

                print('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
                      .format(i, abs_rel[i], sq_rel[i], rmse[i], rmse_log[i], a1[i], a2[i], a3[i]))
                i+=1
    print ('{:>10},{:>10},{:>10},{:>10},{:>10},{:>10},{:>10}'.format('abs_rel','sq_rel','rmse','rmse_log','a1','a2','a3'))
    print ('{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f},{:10.4f}'
           .format(abs_rel.sum()/i,sq_rel.sum()/i,rmse.sum()/i,rmse_log.sum()/i,a1.sum()/i,a2.sum()/i,a3.sum()/i))
    return  abs_rel.sum()/i,sq_rel.sum()/i,rmse.sum()/i,rmse_log.sum()/i,a1.sum()/i,a2.sum()/i,a3.sum()/i

if __name__ == '__main__':
    eval_metric()