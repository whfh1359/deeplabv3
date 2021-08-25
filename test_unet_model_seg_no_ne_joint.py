# This Python 3 environment comes with many helpful analytics libraries installed

import numpy as np  # linear algebra
import random, imutils
import matplotlib.pyplot as plt
import os, math
import tensorflow as tf
import cv2, time
from only_model_advanced import *
from image_util import *
from loss_function import *


def run_model(gpu=1,
              ckpt_path='',
              save_img_path='',
              img_path='',
              img_size=280,
              xml_dir='',
              sum_avg='',
              ):
    # os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
    do_ne = False
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print('gpus', gpus)
    if gpus:
        try:
            # Restrict TensorFlow to only use the fourth GPU
            tf.config.experimental.set_visible_devices(gpus[0], 'GPU')

            # Currently, memory growth needs to be the same across GPUs
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)
    a = os.listdir(ckpt_path)
    b = []
    for i in a:
        try:
            tmp = i[:i.rindex('_')]
            b.append(float(tmp[tmp.index('_') + 1:]))
        except:
            pass

    max_check = max(b)
    print(max_check)
    ckpt_name = [x[:x.rindex('.')] for x in a if str(max_check) in x]
    ckpt_name = ckpt_name[0]
    ckpt_path = ckpt_path + '/' + ckpt_name
    result_iou = 0.0
    result_iou_cnt = 0
    save_img_path = save_img_path + '/'
    batch_size = 1
    totaliouvalue = 0.0
    totalioucnt = 0
    totalpixel_acc_value = 0.0
    totalpixel_acc_cnt = 0
    total_iou_acc_value = 0.0
    total_iou_acc_cnt = 0
    dump = False
    seg_num_class = 1
    filter_cnt = 32
    for_save = False
    xml_dir = xml_dir + '/'
    seed_list = [random.randint(1, 100) for x in range(10)]

    batch_size = 1
    totaliouvalue = 0.0
    totalioucnt = 0
    totalpixel_acc_value = 0.0
    totalpixel_acc_cnt = 0
    total_iou_acc_value = 0.0
    total_iou_acc_cnt = 0
    dump = False
    seg_num_class = 1
    filter_cnt = 32
    for_save = False
    # xml_dir = '/media/cres04/DATA/fracture/segmentation_train/cross_train/scaphoid/total_train/xml/val_0' + '/'
    xml_dir = xml_dir + '/'
    # sum_avg = 'sum'
    model_name = deeplab_v3
    X_start = 'minmax'

    auto_iou_check = False
    cam_threshold = 50

    def makedir(path):
        if not os.path.exists(path):
            os.makedirs(path)

    def set_category(path, category_list):
        train_img_path = path + '/'
        category = [train_img_path + x for x in category_list]

        return category

    # if do_ne:
    #     val_img_path = set_category(img_path, ['negative', 'positive'])
    # else:
    #     val_img_path = set_category(img_path, ['img'])
    val_img_path = img_path + '/'

    # val_img_path, val_mask_path = set_category('/media/cres04/DATA/fracture/segmentation_train/cross_train/scaphoid/total_train/val_0_1370', ['negative', 'positive'])

    # /media/j/DATA/fracture/segmentation_frac/frac_label/la_ulna/val
    # /media/j/DATA/fracture/segmentation_frac/frac_label/styloid/val
    # /media/j/DATA/fracture/segmentation_frac/frac_label/sr_train/val
    # /media/j/DATA/fracture/segmentation_frac/frac_label/radius/val

    print(val_img_path)

    def my_load(img_path, mask_img_path, img_precess):
        # left_img_data = []
        # left_mask_data = []
        # right_img_data = []
        # right_mask_data = []
        img_data = []
        mask_data = []
        print(img_path)

        for ori_img, mask_img in zip(img_path, mask_img_path):
            # print(one_img)
            img = cv2.imread(ori_img)
            mask = cv2.imread(mask_img, 0)
            print(mask_img)
            image_aug = img_precess  # random.randint(0, 4)
            # left_img = img[0:img.shape[0], 0:int(img.shape[1] * 0.5)]
            # left_mask = mask[0: mask.shape[0], 0:int(mask.shape[1] * 0.5)]
            # right_img = img[0:img.shape[0], int(img.shape[1] * 0.5):img.shape[1]]
            # right_mask = mask[0: mask.shape[0], int(mask.shape[1] * 0.5):mask.shape[1]]
            # left_img = img[0:img.shape[0], 0:int(img.shape[1] * 0.6)]
            # left_mask = mask[0: mask.shape[0], 0:int(mask.shape[1] * 0.6)]
            # right_img = img[0:img.shape[0], int(img.shape[1] * 0.4):img.shape[1]]
            # right_mask = mask[0: mask.shape[0], int(mask.shape[1] * 0.4):mask.shape[1]]
            if dump:
                dump_mask = (255 - mask)
                dump_mask = np.expand_dims(dump_mask, axis=2)
                # plt.imshow((255-mask))
                # plt.show()
                mask = np.expand_dims(mask, axis=2)
                mask = np.concatenate((mask, dump_mask), axis=2)
            else:
                mask = np.expand_dims(mask, axis=2)
                # left_mask = np.expand_dims(left_mask, axis=2)
                # right_mask = np.expand_dims(right_mask, axis=2)

                if img_precess == 0:
                    # print(image_aug)
                    pass
                elif img_precess == 1:
                    img = np.flip(img, 1)
                    mask = np.flip(mask, 1)
                    # left_img = np.flip(left_img, 1)
                    # right_img = np.flip(right_img, 1)
                    # left_mask = np.flip(left_mask, 1)
                    # right_mask = np.flip(right_mask, 1)

                    # _, mask = cv2.threshold()
                    # mask = np.expand_dims(mask, axis=2)
                    # print(mask.shape)
                    # mask_show = np.squeeze(mask,axis=2)
                    # plt.imshow(mask_show)
                    # plt.show()

                # print(img.shape)
                # print(mask.shape)
                # mask_show = np.squeeze(mask, axis=2)

                # plt.imshow(img)
                # plt.show()
                # plt.imshow(mask_show)
                # plt.show()
                #
                # print(img.shape)
                # print(mask.shape)
                img = cv2.resize(img, (img_size, img_size))
                mask = np.expand_dims(mask, axis=2)
                # left_img = cv2.resize(left_img, (input_size, input_size))
                # left_mask = cv2.resize(left_mask, (input_size, input_size))
                # left_mask = np.expand_dims(left_mask, axis=2)
                # right_img = cv2.resize(right_img, (input_size, input_size))
                # right_mask = cv2.resize(right_mask, (input_size, input_size))
                # right_mask = np.expand_dims(right_mask, axis=2)
                # print(img.shape)
                # print(mask.shape)
                img_data.append(np.array(img))
                mask_data.append(np.array(mask))
                # left_img_data.append(np.array(left_img))
                # left_mask_data.append(np.array(left_mask))
                # right_img_data.append(np.array(right_img))
                # right_mask_data.append(np.array(right_mask))
            return np.array(img_data), np.array(mask_data)
            # return np.array(left_img_data), np.array(left_mask_data), np.array(right_img_data), np.array(
            #     right_mask_data)

    def list_load(img_path):
        print('img_path', img_path)
        img_list = sorted(os.listdir(img_path))
        img_list = [img_path + '/' + x for x in img_list]

        return img_list

    if do_ne:
        ne_X_val = list_load(val_img_path[0])
        po_X_val = list_load(val_img_path[1])

        X_val = ne_X_val + po_X_val
    else:
        po_X_val = list_load(val_img_path + 'img')
        Y_val = list_load(val_img_path + 'img_mask')
        X_val = po_X_val

    val_total_cnt = len(X_val)

    def seg_miou(predict, gt_label):
        union = 0
        inse = 0
        for col in range(predict.shape[0]):
            for row in range(predict.shape[1]):
                if gt_label.item(col, row) != 0 or predict.item(col, row) != 0:
                    union += 1
                if gt_label.item(col, row) != 0 and predict.item(col, row) == gt_label.item(col, row):
                    inse += 1

        print('inse : ', 2 * inse)
        print('union : ', union + inse)
        # return (2 * inse) / (union + inse)
        return (inse) / (union)

    def cal_iou(predict, gt_label):
        totalcnt = 0
        correct = 0
        # gt_label = cv2.imread(gt_label, cv2.IMREAD_GRAYSCALE)
        print(predict.shape)
        print(gt_label.shape)
        for col in range(predict.shape[0]):
            for row in range(predict.shape[1]):
                if gt_label.item(col, row) == predict.item(col, row):
                    correct += 1
                totalcnt += 1
                # if gt_label.item(col, row) != 0:
                #     totalcnt += 1
                #     if predict.item(col, row) != 0:
                #         correct += 1
        return correct / totalcnt

    def val_next_batch(batch_s, iters):
        count = batch_s * iters
        return X_val[count:(count + batch_s)], Y_val[count:(count + batch_s)]

    input_layer = tf.keras.layers.Input(shape=(img_size, img_size, 3))
    output_layer = model_name(input_layer, minmax=X_start)
    output_layer.trainable = True
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.00001)
    # model.load_weights('/app2/0504_add_data_original_learning_addtrain/iou/1_0.71692_unet')
    model.load_weights(ckpt_path)
    display_count = 1
    best_val_iou = 0
    total_train_loss = 0
    total_train_iou = 0
    val_iou_dict = {}

    total_train_heatmap_loss_list = []
    total_val_heatmap_loss_list = []
    total_train_loss_list = []

    # train_batch_cnt = math.ceil(train_total_cnt // batch_size)
    val_batch_cnt = math.ceil(val_total_cnt // batch_size)
    val_left_total_iou, val_right_total_iou, val_count = 0, 0, 0
    for i in range(val_batch_cnt):
        left_val_batch_X = []
        left_val_batch_Y = []
        right_val_batch_X = []
        right_val_batch_Y = []
        flip_left_val_batch_X = []
        flip_left_val_batch_Y = []
        flip_right_val_batch_X = []
        flip_right_val_batch_Y = []
        # if (i % 10 == 0) and not i == 0:
        # show img
        # for q,c in zip(predict,y):
        #     # a = np.squeeze(i,axis=2)
        #     # a = np.cast[np.uint8](a)
        #
        #     print(q.shape)
        #     out = np.squeeze(q,axis=2)
        #     out1 = np.squeeze(c,axis=2)
        #     # _, out = cv2.threshold(out,0.5,255,cv2.THRESH_BINARY)
        #     # _, out1 = cv2.threshold(out1, 0.5, 255, cv2.THRESH_BINARY)
        #     # a = i * 255
        #     plt.imshow(out1)
        #     plt.show()
        #     plt.imshow(out)
        #     plt.show()

        # val_iter = (val_total_cnt // batch_size) + 1
        print(batch_size)
        print(val_batch_cnt)

        val_batch_X, val_batch_Y = val_next_batch(batch_size, i)
        # val_batch_Y = val_next_batch(Y_val, batch_size)
        print(val_batch_X)
        print(val_batch_Y)
        img_X, img_Y = my_load(val_batch_X, val_batch_Y, 0)
        print(img_X.shape)
        # leftimg_X, leftimg_Y, rightimg_X, rightimg_Y = my_load(val_batch_X, val_batch_Y, 0)
        # flip_leftimg_X, flip_leftimg_Y, flip_rightimg_X, flip_rightimg_Y = my_load(val_batch_X, val_batch_Y,
        #                                                                            1)
        # leftimg_X = np.squeeze(leftimg_X, 0)
        # leftimg_Y = np.squeeze(leftimg_Y, 0)
        # rightimg_X = np.squeeze(rightimg_X, 0)
        # rightimg_Y = np.squeeze(rightimg_Y, 0)
        # left_val_batch_X.append(leftimg_X)
        # left_val_batch_Y.append(leftimg_Y)
        # right_val_batch_X.append(rightimg_X)
        # right_val_batch_Y.append(rightimg_Y)
        # with tf.GradientTape() as tape:
        val_y_ = model(img_X)

#        plt.imshow(val_y_[0,:,:,:])
#        plt.show()
#        print(val_y_.shape)
#        print(1/'1')

        img_path = val_batch_X[0]
        x_original = cv2.imread(img_path)
        # left_x_original = x_original[0:x_original.shape[0], 0:int(x_original.shape[1] * 0.5)]
        #
        # right_x_original = x_original[0:x_original.shape[0], int(x_original.shape[1] * 0.5):x_original.shape[1]]
        # left_x_original = x_original[0:x_original.shape[0], 0:int(x_original.shape[1] * 0.6)]
        #
        # right_x_original = x_original[0:x_original.shape[0], int(x_original.shape[1] * 0.4):x_original.shape[1]]

        img_name = img_path[img_path.rindex('/') + 1:]

        # flip_left_val_predict = flip_left_val_predict[::-1, :, :]
        # flip_right_val_predict = flip_right_val_predict[::-1, :, :]
        # print('/app2/0504_add_data_original_learning_addtrain/img_result/' + img_name)
        val_y_ = np.squeeze(val_y_, axis=0)
        val_y_ = np.copy(val_y_)
        print(type(val_y_))
        print(val_y_.shape)
        # val_y_ = np.squeeze(val_y_, axis=2)
        # val_y_[val_y_ >= 0.5] = 1
        # val_y_[val_y_ < 0.5] = 0
        val_y_ = val_y_ * 255
        print(np.unique(val_y_))
        # cv2.imwrite('/app2/0504_add_data_original_learning_addtrain/img_result/' + img_name, val_y_)
        # val_y_[val_y_ >= 0.5] = 1.
        # val_y_[val_y_ < 0.5] = .0
        #
        # val_left_total_iou += seg_miou(flip_temp_left_predict, temp_left_batch_Y)
        # val_right_total_iou += seg_miou(flip_temp_right_predict, temp_right_batch_Y)

        max_list = []
        val_count += 1

        if sum_avg == 'sum':
            # if 'positive' in img_name:
            val_predict = np.sum(val_predict, 0)
        elif sum_avg == 'avg':
            # left_val_predict = np.average(left_val_predict, 0)
            # print(left_val_predict)
            # right_val_predict = np.average(right_val_predict, 0)
            # print(right_val_predict)
            val_predict = np.average(val_y_, 0)
        elif sum_avg == 'avg_zero':
            # val_predict = np.average(np.squeeze(val_predict, axis=3),0)
            c = []
            for x in range(val_predict.shape[0]):
                print('shape', val_predict[x, :, :].shape)
                c.append(list(val_predict[x, :, :].flatten()))

            new_img = []

            for i in range(len(c[0])):
                sum_cnt = 0
                zero_cnt = 0
                for k in c:
                    if k[i] == 0:
                        zero_cnt += 1
                    sum_cnt += k[i]
                if zero_cnt == len(c):
                    new_img.append(0)
                else:
                    new_img.append(sum_cnt / (len(c) - zero_cnt))

            val_predict = np.reshape(new_img, (val_predict.shape[1], val_predict.shape[2]))
            print(val_predict)
            val_predict = val_predict.astype(np.float32)
        elif sum_avg in ['max2', 'max3', 'max4']:
            check_val_predict = copy.deepcopy(val_predict)
            new_val_predict = []
            cnt = int(sum_avg[-1])

            for i in range(cnt):
                check_val_predict = np.sort(check_val_predict, axis=0)[::-1]
                # print('check_val_predict',check_val_predict.shape)
                val_predict_max = np.max(check_val_predict, axis=0)
                new_val_predict.append(val_predict_max)
                check_val_predict[0, :, :] = 0
            new_val_predict = np.array(new_val_predict)
            val_predict = np.mean(new_val_predict, axis=0)
        elif sum_avg == 'max':
            val_predict = np.max(val_predict, axis=0)

        y_predict_resize = cv2.resize(val_y_, (x_original.shape[1], x_original.shape[0]))


        # if 'positive' in img_path:
        #     plt.imshow(y_predict_resize)
        #     plt.show()
        y_predict_resize = np.stack([y_predict_resize, y_predict_resize, y_predict_resize], axis=2)
        #print('y_predict_resize:', y_predict_resize)
        # print('time', end - start)

        if auto_iou_check:
            if xml_dir:
                if 'positive' in img_path:
                    print(img_path)
                    # try:
                    xml = xml_dir + img_name.replace('.jpg', '.xml').replace('.png', '.xml')
                    # xml = os.path.join(img_path.replace('.jpg', '.xml').replace('.png', '.xml'))
                    print('yes')
                    f = open(xml, 'r')
                    read = f.readlines()
                    for i in read:
                        if 'xmin' in i:
                            xmin = int(
                                i.replace('\t', '').replace('<xmin>', '').replace('</xmin>', '').replace('\n',
                                                                                                         ''))
                        elif 'ymin' in i:
                            ymin = int(
                                i.replace('\t', '').replace('<ymin>', '').replace('</ymin>', '').replace('\n',
                                                                                                         ''))
                        elif 'xmax' in i:
                            xmax = int(
                                i.replace('\t', '').replace('<xmax>', '').replace('</xmax>', '').replace('\n',
                                                                                                         ''))
                        elif 'ymax' in i:
                            ymax = int(
                                i.replace('\t', '').replace('<ymax>', '').replace('</ymax>', '').replace('\n',
                                                                                                         ''))
                    f.close()
                    y_predict_resize_normalize = y_predict_resize * 255

                    # y_predict_resize_normalize = None
                    # y_predict_resize_normalize = cv2.normalize(y_predict_resize, y_predict_resize_normalize, 0, 255,
                    #                                            cv2.NORM_MINMAX)
                    y_predict_resize_normalize = np.cast[np.uint8](y_predict_resize_normalize)
                    # if 'posi_1370' in img_name:
                    #     plt.imshow(y_predict_resize_normalize)
                    #     plt.show()

                    # plt.imshow(y_predict_resize_normalize)
                    # plt.show()
                    # print(y_predict_resize_normalize)

                    crop_heatmap = y_predict_resize_normalize[ymin:ymax, xmin:xmax]
                    # if 'posi_1370' in img_name:
                    #     plt.imshow(crop_heatmap)
                    #     plt.show()

                    print(y_predict_resize_normalize.shape)
                    print('shape', crop_heatmap.shape)
                    # cv2.imwrite(save_path + 'heatmap_' + label_name[int(class_idx)] + '/crop_' + file_name[k],
                    #             crop_heatmap)

                    _, original_thresh = cv2.threshold(y_predict_resize_normalize, cam_threshold, 255,
                                                       cv2.THRESH_BINARY)
                    original_cam = int(np.sum(original_thresh) / 255)

                    _, crop_thresh = cv2.threshold(crop_heatmap, cam_threshold, 255, cv2.THRESH_BINARY)
                    crop_cam = int(np.sum(crop_thresh) / 255)

                    total_cam_in_gs_cam = round(crop_cam / (original_cam + 1e-5), 4)
                    total_gs_in_gs_cam = round(crop_cam / ((crop_heatmap.shape[0] * crop_heatmap.shape[1]) + 1e-5),
                                               4)

                    print('a', original_cam)
                    print('b', crop_cam)
                    print(crop_heatmap.shape[0])
                    print(crop_heatmap.shape[1])

                    print(total_cam_in_gs_cam)
                    print(total_gs_in_gs_cam)
                    # cam_result_dic[img_path] = img_path + ' : ' + str(
                    #     total_cam_in_gs_cam) + ' / ' + str(total_gs_in_gs_cam)

                    # except:
                    #     print('no xml')


        # plt.imshow(y_predict_resize)
        # plt.show()

        # print(type(y_predict_resize))
        # print(type(x_original))

        # b = 1.0 - 0.4
        # dst = cv2.addWeighted(y_predict_resize, 0.4, x_original, b, 0)
        # plt.imshow(dst)
        # plt.show()

        save_img_folder = save_img_path + 'img' + '/'
        save_y_true = save_img_path + 'y_true' + '/'
        save_y_pred = save_img_path + 'y_pred' + '/'

        folders = [save_img_folder, save_y_true, save_y_pred]
        [makedir(x + 'positive') for x in folders]

        # if do_ne:
        [makedir(x + 'negative') for x in folders]
        # kl-score 0~4 make left & right

        # cv2.imwrite(save_img_folder + str(cnt) + '.png', x_original)
        # cv2.imwrite(save_y_true + str(cnt) + '.png', true_y * 255)
        img_name = img_path[img_path.rindex('/') + 1:]
        img_path = img_path[:img_path.rindex('/')]
        print(img_name)

        # _, y_predict_resize= cv2.threshold(y_predict_resize,0.5,255, cv2.THRESH_TOZERO)
        # print(y_predict_resize)
        # left_temp_y_predict = left_y_predict_resize.copy()
        predict_copy = np.copy(y_predict_resize)
        # y_predict_resize[y_predict_resize >= 0.5] = 1
        # y_predict_resize[y_predict_resize < 0.5] = 0
        y_predict_resize *= 255
        # y_predict_resize *= 255
        # predict_copy[predict_copy >= 0.5] = 1
        # predict_copy[predict_copy < 0.5] = 0
        # predict_copy *= 255
        predict_copy = np.clip(predict_copy, 0, 255)
        y_predict_resize = np.cast[np.uint8](y_predict_resize)
        # y_predict_resize = np.cast[np.uint8](y_predict_resize)
        height, width = y_predict_resize.shape[:2]
        y_predict_resize = cv2.cvtColor(y_predict_resize, cv2.COLOR_BGR2GRAY)
        predict_copy = np.cast[np.uint8](predict_copy)
        y_predict_resize_copy = cv2.cvtColor(predict_copy, cv2.COLOR_BGR2GRAY)
        if 'pilot' in img_path:
            r_threshold = 128
            # gap = int((x_original.shape[1] * 0.5) - (x_original.shape[1] * 0.4))
            #
            # left_y_predict_resize = left_y_predict_resize[:, 0:left_y_predict_resize.shape[1] - gap]
            # right_y_predict_resize = right_y_predict_resize[:, 0 + gap:right_y_predict_resize.shape[1]]
            _, total_thres = cv2.threshold(y_predict_resize, 0, 1, cv2.THRESH_BINARY)
            _, r_thres = cv2.threshold(y_predict_resize, r_threshold, 255, cv2.THRESH_BINARY)
            # _, left_total_thres = cv2.threshold(left_y_predict_resize, 0, 1, cv2.THRESH_BINARY)
            # _, left_r_thres = cv2.threshold(left_y_predict_resize, r_threshold, 255, cv2.THRESH_BINARY)
            # _, right_total_thres = cv2.threshold(right_y_predict_resize, 0, 1, cv2.THRESH_BINARY)
            # _, right_r_thres = cv2.threshold(right_y_predict_resize, r_threshold, 255, cv2.THRESH_BINARY)
            # sum_predict = np.sum(total_thres)
            # # left_sum_predict = np.sum(left_total_thres)
            # # right_sum_predict = np.sum(right_total_thres)
            # r_thres = np.sum(r_thres)
            # # right_r_thres = np.sum(right_r_thres)
            # #
            # if sum_predict != 0:
            #     check_heatmap = r_thres / sum_predict
            # else:
            #     check_heatmap = 0
            # if left_sum_predict != 0:
            #     left_check_heatmap = left_r_thres / left_sum_predict
            # else:
            #     left_check_heatmap = 0
            # if right_sum_predict != 0:
            #     right_check_heatmap = right_r_thres / right_sum_predict
            # else:
            #     right_check_heatmap = 0

            # print('sum_predict', sum_predict, 'r_thres', r_thres)
            # print('check_heatmap', check_heatmap)
            # predict_copy *= 255
            print(np.unique(predict_copy))
            heatmap = cv2.applyColorMap(predict_copy, cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            # left_heatmap = cv2.applyColorMap(left_y_predict_resize, cv2.COLORMAP_JET)
            # # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            heatmap[(heatmap[:, :, 0] == 0) & (heatmap[:, :, 1] == 0) & (heatmap[:, :, 2] != 0)] = 0
            #
            # right_heatmap = cv2.applyColorMap(right_y_predict_resize, cv2.COLORMAP_JET)
            # # heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
            # right_heatmap[
            #     (right_heatmap[:, :, 0] != 0) & (right_heatmap[:, :, 1] == 0) & (right_heatmap[:, :, 2] == 0)] = 0
            # temp_heatmap = copy.deepcopy(heatmap)
            # r_threshold = 1
            # r_heatmap = temp_heatmap[:, :, 2]
            # r_heatmap[r_heatmap > r_threshold] = 1
            # r_heatmap = np.sum(r_heatmap)
            #
            # temp_heatmap[temp_heatmap > 0] = 1
            # temp_heatmap = np.sum(temp_heatmap)
            # print('temp_heatmap', temp_heatmap, 'r_heatmap', r_heatmap)
            # # plt.imshow(heatmap)
            # # plt.show()
            #
            # if temp_heatmap > 0:
            #     check_heatmap = r_heatmap / temp_heatmap
            #     # print('check_heatmap', check_heatmap)
            # else:
            #     check_heatmap = 0
            # print(right_x_original.shape)
            # print(type(right_x_original))
            # print(right_heatmap.shape)
            # print(type(right_heatmap))
            heatmap = cv2.resize(heatmap, (x_original.shape[1], x_original.shape[0]))
            # right_heatmap = cv2.resize(right_heatmap, (right_x_original.shape[1], right_x_original.shape[0]))
            # superimposed_img = cv2.addWeighted(left_x_original, 0.8, left_heatmap, 0.2, 0)
            # # right_superimposed_img = cv2.addWeighted(right_x_original, 0.8, right_heatmap, 0.2, 0)
            #
            # gap = int((x_original.shape[1] * 0.5) - (x_original.shape[1] * 0.4))
            #
            # superimposed_img = y_predict_resize[:, 0:y_predict_resize.shape[1] - gap]
            #
            #
            # # right_x_original = x_original[0:x_original.shape[0], int(x_original.shape[1] * 0.4):x_original.shape[1]]
            # # left_superimposed_img = left_superimposed_img[:, 0:int(left_superimposed_img.shape[1] * 0.8)]
            # # right_superimposed_img = right_superimposed_img[:, int(right_superimposed_img.shape[1] * 0.2):right_superimposed_img.shape[1]]
            #
            # overlayimg = cv2.hconcat([left_superimposed_img, right_superimposed_img])
            # cv2.imwrite(save_y_pred + 'positive/right_' + img_name, right_superimposed_img)
            cv2.imwrite(save_y_pred + 'positive/' + img_name, predict_copy)

        # wr = csv.writer(w)
        # wr.writerow([img_name,check_heatmap])

        # plt.imshow(heatmap)
        # plt.show()

    # w.close()
    print('left iou : ', val_left_total_iou / val_count)
    print('right iou : ', val_right_total_iou / val_count)
    print(val_left_total_iou / val_count)
    # print(val_acc)
    print(val_total_cnt)
    end = time.time()
    # print('val loss :', val_total_loss / (val_total_cnt))
    # print('val iou : ', val_total_iou / (val_total_cnt))
    # print('val acc : ', val_acc / (val_total_cnt))
    # print('total pixel_acc : ', totaliouvalue / totalioucnt)
    # # print('total iou : ', totalpixel_acc_value / totalpixel_acc_cnt)
    # print('total iou 0.5 : ', total_iou_acc_value / total_iou_acc_cnt)
    # f = open(save_img_path + 'result.txt', 'a')
    # f.write(str(end) + '\n')
    # f.close()
    # print('start spend time : ' + str(start))
    # print('end spend time : ' + str(end))
    # print("WorkingTime: {} sec".format(end - start))
    # # print('val classification loss : ', val_total_c_lo / ((val_total_cnt // batch_size) + 1))
    #
    # # print(str(display_count), 'time : ', round(train_time, 5), " training loss:", str(loss_value),
    # #       str(round(pred_acc, 6)))
    # # print(np.argmax(c_lo,axis=1))
    # # print(np.argmax(cy,axis=1))
    #
    # display_count += 1
    #

    # batch_count += 1
    # print(batch_count)
    #
    # if xml_dir:
    #     w = open(ckpt_path[:ckpt_path.rindex('/') + 1] + str(cam_threshold) + '_' + sum_avg + '_cam_result.txt', 'w')
    #
    #     w.write('\t\t\t gs_cam/total_cam \t gs_cam/total_gs\n')
    #     for i in cam_result_dic.values():
    #         w.write(i)
    #         w.write('\n')
    #     w.close()
    #
    # path = ckpt_path[:ckpt_path.rindex('/') + 1] + str(cam_threshold) + '_' + sum_avg + '_cam_result.txt'
    #
    # f = open(path,'r')
    # a = f.readlines()
    #
    # a = [x.replace('\t','').replace('\n','').replace(' ','') for x in a[1:]]
    #
    # def my_sort(i):
    #     return float(i[i.rindex(':') + 1:i.rindex('/')])
    #
    # # for i in a:
    # #     print(i)
    # #     print(i[i.rindex(':') + 1:i.rindex('/')])
    #
    # a = sorted(a,key=my_sort,reverse=True)
    #
    # total_cnt = len(a)
    # cnt = 0
    #
    # for i in a:
    #     print(i)
    #     if float(i[i.rindex(':') + 1:i.rindex('/')]) >= 0.1 and float(i[i.rindex('/') + 1:]) >= 0.1:
    #         cnt += 1
    #
    # #print(cnt / total_cnt)
    # print(cnt, total_cnt)
    #
    # f.close()
    #
