import os, sys, cv2
import numpy as np
import shutil
mask_path = '/app2/train/img_mask'
img_path = '/app2/train/img'
save_path = '/app2/train/overlay_total'
# """
# overlay original image and mask image
# """
# if not os.path.isdir('/app2/train/overlay_total/'):
#     os.makedirs('/app2/train/overlay_total/')
# masks = os.listdir(mask_path)
# imgs = os.listdir(img_path)
# print('mask : ', len(masks))
# print('img : ', len(imgs))
#
# for img_name in imgs:
#     mask_img = cv2.imread(os.path.join(mask_path, img_name))
#     img = cv2.imread(os.path.join(img_path, img_name))
#     #
#     # print(mask_img.shape)
#     # print(np.unique(mask_img))
#     # print(img.shape)
#     # print('---')
#     mask_img *= 255
#     superimposed_img = cv2.addWeighted(img, 0.8, mask_img, 0.2, 0)
#     cv2.imwrite(os.path.join(save_path, img_name), superimposed_img)

exclude_path = '/app2/train/re_draw'

# exclude_list = ['9339860L96', '9355257R24', '9372474L72', '9373191L12', '9377882R24', '9379276R00', '9379540L12', '9385075R36', '9395979R00', '9395979R12', '9401202R12', '9405933R24', '9406962L00', '9409941R12',
#                 '9410340L24', '9413071R12', '9420706R72', '9421492R00', '9435335R00', '9436426L36', '9438523R12', '9438852L12', '9438852L36', '9439428R00', '9440518L12', '9449834L48', '9449834R48', '9451114L12', '9451114L36',
#                 '9452100L00', '9452100L12', '9456244R48', '9466438L36', '9467252L48', '9467311L00', '9477205L00', '9498865L12', '9508335R36', '9509417R12', '9510943L00', '9513860R24', '9539593R36', '9540883L12', '9547902L12',
#                 '9568974L36', '9573434R24', '9581253R48', '9604541R24', '9686590L48', '9703327R36', '9725978L36', '9745368R12', '9872052L36', '9911575L48', '9959640L24']
# for idx, name in enumerate(exclude_list):
#     exclude_list[idx] = name + '.png'
#
# imgs = os.listdir(img_path)
# # masks = os.listdir(mask_path)
# for img_name in imgs:
#     # print(img_name)
#     if img_name in exclude_list:
#         shutil.copy(os.path.join(img_path, img_name), os.path.join(exclude_path, img_name))
        # shutil.move(os.path.join(mask_path, img_name), os.path.join(exclude_path, (img_name.split('.png')[0] + '_mask.png')))
# re_draw_folders = os.listdir(exclude_path)
# for json_name in re_draw_imgs:
#     if '.json' in json_name:
#         # print(json_name)
#         os.system('labelme_json_to_dataset ' + os.path.join(exclude_path, json_name) + ' -o ' + os.path.join(exclude_path, json_name.split('.json')[0]))
re_draw_folders = '/app2/train/re_draw/masks'
masks = os.listdir(re_draw_folders)

for mask_name in masks:
    mask = cv2.imread(os.path.join(re_draw_folders, mask_name), cv2.IMREAD_GRAYSCALE)
    print(np.unique(mask))
    # img_col, img_row = mask.shape
    #
    # for col in range(0, img_col):
    #     for row in range(0, img_row):
    #         if mask.item(col, row) != 0:
    #             mask.itemset(col, row, 1)
    # print(np.unique(mask))
    # cv2.imwrite(os.path.join(re_draw_folders, mask_name), mask)
    #



