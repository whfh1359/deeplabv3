python -c "from test_unet_model_seg_no_ne_joint import run_model; \
run_model(0, \
'/app/0819_first/iou', \
'/app/0819_first/pilot_result_new', \
'/app/pilot', \
280, \
'', \
'', \
)"

python -c "from test_unet_model_seg_no_ne_joint_separate_oai import run_model; \
run_model(0, \
'/app2/0802_new_img_half_augment_data_relabeled_redraw_data/iou', \
'/app2/0802_new_img_half_augment_data_relabeled_redraw_data/pilot_result_new', \
'/app/keras_retinanet/keras_retinanet/bin/oai_pilot_ori/joint', \
280, \
'', \
'', \
)"


python -c "from unet_model_seg_no_ne_noise import run_model; \
run_model(1, \
'/app2/train', \
'/app2/val', \
280, \
1000,
16, \
32, \
'/app2/0531_image_test', \
'deeplab_v3', \
'minmax', \
1, \
'softmax', \
'softmax', \
'no', \
'no', \
'no', \
'/media/cres04/DATA/fracture/sr_train/kah_add_preprocess_train/histo_sample/00234306_F850000_1.png')"
python -c "from unet_model_seg_no_ne_noise import run_model; \
run_model(1, \
'/app2/train', \
'/app2/val', \
280, \
1000,
8, \
16, \
'/app2/0630_new_img_learning', \
'deeplab_v3', \
'minmax', \
1, \
'softmax', \
'softmax', \
'no', \
'no', \
'no', \
'/media/cres04/DATA/fracture/sr_train/kah_add_preprocess_train/histo_sample/00234306_F850000_1.png')"

python -c "from unet_model_seg_no_ne_noise_separate_area import run_model; \
run_model(1, \
'/app2/train', \
'/app2/val', \
280, \
1000,
8, \
16, \
'/app2/0802_new_img_half_augment_data_relabeled_redraw_data', \
'deeplab_v3', \
'minmax', \
1, \
'softmax', \
'softmax', \
'no', \
'no', \
'no', \
'')"