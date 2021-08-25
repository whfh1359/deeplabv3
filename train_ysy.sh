python -c "from unet_model_seg_no_ne_noise import run_model; \
run_model(1, \
'/app/train', \
'/app/val', \
280, \
50,
2, \
32, \
'/app/0819_first', \
'deeplab_v3', \
'minmax', \
1, \
'softmax', \
'softmax', \
'no', \
'no', \
'no', \
'')"