import os, sys, cv2

path = '/app2/train/img'
print(len(os.listdir(path)))

for filename in os.listdir(path):
    im = cv2.imread(os.path.join(path, filename))
    print(im.shape)
    cv2.imwrite(os.path.join(path, filename), im)