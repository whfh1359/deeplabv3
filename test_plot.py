import matplotlib.pyplot as plt
import os, sys
if __name__ == '__main__':
   # order = sys.argv[1]
   # train_name = 'train_' + order
   # val_name = 'val_' + order
   train_hitmap = []
   val_hitmap = []
   with open('/home/crescom/Joint_Data/0507_add_data_original_learning_separate_addtrain/trainiou_per_epoch.txt', 'r') as f:
       while True:
           line = f.readline()
           if not line: break
           raw = line.split()
           train_hitmap.append(float(raw[0]))
   display_count = 1
   plt.plot(train_hitmap, marker='', color='blue', label="train_hitmap" if display_count == 1 else "")
   # plt.plot(val_hitmap, marker='', color='orange', label="validation_hitmap_" + order if display_count == 1 else "")
   plt.legend()
   plt.savefig('/home/crescom/Joint_Data/0507_add_data_original_learning_separate_addtrain/train_hitmap.png')