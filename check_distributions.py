import os
import os.path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


folder_path = r"/home/apetropc/y_record"

clean_distr = np.load("clean.npy")
keeped_samples = np.load("5000.npy")
clean_distr = clean_distr[keeped_samples]

idx = np.argmax(clean_distr, axis =1)
idx = torch.tensor(idx)

# number of samples 
N = 5000

start_y = 0 # 65
end_y = 135
acc_list = []


for k in range(start_y,end_y) :
    num_correct = 0

    if(k>=100):
        init_str = "y_"
    elif(k<10):
        init_str = "y_00"
    else:
        init_str = "y_0"

    current_distr = np.load("record/"+init_str+str(k)+".npy")
    current_distr = torch.tensor(current_distr)
    correct = torch.eq(torch.max(F.softmax(current_distr, dim = 0), dim=1)[1],idx).view(-1)
    num_correct += torch.sum(correct).item()
    acc = num_correct/N
    acc_list.append(acc)



    
# visualize the loss as the network trained
fig = plt.figure(figsize=(10,8))
plt.plot(range(start_y,end_y),acc_list, label='Corrected Labels with PENCIL')

# find position of lowest validation loss
#minposs = valid_loss.index(min(valid_loss))+1 
#plt.axvline(minposs, linestyle='--', color='r',label='Early Stopping Checkpoint')

plt.xlabel('epochs')
plt.ylabel('Correct Labels')
plt.autoscale()
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
fig.savefig('correct_labels_plot.png', bbox_inches='tight')
