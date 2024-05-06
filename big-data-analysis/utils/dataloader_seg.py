import os
import numpy as np
from tqdm import tqdm # TqdmExperimentalWarning
from scipy.signal import cheby2, lfilter, resample
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
import torch
from torch.utils.data import Dataset
'''
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
'''


class LoadDataset_from_numpy(Dataset):
    # Initialize your data, download, etc.
    def __init__(self, np_dataset, num_classes=5, esize=12, channel=0,
                 augmentation=None, 
                 max_shift=0.15, ):

        super(LoadDataset_from_numpy, self).__init__()
        X = []
        y = []
       #  mms = StandardScaler() #  MinMaxScaler() #RobustScaler()
        print('No. of Dataset', len(np_dataset), 'Epoch size', esize)
        
        for _, np_file in enumerate(tqdm(np_dataset)):
            
            file = np.load(np_file, allow_pickle=True)
            
            if file['x'].shape[2] > 1: 
                
                raw, label = file['x'][:, :, channel], file['y']   
                if len(raw.shape) < 3: 
                    raw = np.expand_dims(raw, 2)
                    #print(raw.shape)  
                # raw = raw[:, :, channel]   
            else: 
                raw, label = file['x'], file['y'] 
            
            num_sig = raw.shape[0] 
            sig_len = raw.shape[1] 
            
            
            fs = sig_len / 30 
            if augmentation is not None:

                shiftno = 10 
                shifts = np.linspace(0, max_shift, num=shiftno) 

                for shift_value in shifts: 
                    
                    shift_amount = int(sig_len * shift_value)                
                    for sigidx in range(num_sig):
                        if sigidx < esize:
                            rep = np.tile(raw[:1,:,:], (esize - sigidx - 1,1,1)) # 
                            value = np.concatenate((rep, raw[:sigidx+1,:,:]),axis=0).reshape(-1, 1) 
                        else:
                            value = raw[sigidx - esize: sigidx, :, :].reshape(-1, 1)                     
                        if (shift_amount > 0) & (sigidx >= esize): 
                            # values from previous epoch to append in front 

                            sig_to_append = raw[sigidx-esize-1] 
                            start = sig_len - shift_amount
                            sig_to_append = sig_to_append[start:, :]
                        
                            # crop values at the end (Current signal)
                            value = value[:-shift_amount] 
                            value = np.concatenate([sig_to_append, value])  

                        X.append(value) # (1500,1) 
                        y.append(label[sigidx])
            else:
                # for sigidx in range(esize + 1, num_sig - esize):   
                #     value = raw[sigidx - esize: sigidx, :, :].reshape(-1, 1) 
                for sigidx in range(num_sig):
                    if sigidx < esize:
                        rep = np.tile(raw[:1,:,:], (esize - sigidx - 1,1,1)) # 
                        value = np.concatenate((rep, raw[:sigidx+1,:,:]),axis=0).reshape(-1, 1) 
                    else:
                        value = raw[sigidx - esize: sigidx, :, :].reshape(-1, 1)
                    X.append(value) 
                    y.append(label[sigidx]) 
        # self.c = file['cols'][channel]
        # print('CHANNEL SELECTED', self.c)
        X_train = np.stack((X), axis=0) 
        y_train = np.array(y)
        print('Shape Before Augmentation:', X_train.shape, y_train.shape)

        ## Classes
        if num_classes == 4:
            y_train = np.array([convert4class(x) for x in y_train])
        elif num_classes == 3:
            y_train = np.array([convert3class(x) for x in y_train])        
        elif num_classes == 5:
            y_train = y_train
        else:
            print("class 개수가 잘못되었습니다.")

        bin_labels = np.bincount(y_train)
        print(f"Labels count before Augmentation: {bin_labels}. \
              Major percentage: {100*np.max(bin_labels)/np.sum(bin_labels):.3f}")
        
        ## Augmentation        
        X_origin = X_train.copy()
        if augmentation == "SMOTE":
            print(f"Apply {augmentation}")
            X_train, y_train = SMOTE().fit_resample(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]), y_train)
            X_train = X_train.reshape(X_train.shape[0], X_origin.shape[1], X_origin.shape[2])
        elif augmentation == "TLink":
            print(f"Apply {augmentation}")
            X_train, y_train = TomekLinks(sampling_strategy="all", n_jobs=8).fit_resample(X_train.reshape(X_train.shape[0], X_train.shape[1]*X_train.shape[2]), y_train)
            X_train = X_train.reshape(X_train.shape[0], X_origin.shape[1], X_origin.shape[2])
        else:
            pass

        self.len = X_train.shape[0]
        self.x_data = torch.from_numpy(X_train)
        self.y_data = torch.from_numpy(y_train).long()
        self.num_classes = num_classes

        # Correcting the shape of input to be (Batch_size, #channels, seq_len) where #channels=1
        if len(self.x_data.shape) == 3:
            if self.x_data.shape[1] != 1:
                self.x_data = self.x_data.permute(0, 2, 1)
        else:
            self.x_data = self.x_data.unsqueeze(1)
        
        bin_labels = np.bincount(self.y_data)
        print(X_train.shape, y_train.shape)
        print(f"Labels count: {bin_labels}. Major percentage: {100*np.max(bin_labels)/np.sum(bin_labels):.3f}")
        # print(f"Shape of Input : {self.x_data.shape}")
        # print(f"Shape of Labels : {self.y_data.shape}")

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len
    def __getlabel__(self):
        return self.y_data.numpy()

def data_generator_np(training_files, subject_files, batch_size, num_classes = 5, num_workers=4, esize=12, channel=0, augmentation="s"):
    train_dataset = LoadDataset_from_numpy(training_files, num_classes = num_classes, esize=esize, channel=channel, augmentation=augmentation)
    test_dataset = LoadDataset_from_numpy(subject_files, num_classes = num_classes, esize=esize, channel=channel) 

    # to calculate the ratio for the CAL
    all_ys = np.concatenate((train_dataset.y_data, test_dataset.y_data))
    all_ys = all_ys.tolist()
    num_classes = len(np.unique(all_ys))
    counts = [all_ys.count(i) for i in range(num_classes)]
    # c = train_dataset.c 

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True,
                                               drop_last=False,
                                               num_workers=num_workers)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False,
                                              drop_last=False,
                                              num_workers=num_workers)

    return train_loader, test_loader, counts # ,c


## S0(Wake):0, S1:1, S2:2, S3:3, REM:4
def convert3class(y): 
    if y >= 4: ## REM 
        return 2
    ## NON-REM 
    if y == 3: ## DeepSleep 
        return 1
    if y == 2: ## & 1 LightSleep 
        return 1
    else:      # 0 Wake 
        return y

def convert4class(y):
    if y >= 4: ## REM
        return 3
    ## NON-REM 
    if y == 3: ## DeepSleep
        return 2
    if y == 2: ## & 1 LightSleep 
        return 1
    else:      ## wake = 0
        return y

def filt_resam_Signal(sig, filter=True, fs=256, order=8, cutoff_freq=8, dB=40):
    # Param of filter, For MESA
    if filter:
        fs = fs
        order = order
        cutoff_freq = cutoff_freq
        stopband_attenuation = dB

        nyquist_freq = 0.5 * fs
        normalized_cutoff_freq = cutoff_freq / nyquist_freq

        b, a = cheby2(order, stopband_attenuation, normalized_cutoff_freq, btype='low', analog=False)
        sig = lfilter(b, a, sig)
    sig = np.apply_along_axis(lambda x: resample(x, 750), 1, sig)
    return sig

# def convert5class(y):
#     if y >= 4:
#         return 4
#     else:    
#         return y

if __name__=="__main__":
    from glob import glob
    DATA_PATH = "/home/workspace/data/maxim_750_train"
    ## DataPath
    list_files = sorted(glob(os.path.join(DATA_PATH,"*.npz")))
    #list_name = np.unique([x.split('-')[0] for x in list_files])
    
    #train_valid_list = list_name[2:]
    # print(train_valid_list)
    #test_list = [x for x in list_files if x.split('-')[0] in list_name[:2]]
    test_dataset = LoadDataset_from_numpy(list_files, num_classes = 3, esize=12, augmentation="SMOTE")
