import os
import pickle
import random
import argparse
from glob import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import KFold # , LeaveOneOut
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix, accuracy_score

import utils.loss as loss
from utils.train import fit, test
from utils.util import calc_class_weight
from utils.loss import FocalLoss, weighted_CrossEntropyLoss
from utils.dataloader_seg import data_generator_np
from models.TinySleepNet import TinySleepNet

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore
    
def weights_init_normal(m):
    if type(m) == nn.Conv2d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.Conv1d:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif type(m) == nn.BatchNorm1d:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
        
def main(args):
    set_seed(42)
    ## Initial params
    DATA_PATH = f'./data/{args.data_path}' # f"../data/{args.data_path}"
    RUN_NAME = args.save_path
    BATCH_SIZE = args.batch_size
    SAVE_PATH = f'./saved_model/{args.classes}/c{args.channel}/{args.data_path}_{args.model}'
    if args.pretrained:     
        SAVE_PATH = f'./saved_model/{args.classes}/c{args.channel}/pre_{args.pretrained}_{args.data_path}_{args.model}'
        
    if not os.path.exists(SAVE_PATH):
        os.makedirs(SAVE_PATH)
        print('make', SAVE_PATH)
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= f"{args.gpu}"
    DEVICE = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
    print('Device:', DEVICE)
    print('Current cuda device:', torch.cuda.current_device())
    print('Count of using GPUs:', torch.cuda.device_count())
    NUM_CHANNELS = args.classes

    list_files = sorted(glob(os.path.join(DATA_PATH,"*.npz")))
    print(f"Total participants = {len(list_files)}")

    ## Fold    
    n_folds = args.fold 
    splits = KFold(n_splits=5) # LeaveOneOut()
    results = dict()
    all_pred = []
    all_true = []
    NUM_CLASSES = int(args.classes)
    
    for fold_, (train_idx, valid_idx) in enumerate(splits.split(list_files)):
        print(f"Fold {fold_+1} start." )
        if os.path.exists(os.path.join(SAVE_PATH, f"{args.save_path}_hist_fold{fold_+1}.pkl")):
           print(os.path.join(SAVE_PATH, f"{args.save_path}_hist_fold{fold_+1}.pkl"), " already exists.")
           continue
        modelPath = os.path.join(SAVE_PATH, f'{args.save_path}_fold{fold_+1}.pt')
        
        train_list = [list_files[idx] for idx in train_idx]
        valid_list = [list_files[idx] for idx in valid_idx]
        train_loader, valid_loader, counts = data_generator_np(train_list, valid_list, BATCH_SIZE, num_classes = NUM_CHANNELS, esize=args.esize, channel=args.channel, augmentation=args.augmentation)
        weights_for_each_class = calc_class_weight(counts, method = "Scale")
        ## Model
        if args.model == "TinySleepNet":
            model = TinySleepNet(in_channel=args.in_channel, input_size = args.esize*750, num_classes=NUM_CLASSES)
        else:
            print('Model cannot be recognized')
            return 
        model = model.apply(weights_init_normal)

        if args.pretrained:
            pt_folds = 0 
            PretrainPath = os.path.join(f'./saved_model/{NUM_CLASSES}/{args.pretrained}_{args.model}/test_fold{pt_folds+1}.pt') # {args.classes}', f'{args.data_path} 
            model.load_state_dict(torch.load(PretrainPath), strict=False)
            model.dense = nn.Linear(256, args.classes)   
            print("Load pretrained checkpoint", PretrainPath)

        if args.freeze: # for finetuning:
            print("Fine-tuning processed.")
            for param in model.parameters():
                param.requires_grad = True
            for name, param in model.named_parameters():
                if name.__contains__("feature_extract"):
                    param.requires_grad = False

        # train/validate now
        if args.loss == "WCE":
            print('weighted_CrossEntropyLoss')
            criterion = getattr(loss, "weighted_CrossEntropyLoss")
        elif args.loss == "CE":
            criterion = nn.CrossEntropyLoss()
            weights_for_each_class = None
        else :
            criterion = FocalLoss(weight = torch.tensor(weights_for_each_class.copy()).to(DEVICE), device = DEVICE)
            weights_for_each_class = None
        optimizer = torch.optim.Adam(params = model.parameters(),lr = args.initial_lr, weight_decay = 5e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode = 'min', factor = 0.1, patience = 3)
        history = fit(epochs = args.epochs,
                      model = model,
                      train_loader = train_loader,
                      val_loader = valid_loader,
                      criterion = criterion, 
                      optimizer = optimizer, 
                      path = modelPath,
                      class_weights = weights_for_each_class,
                      scheduler= scheduler,
                      earlystop = 10,
                      device = DEVICE)
        results[fold_+1] = history["val_acc"][np.argmin(history["val_loss"])]
        with open(os.path.join(SAVE_PATH, f"{args.save_path}_hist_fold{fold_+1}.pkl"), "wb") as f:
            pickle.dump(history, f)

        ## Valid check for classification report
        
        model.load_state_dict(torch.load(modelPath))
        yPred, yTrue = test(model,valid_loader, DEVICE)
        all_pred.extend(yPred)
        all_true.extend(yTrue)

        del model, train_loader, valid_loader 

    # Print fold results
    t = f'\n{SAVE_PATH} CROSS VALIDATION RESULTS FOR {n_folds} FOLDS'
    t = t + '\n--------------------------------'
    sum = 0.0
    for key, value in results.items():
        t = t + f'\nFold {key}: {value*100} %'
        sum += value
    t = t + f'\nAverage: {sum/len(results.items())*100} %'
    print(t)    

    r = classification_report(all_true, all_pred, digits=6, output_dict=True)
    cm = confusion_matrix(all_true, all_pred)
    df = pd.DataFrame(r)
    df["cohen"] = cohen_kappa_score(all_true, all_pred)
    df["accuracy"] = accuracy_score(all_true, all_pred)
    df = df * 100
    file_name = RUN_NAME + "_classification_report.csv"
    report_Save_path = os.path.join(SAVE_PATH, file_name)
    df.to_csv(report_Save_path)

    cm_file_name = RUN_NAME + "_confusion_matrix.torch"
    cm_Save_path = os.path.join(SAVE_PATH, cm_file_name)
    torch.save(cm, cm_Save_path)
    
    return t

if __name__ == "__main__":
    
    ## Argparse
    parser = argparse.ArgumentParser(description='Sleep_python', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    ## Fix setting
    parser.add_argument('--epochs', default=100, type=int,
                        help='Epochs')
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch Size')

    ## Changable settings
    parser.add_argument('--gpu', default=1, type=int,
                        help='Epochs')    
    parser.add_argument('--data_path', default="mesa_750_norm", type=str,
                        help='Set source(train) path')
    parser.add_argument('--model', default='TinySleepNet', type=str,
                        help='Model')
    parser.add_argument('--esize', default=12, type=int,
                        help='Model')
    parser.add_argument('--initial_lr', default=1e-4, type=float,
                        help='Set initial learning rate')
    parser.add_argument('--classes', default=4, type=int,
                        help='classes')                  
    parser.add_argument('--pt_fold', default=1, type=int,
                        help='Best folds')         
    parser.add_argument('--channel', default=0, type=int,
                        help='Best folds')                
    parser.add_argument('--in_channel', default=1, type=int,
                        help='number of channel')
    parser.add_argument('--fold', default=5, type=int,
                        help='Best folds')                
    parser.add_argument('--pretrained', default=None, type=str,
                        help='Load pretrained model')
    parser.add_argument('--save_path', default='test', type=str,
                        help='Set save path')   
    parser.add_argument('--freeze', default=0, type=int,
                        help='Use fine-tuning')
    parser.add_argument('--augmentation', default=None, type=str,
                        help='Augmentation')      
    parser.add_argument('--loss', default="WCE", type=str,
                        help='Set Loss functions')

    args = parser.parse_args()

    main(args)