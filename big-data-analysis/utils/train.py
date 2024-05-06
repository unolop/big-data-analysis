import time
from tqdm import tqdm
from thop import profile 
import torch
import wandb
import numpy as np
from sklearn.metrics import f1_score 

def fit(epochs, model, train_loader, val_loader, criterion, optimizer, path, class_weights = None, scheduler = None, earlystop = 20, device="cuda"):
    ## Initial
    torch.cuda.empty_cache()    
    train_losses = []
    test_losses = []
    val_acc = []
    train_acc = []
    min_loss = np.inf
    not_improve = 0
    warmup = -1
    model.to(device)
    #wandb.watch(model, log="all")
    fit_time = time.time()

    ## Run epochs
    for e in range(epochs):
        since = time.time()
        running_loss = 0       
        running_accuracy = 0 
        
        # training loop
        model.train()
        for _, data in enumerate(tqdm(train_loader)):
            # training phase           
            
            inputs, labels = data
            inputs = inputs.to(device).float()
            labels = labels.to(device).long()
            optimizer.zero_grad()  # reset gradient
            
            # forward            
            outputs = model(inputs)            
            # macs, params = profile(model.to(device), inputs=(inputs, ))    
            _, preds = torch.max(outputs, 1)

            if class_weights is not None:
                loss = criterion(outputs, labels,class_weights,device)    
            else:
                loss = criterion(outputs, labels)

            # backward
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            running_accuracy += torch.sum(preds == labels.data).detach().cpu().numpy()/inputs.size(0)            
            
            
        model.eval()
        val_loss = 0
        val_accuracy = 0
        # validation loop
        with torch.no_grad():
            predictions = []
            true_labels = []               
            for _, data in enumerate(tqdm(val_loader)):  
                
                inputs, labels = data
                inputs = inputs.to(device).float()
                labels = labels.to(device).long()

                # forward
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)

                # evaluation metrics
                # loss
                if class_weights is not None:
                    loss = criterion(outputs, labels,class_weights,device)            
                else:
                    loss = criterion(outputs, labels)

#                loss = criterion(outputs, labels,class_weights,device)
                val_loss += loss.item()
                val_accuracy += torch.sum(preds == labels.data).detach().cpu().numpy()/inputs.size(0)
                predictions.extend(preds.detach().cpu().numpy())  
                true_labels.extend(labels.detach().cpu().numpy())

        # calculate mean for each batch
        train_losses.append(running_loss / len(train_loader))
        test_losses.append(val_loss / len(val_loader))


        ## Warmup and Earlystopping
        if e > warmup:
            if min_loss - (val_loss / len(val_loader)) > 0.001 :
                print('Loss Decreasing {:.3f} >> {:.3f} . Save model to {} '.format(min_loss, (val_loss / len(val_loader)), path))
                unique, counts = np.unique(predictions, return_counts=True)                
                print(f"prediction: {dict(zip(unique, counts))}")
                min_loss = (val_loss / len(val_loader))            
                torch.save(model.state_dict(), path)
                not_improve = 0
                
            else:
                not_improve += 1
                print(f'Loss Not Decrease for {not_improve} time. Current Loss = {min_loss:.3f}')
                if not_improve == earlystop:
                    print(f'Stop Training. Final Loss = {min_loss:.3f}')
                    break
            if scheduler != None:   
                scheduler.step(val_loss)            
        else:
            print(f"Warmup until {warmup} epochs")

        train_acc.append(running_accuracy / len(train_loader))
        val_acc.append(val_accuracy / len(val_loader))
        print("Epoch:{}/{}..".format(e + 1, epochs),
                "Train Loss: {:.3f}..".format(running_loss / len(train_loader)),
                "Val Loss: {:.3f}..".format(val_loss / len(val_loader)),
                "Train Acc:{:.3f}..".format(running_accuracy / len(train_loader)),
                "Val Acc:{:.3f}..".format(val_accuracy / len(val_loader)),
                "Valid F1:{:.3f}..".format(f1_score(true_labels, predictions, average='macro')),
                "Learning Rate:{}..".format(scheduler.optimizer.param_groups[0]['lr']),
                "Time: {:.2f}m".format((time.time() - since) / 60))

        metrics_log = {"train_epoch": e + 1,
                    "Train Loss": running_loss / len(train_loader),
                    "Val Loss": val_loss / len(val_loader),
                    "Train Acc": running_accuracy / len(train_loader),
                    "Val Acc": val_accuracy / len(val_loader),
                    "Valid F1":f1_score(true_labels, predictions, average='macro'),                    
                    "Learning Rate":scheduler.optimizer.param_groups[0]['lr'],
                    "Time": (time.time() - since) / 60
                    }
        #wandb.log(metrics_log)


    history = {'train_loss': train_losses, 'val_loss': test_losses,
               'train_acc': train_acc, 'val_acc': val_acc}
    print('Total time: {:.2f} m'.format((time.time() - fit_time) / 60))
    return history

def test(model, test_loader, device="cuda"):
    model.eval()
    model.to(device)
    yPred = []
    yTrue = []
    with torch.no_grad():
        for _, data in enumerate(tqdm(test_loader)):
            inputs, labels = data                    
            inputs = inputs.to(device).float()            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)                        
            yPred.extend(preds.detach().cpu().numpy())                              
            yTrue.extend(labels.detach().cpu().numpy())                              
    return yPred, yTrue
