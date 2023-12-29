import monai
from dataloader.get_dataloaders import get_dataloaders
import torch
from monai.losses.dice import DiceLoss
import wandb 
from tqdm import tqdm
import torch.nn as nn
from Metrics.calculate_metrics import calculate_metrics
from operator import add
import pandas as pd

def training_model(train_dataloader,val_dataloader, dropout, epochs, learning_rate):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = monai.networks.nets.UNet(
        spatial_dims=3,  # 2 or 3 for a 2D or 3D network
        in_channels=1,  # number of input channels
        out_channels=3,  # number of output channels
        channels=[8, 16, 32],  # channel counts for layers
        strides=[2, 2],  # strides for mid layers
        dropout = dropout
        ).to(device)
    num_epochs = epochs
    loss_function = DiceLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(1,num_epochs+1):
        overall_train_loss_per_epoch = []
        overall_val_loss_per_epoch = []
        overall_train_jaccard_per_epoch = []
        overall_val_jaccard_per_epoch = []
        overall_train_acc_per_epoch = []
        overall_val_acc_per_epoch = []
        
        epoch_train_loss = 0.0
        epoch_val_loss = 0.0  
        metrics_score = [0.0, 0.0]

        model.train()
        for batch_data in tqdm(train_dataloader, desc=f'Training Epoch {epoch}/{num_epochs}', unit='epoch'):
            inputs = batch_data["image"].float().to(device)
    #         print(inputs.shape)
            labels = batch_data["seg"].float().to(device)
    #         print(labels.shape)

    #         print(torch.max(inputs))
    #         print(torch.min(inputs))

    #         print(torch.max(labels))
    #         print(torch.min(labels))

            optimizer.zero_grad()
            outputs = model(inputs)
    #         print(outputs.shape)
            out_softmax = nn.Softmax(dim=1)(outputs)
    #         print(out_softmax.shape)

    #         print(torch.max(outputs))
    #         print(torch.min(outputs))
    #         print(outputs.shape)
    #         print(labels.shape)

            train_loss = loss_function(out_softmax, labels)
            # train_loss.requires_grad = True
            train_loss.backward()
    #         print(outputs)
    #         print('------------------')
    #         print(labels)
            score = calculate_metrics(out_softmax, labels)
            metrics_score = list(map(add, metrics_score, score))

            optimizer.step()
    #         print(train_loss.item())
    #         print(type(epoch_train_loss))
            epoch_train_loss += train_loss.item()

            epoch_train_loss = epoch_train_loss/len(train_dataloader)
            epoch_train_jaccard = metrics_score[0]/len(train_dataloader)
            epoch_train_acc = metrics_score[1]/len(train_dataloader)
            
            wandb.log({
            "train_loss":epoch_train_loss,
            "train_jaccard":epoch_train_jaccard,
            "train_acc":epoch_train_acc,
            }, step=epoch)


        overall_train_loss_per_epoch.append(train_loss.item())
        overall_train_jaccard_per_epoch.append(epoch_train_jaccard)
        overall_train_acc_per_epoch.append(epoch_train_acc)


        model.eval()
        metrics_score = [0.0, 0.0]
        with torch.no_grad():
            for batch_data in tqdm(val_dataloader, desc=f'Test Epoch {epoch}/{num_epochs}', unit='epoch'):
                inputs = batch_data["image"].float().to(device)
                labels = batch_data["seg"].float().to(device)
                outputs = model(inputs)
    #             print(outputs.shape)
                out_softmax = nn.Softmax(dim=1)(outputs)
    #             print(out_softmax.shape)

                val_loss = loss_function(outputs, labels)

                score = calculate_metrics(outputs, labels)
                metrics_score = list(map(add, metrics_score, score))

                optimizer.step()
    #             print(test_loss.item())
                epoch_val_loss += val_loss.item()
                epoch_val_loss = epoch_train_loss/len(val_dataloader)
                epoch_val_jaccard = metrics_score[0]/len(val_dataloader)
                epoch_val_acc = metrics_score[1]/len(val_dataloader)
                
                wandb.log({
                    "val_loss":epoch_val_loss,
                    "val_jaccard":epoch_val_jaccard,
                    "val_acc":epoch_val_acc,
                    }, step=epoch)


    #             print(epoch_test_loss)

            overall_val_loss_per_epoch.append(val_loss.item()) 
            overall_val_jaccard_per_epoch.append(epoch_val_jaccard)
            overall_val_acc_per_epoch.append(epoch_val_acc)
            
        

        print(f'Epoch [{epoch + 1}/{num_epochs}], '
            f'Train Loss: {train_loss.item():.4f}, '
            f'Train Jaccard: {epoch_train_jaccard.item():.4f}, '
            f'Train Accuracy: {epoch_train_acc.item():.4f}, '
            f'Val Loss: {val_loss.item():.4f}, '
            f'Val Jaccard: {epoch_val_jaccard.item():.4f}, '
            f'Val Accuracy: {epoch_val_acc.item():.4f}, ')


if __name__=='__main__':
    wandb.login()
    wandb.init(project="3D-Liver-Segmentation",
           config={"learning_rate":1e-5, 
                   "epochs":20, 
                   "num_classes":3,
                   "dropout":0.5,
                   "batch_size":1})
    
    train_dataloader, val_dataloader, image_test, segmentation_test = get_dataloaders(wandb.config["batch_size"], wandb.config["num_classes"])
    
    
    model, num_epochs,optimizer, loss, overall_train_loss_per_epoch, overall_train_jaccard_per_epoch, overall_train_acc_per_epoch, overall_test_loss_per_epoch, overall_test_jaccard_per_epoch, overall_test_acc_per_epoch = training_model(train_dataloader,val_dataloader,wandb.config["dropout"],wandb.config["epochs"], wandb.config["learning_rate"]) 

    wandb.watch(model,loss,log="all", log_freq=10)

    columns = ['image', 'segmentation']
    df = pd.DataFrame(columns=columns)

    for index in tqdm(range(len(image_test))):
        df.loc[index, 'image'] = image_test[index]
        df.loc[index, 'segmentation'] = segmentation_test[index]
    df.to_csv('test_dataset.csv')
    




    
