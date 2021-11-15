import os
import math
import argparse
from torch.utils.data import Dataset
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

import pandas as pd
import vit_model_for_intent
from utils import read_split_data, train_one_epoch, evaluate

class MyDataSetNgsim(Dataset):
    """自定义数据集"""
    def __init__(self,  path):
        self.path=path          # csv  path  
        self.data_df=pd.read_csv(self.path)
        self.data_df=self.data_df[["v_Vel","delta_x","delta_y","yaw","label"]]
        print(len(self.data_df))



    def __len__(self):
        return len(self.data_df) // 10

    def __getitem__(self, index) :
        the_df_be_got=self.data_df.iloc[index*10:index*10+10,:]
        label=the_df_be_got["label"].iloc[0]   # label is a number 
        the_df_be_got= the_df_be_got.drop(["label"], axis=1)
        the_array_be_got=the_df_be_got.values
        return the_array_be_got, label


def main():
    epochs=100
    device = torch.device( 'cuda:0' if torch.cuda.is_available() else "cpu")

    if os.path.exists("./weights") is False:
        os.makedirs("./weights")

    tb_writer = SummaryWriter()

    us101_train_dataset=MyDataSetNgsim("/home/l/code_for_vscode/deep-learning-for-image-processing/pytorch_classification/data/ngsim_data_us101/train.csv")
    # 每次取出一个array  和 一个 label
    train_loader= torch.utils.data.DataLoader(us101_train_dataset, batch_size=1,  shuffle=True )

    us101_val_dataset=MyDataSetNgsim("/home/l/code_for_vscode/deep-learning-for-image-processing/pytorch_classification/data/ngsim_data_us101/val.csv")
    # 每次取出一个array  和 一个 label
    val_loader= torch.utils.data.DataLoader(us101_train_dataset, batch_size=1,  shuffle=True )


    model=vit_model_for_intent.create_model().to(device)

    pg = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(pg, lr=0.001, momentum=0.9, weight_decay=5E-5)
    
    lf = lambda x: ((1 + math.cos(x * math.pi / epochs)) / 2) * (1 - 0.01) + 0.01  # cosine
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    for epoch in range(epochs):
        # train
        train_loss, train_acc = train_one_epoch(model=model,
                                                optimizer=optimizer,
                                                data_loader=train_loader,
                                                device=device,
                                                epoch=epoch)

        scheduler.step()

        # validate
        val_loss, val_acc = evaluate(model=model,
                                     data_loader=val_loader,
                                     device=device,
                                     epoch=epoch)

        tags = ["train_loss", "train_acc", "val_loss", "val_acc", "learning_rate"]
        tb_writer.add_scalar(tags[0], train_loss, epoch)
        tb_writer.add_scalar(tags[1], train_acc, epoch)
        tb_writer.add_scalar(tags[2], val_loss, epoch)
        tb_writer.add_scalar(tags[3], val_acc, epoch)
        tb_writer.add_scalar(tags[4], optimizer.param_groups[0]["lr"], epoch)

        torch.save(model.state_dict(), "./weights/model-{}.pth".format(epoch))


if __name__ == '__main__':
    main()
