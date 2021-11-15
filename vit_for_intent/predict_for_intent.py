import os
import json
import torch
from PIL import Image
from torchvision import transforms
from vit_model_for_intent import create_model
import pandas as pd
import numpy as np

list=["left change", "keep lane", "right change"]
def main():              # 现在接收的数据是10*4的dataframe类型的数据
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    f=pd.read_csv("/home/l/code_for_vscode/deep-learning-for-image-processing/pytorch_classification/data/ngsim_data_us101/train.csv")
    f=f[["v_Vel","delta_x","delta_y","yaw"]]
    x=f.iloc[:10,:]
    x_array=x.values
    x_array=x_array[np.newaxis,:]
    x_tensor=torch.from_numpy(x_array)
    x_tensor_float=x_tensor.float()
    print(x_tensor.shape)
    # create model
    model = create_model().to(device)
    # load model weights
    model_weight_path = "./weights/model-99.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(x_tensor_float.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(list[predict_cla],
                                                 predict[predict_cla].numpy())

    print(print_res)

if __name__ == '__main__':
    main()
