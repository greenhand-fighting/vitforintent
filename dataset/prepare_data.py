"""
https://link.zhihu.com/?target=https%3A//pan.baidu.com/s/1PYVio1QxK6wUTlcd0pTxpA
13ux
"""

import  pandas as pd
import json
import math


with open("./data.json") as f:
    conf=json.load(f)
init_df = pd.read_csv(conf['data_source'], usecols=conf['useCols'])  #   文件太大，取自己需要的几列，，'data_source' 表示NGSIM数据的path
us101_df = init_df[init_df['Location'] ==conf["road"] ]   # 只需要us101
us101_df=us101_df.sort_values(by='Global_Time', ascending=True)  # 根据全局时间排序
ft_to_m = 0.3048
for strs in ["Global_X", "Global_Y", "Local_X", "Local_Y", "v_length", "v_Width"]:
     us101_df[strs] =us101_df[strs] * ft_to_m
us101_df["v_Vel"] = us101_df["v_Vel"] * ft_to_m*3.6    # ft/h  ---- m/s
us101_df_sort_by_glotime=us101_df[us101_df["v_Class"] == 2]   # 只要car类别
us101_df_sort_by_glotime_all_car=us101_df_sort_by_glotime.drop(["v_Class","Location","v_length","v_Width","Global_X","Global_Y","Global_Time"], axis=1)  # 删除一些列

left_df=pd.DataFrame()
keep_df=pd.DataFrame()
right_df=pd.DataFrame()

us101_df_sort_by_glotime_all_car=us101_df_sort_by_glotime_all_car.drop_duplicates()  # 把相同的行删除掉
id_number=0     # 重新编注id，从小到大
while len(us101_df_sort_by_glotime_all_car):
    now_vehicle_id=us101_df_sort_by_glotime_all_car["Vehicle_ID"].iloc[0]                   # 根据第一行的id 和帧数，得到一个车的信息。
    now_total_frames=us101_df_sort_by_glotime_all_car["Total_Frames"].iloc[0]              # 根据第一行的id 和帧数，得到一个车的信息。
    one_car_df=us101_df_sort_by_glotime_all_car[(us101_df_sort_by_glotime_all_car["Vehicle_ID"] == now_vehicle_id) &           # 根据第一行的id 和帧数，得到一个车的信息。
    (us101_df_sort_by_glotime_all_car["Total_Frames"] == now_total_frames)]                        # 根据第一行的id 和帧数，得到一个车的信息。
    indexs=one_car_df.index.tolist()   #找出 one car信息的index，
    us101_df_sort_by_glotime_all_car=us101_df_sort_by_glotime_all_car.drop(indexs) # 根据index，将相应的行删除掉
    the_lanes_of_car=one_car_df["Lane_ID"].unique()  # 类型是class
    if len(one_car_df) != now_total_frames:   #判断满足条件的条目是不是来自同一辆车  即:  相同的id  相同的 total frames
        print("more than one car ! ")
    if len(the_lanes_of_car)==1:  # keep     只有一个lane
        if len(one_car_df)>=10:
            one_car_df=one_car_df.iloc[0:10,:]
            one_car_df["Vehicle_ID"]=id_number
            id_number=id_number+1
            keep_df=keep_df.append(one_car_df)
    else:   # change lane
        the_first_lane=the_lanes_of_car[0]
        the_second_lane=the_lanes_of_car[1]
        if the_first_lane<=5 and the_second_lane<=5:           # 大于5 的那些道路是大路旁边的匝道
            the_lanes_list_of_car=one_car_df["Lane_ID"].tolist()
            diffirent_index=0
            for i in range(len(the_lanes_list_of_car)):
                if the_lanes_list_of_car[i] != the_first_lane:
                    diffirent_index=i
                    break
            if the_second_lane>the_first_lane:   # 右转 #这里是保证每一个点，每一个车都有十个信息点
                if diffirent_index>=10:
                    one_car_df=one_car_df.iloc[diffirent_index-10:diffirent_index,:]
                    one_car_df["Vehicle_ID"]=id_number
                    right_df=right_df.append(one_car_df)
                    id_number+=1
            else:  # 左转
                if diffirent_index>=10:   #这里是保证每一个点，每一个车都有十个信息点
                    one_car_df=one_car_df.iloc[diffirent_index-10:diffirent_index,:]
                    one_car_df["Vehicle_ID"]=id_number
                    left_df=left_df.append(one_car_df)
                    id_number+=1
            
        else :
            #这里出现了匝道，就不要导入换道df了
            pass

####-----添加delta x   ，  delta_y  和 yaw 三列!！!！!！!！
keep_df["delta_x"]=0
keep_df["delta_y"]=0
keep_df["yaw"]=0
for i in range(len(keep_df)):
    if i % 10 ==0:
        pass
    else:
        keep_df["delta_x"].iloc[i]=keep_df["Local_X"].iloc[i]-keep_df["Local_X"].iloc[i-1]
        keep_df["delta_y"].iloc[i]=keep_df["Local_Y"].iloc[i]-keep_df["Local_Y"].iloc[i-1]
        if  keep_df["delta_x"].iloc[i]==0:
            keep_df["yaw"].iloc[i]=90
        else :
            tan_yaw=keep_df["delta_y"].iloc[i] / keep_df["delta_x"].iloc[i]
            keep_df["yaw"].iloc[i]=math.atan2(keep_df["delta_y"].iloc[i],keep_df["delta_x"].iloc[i])

left_df["delta_x"]=0
left_df["delta_y"]=0
left_df["yaw"]=0
for i in range(len(left_df)):
    if i % 10 ==0:
        pass
    else:
        left_df["delta_x"].iloc[i]=left_df["Local_X"].iloc[i]-left_df["Local_X"].iloc[i-1]
        left_df["delta_y"].iloc[i]=left_df["Local_Y"].iloc[i]-left_df["Local_Y"].iloc[i-1]
        if  left_df["delta_x"].iloc[i]==0:
            left_df["yaw"].iloc[i]=90
        else :
            tan_yaw=left_df["delta_y"].iloc[i] / left_df["delta_x"].iloc[i]
            left_df["yaw"].iloc[i]=math.atan2(left_df["delta_y"].iloc[i], left_df["delta_x"].iloc[i])

right_df["delta_x"]=0
right_df["delta_y"]=0
right_df["yaw"]=0
for i in range(len(right_df)):
    if i % 10 ==0:
        pass
    else:
        right_df["delta_x"].iloc[i]=right_df["Local_X"].iloc[i]-right_df["Local_X"].iloc[i-1]
        right_df["delta_y"].iloc[i]=right_df["Local_Y"].iloc[i]-right_df["Local_Y"].iloc[i-1]
        if  right_df["delta_x"].iloc[i]==0:
            right_df["yaw"].iloc[i]=90
        else :
            tan_yaw=right_df["delta_y"].iloc[i] / right_df["delta_x"].iloc[i]
            right_df["yaw"].iloc[i]=math.atan2(right_df["delta_y"].iloc[i], right_df["delta_x"].iloc[i])
####-----添加delta x   ，  delta_y  和 yaw 三列!！!！!！!！


keep_df["label"]=1
left_df["label"]=0
right_df["label"]=2    # 添加label



left_df=left_df.drop(["Local_X","Local_Y","Total_Frames"], axis=1)
keep_df=keep_df.drop(["Local_X","Local_Y","Total_Frames"], axis=1)
right_df=right_df.drop(["Local_X","Local_Y","Total_Frames"], axis=1)      # "Local_X","Local_Y","Total_Frames" 好像也没有用了，删除掉

train_df=pd.DataFrame()
train_df=train_df.append(left_df.iloc[:5000,:])
train_df=train_df.append(keep_df.iloc[:40000,:])
train_df=train_df.append(right_df.iloc[:2500,:])
print(len(train_df))
train_df.to_csv("train.csv")

val_df=pd.DataFrame()
val_df=val_df.append(left_df.iloc[5000:,:])
val_df=val_df.append(keep_df.iloc[40000:,:])
val_df=val_df.append(right_df.iloc[2500:,:])
print(len(val_df))
val_df.to_csv("val.csv")