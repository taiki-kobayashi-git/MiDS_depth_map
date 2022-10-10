from turtle import width
import cv2
import torch
import numpy as np
import time
import PIL


def normalize_depth(depth, bits):
    depth_min = depth.min()
    depth_max = depth.max()
    max_val = (2**(8*bits))-1
    if depth_max - depth_min > np.finfo("float").eps:
        out = max_val * (depth - depth_min) / (depth_max - depth_min)
    else:
        out = np.zeros(depth.shape, dtype=depth.type)
    if bits == 1:
        return out.astype("uint8")
    elif bits == 2:
        return out.astype("uint16")

def cut_img(img):
    h, w = img.shape[:2]
    #トリミングする左上の座標
    left, top = 2*w/20, 4*h/20
    #トリミングする右上の座標
    right, bottom = 18*w/20, 16*h/20
    cropped_image = img[400:1500, 800:1700]
    return cropped_image
    


def main():
    n = 2
    #平均計算のためのリスト内の個数
    img_list = [] #narray型の画像格納リスト
    
    ##モデルのインポート
    model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)
    #model_type = "DPT_Hybrid"  # MiDaS v3 - Hybrid
    #model_type = "DPT_Large"  # MiDaS v3 - Large

    midas = torch.hub.load("intel-isl/MiDaS", model_type)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    midas.to(device)

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

    if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
        transform = midas_transforms.dpt_transform
    else:
        transform = midas_transforms.small_transform

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        input_batch = transform(frame).to(device)
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth_frame = prediction.cpu().numpy()
        depth_frame = normalize_depth(depth_frame, bits=1)
        cv2.imshow('Depth Frame', depth_frame)
        
        #print(depth_frame)
        #フィルター
        #ret, img_thresh = cv2.threshold(depth_frame, 180, 10, cv2.THRESH_BINARY)
        #ret, img_thresh3 = cv2.threshold(depth_frame, 200, 255, cv2.THRESH_BINARY)
        #cv2.imshow("sss",img_thresh3)
        #print(img_thresh)
        #img_array = np.array(depth_frame)
        #平均を取る
        if len(img_list) >= n:
            img_list.pop(0)
        img_list.append(depth_frame)
        
        
        img_ave = np.array(sum(img_list)/len(img_list))
        img_ave = np.array(sum(img_list)/len(img_list))
        print(img_ave)
        img_thresh2 = (img_ave >= 5) * 255
        #print(img_thresh2)
        cv2.imshow('average', img_thresh2.astype(np.float32))

        
        

        #time.sleep(0.1)
        
        if cv2.waitKey(1) == 27:
            break
        

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()