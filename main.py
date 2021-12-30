# -*- coding: utf-8 -*-
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import glob2
import argparse
import numpy as np
import copy
import sys
import random
import pandas as pd
import math
from pathlib import Path
import albumentations.pytorch
from pytorchcv.model_provider import get_model as ptcv_get_model

args = {
    "gpu_num": 0,
    "fps": 5,
    "img_row": 1280,
    "yolo_weight": "weights/yolov5/yolov5x_best.pt",
    "human_cls_weight": "",
    "falldown_cls_weight": (
        "weights/falldown_classification/_efficientnetb4b_29.pth"
    ),
    "yolo_conf_thres": 0.01,
    "yolo_max_det": 1000,
    "calibrate_confidence_l1": 1.0,
    "calibrate_confidence_l2": 2.0,
    "yolo_threshold": 0.01,
    "human_threshold": 0.0,
    "falldown_threshold": 0.25,
    "dataset_dir": "./",
    "result": "./result.json",
    "create_main_bbox_threshold1": 0.8,
    "create_main_bbox_threshold2": 0.2,
    "create_main_bbox_length": 10,
    "tracking_iou_threshold": 0.7,
    "tracking_frame_length": 10,
    "tracking_time_threshold": 10,
    "before_falldown_threshold": 0.8,
    "before_falldown_remove_length": 10,
    "concat_iou_threshold": 0.2,
    "concat_intersection_threshold": 0.5,
    "concat_human_threshold": 0.7,
    "concat_falldown_threshold": 0.6,
    "isBrighterOn": True,
    "bri": 100,
    "clip_th": 50,
    "inp_boundary_iou_threshold": 0.5,
    "inp_boundary_intersection_threshold": 0.5,
    "inp_boundary_time_threshold": 12,
}

sys.path.append("." + "/yolov5/")


class EfficientNet_model(nn.Module):
    def __init__(self, net):
        super(EfficientNet_model, self).__init__()
        self.backbone = net.features
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(1792, 1)

    def forward(self, input):
        x = self.backbone(input)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        output = self.out(x)

        return output


# local path for model weights
yolov5_weight = args["yolo_weight"]
human_cls_weight = args["human_cls_weight"]
falldown_cls_weight = args["falldown_cls_weight"]

# model
from yolov5 import models

# yolo image row and col
img_row = args["img_row"]

# resize row and col ratio
img_col = int(img_row * 1080 / 1920)

# Variable as dictionary
file_names = {
    "result": args["result"],
}

# torch initializaton
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)
random.seed(0)
np.random.seed(0)
torch.cuda.empty_cache()

# GPU 할당 변경하기
GPU_NUM = args["gpu_num"]
device = torch.device(
    f"cuda:{GPU_NUM}" if torch.cuda.is_available() else "cpu"
)
torch.cuda.set_device(device)  # change allocation of current GPU

# load yolo model
from yolov5.models.experimental import attempt_load
from yolov5.utils.general import non_max_suppression, scale_coords
from yolov5.utils.augmentations import letterbox

model_y_new = attempt_load(
    yolov5_weight, map_location=device, inplace=True, fuse=True
)
model_y_new.classes = [0]
model_y_new.to(device)
model_y_new.eval()

# +
# human classification model
# net_p = ptcv_get_model('efficientnet_b4b', pretrained=False)
# Net_P = EfficientNet_model(net_p).to(device)
# Net_P.load_state_dict(torch.load(human_cls_weight, map_location=device))
# Net_P.requires_grad_(False)
# Net_P.eval()
# -

# falldown classification model
net_f = ptcv_get_model("efficientnet_b4b", pretrained=False)
Net_F = EfficientNet_model(net_f).to(device)
Net_F.load_state_dict(torch.load(falldown_cls_weight, map_location=device))
Net_F.requires_grad_(False)
Net_F.eval()

classifier_list = {
    # "person_classification" : Net_P,
    "falldown_classification": Net_F
}


def img_estim(image, threshold=50):
    if np.mean(image) > threshold:
        mode = "light"
    elif np.mean(image) > threshold // 2 and np.mean(image) <= threshold:
        mode = "dark"
    else:
        mode = "verydark"
    return mode, np.mean(image)


# +
# image processor
# image bright / contrast / filter application
def sharpening(image):
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(image, -1, kernel)
    return img


def contrast_img(image, clip_th=3.0, tileGridSize=(8, 8)):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_th, tileGridSize=tileGridSize)
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return img


def brighter(image, bri=args["bri"]):
    M = np.ones(image.shape, dtype="uint8") * bri
    img = cv2.add(image, M)
    return img


def brighter_revised(image, mul=2):
    img = cv2.multiply(image, (mul, mul, mul, 0))
    return img


def preprocess_Dark(image, bri=args["bri"], clip_th=args["clip_th"]):
    mode, mean = img_estim(image)
    if mode == "dark":
        # img = contrast_img(image, clip_th=3.0+(clip_th-mean)/1)
        img = brighter_revised(image, mul=2)
    elif mode == "verydark":
        # img = brighter(image, bri=int((bri-mean)))
        # img = contrast_img(img, clip_th=3.0+(clip_th-mean)/1)
        img = brighter_revised(image, mul=3)
    else:
        img = image

    return img, mode


# -

# image crop
def crop(image, label, isBrighterOn=args["isBrighterOn"]):
    images = []
    img = cv2.imread(image, cv2.IMREAD_COLOR)[..., ::-1]
    ys, xs, _ = img.shape

    for i in range(len(label)):
        lbl = label[i]
        y1 = max(0, lbl[1] - 3)  # 3씩 +/- 하는 이유는..?
        x1 = max(0, lbl[0] - 3)
        if x1 > xs:
            x1 = xs - 1
        if y1 > ys:
            y1 = ys - 1
        y2 = min(ys, lbl[3] + 3)
        x2 = min(xs, lbl[2] + 3)
        imgs = img[int(y1) : int(y2) + 1, int(x1) : int(x2) + 1]

        # preprocess dark
        if isBrighterOn:
            imgs, _ = preprocess_Dark(imgs)

        if len(imgs) > 0:
            images.append(imgs)

    return images


class CLSDataset(torch.utils.data.Dataset):
    def __init__(self, img_ids, transform):
        self.img_ids = img_ids
        self.transform = transform

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        image = self.img_ids[idx]
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image


# albumentation
# image modifying, resize -> normalize -> convert to tensor
val_transform = albumentations.Compose(
    [
        albumentations.Resize(224, 224),
        albumentations.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        ),
        albumentations.pytorch.transforms.ToTensorV2(),
    ]
)


def compute_ratio(bbox):
    return (bbox[3] - bbox[1]) / (bbox[2] - bbox[0])


def compute_iou2(box_a, box_b):

    # compute iou with method 2 : just multiply the difference of x and y
    max_x = min(box_a[2], box_b[2])
    max_y = min(box_a[3], box_b[3])
    min_x = max(box_a[0], box_b[0])
    min_y = max(box_a[1], box_b[1])

    intersection = max(max_x - min_x, 0) * max(max_y - min_y, 0)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union


def make_int(bbox):
    if len(bbox) > 0:
        return [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]
    else:
        return []


# +
def yolo_find_allbbox_ctrl_size(model_y, device, img_fld, **kwargs):

    detection = []
    confidence = []

    img_row = kwargs["img_row"]
    conf_thres = kwargs["conf_thres"]
    max_det = kwargs["max_det"]

    for img in img_fld:
        img = cv2.imread(img)
        img0s = img.copy()
        img = letterbox(
            img, img_row, stride=int(model_y.stride.max()), auto=True
        )[0]
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).to(device).float()
        img = img[None]
        img = img / 255.0

        pred = model_y(img, augment=False, visualize=False)
        pred = non_max_suppression(
            pred[0], conf_thres=conf_thres, max_det=max_det
        )

        if device == "cpu":
            result = pred[0].numpy()
        else:
            result = pred[0].detach().cpu().numpy()

        dets = []
        confs = []

        for comp in result:
            bbox = comp[0:4].tolist()
            p = comp[4].tolist()

            dets.append(bbox)
            confs.append(p)

        if len(dets) >= 1:
            dets = torch.Tensor(dets)
            dets = scale_coords(img.shape[2:], dets, img0s.shape).round()
            dets = dets.numpy().tolist()
        else:
            dets = []

        detection.append(dets)
        confidence.append(confs)

    return detection, confidence


def yolo_process(folder, model_y_new, device, **kwargs):

    fps = kwargs["fps"]
    img_row = kwargs["img_row"]
    conf_thres = kwargs["conf_thres"]
    max_det = kwargs["max_det"]

    data = dict()
    model_y_new.to(device)

    for i in range(len(folder)):
        img_fld = []
        fld = sorted(glob2.glob(folder[i] + "/*.jpg"))

        image = cv2.imread(fld[0])
        mode, mean = img_estim(image)

        for j in range(len(fld)):
            if (j + 1) % fps == 0:
                img_fld.append(fld[j])

        fld_name = folder[i]  # .split('/')[-1]
        fld_data1 = yolo_find_allbbox_ctrl_size(
            model_y_new,
            device,
            img_fld,
            img_row=img_row,
            conf_thres=conf_thres,
            max_det=max_det,
        )

        fld_detect = []
        fld_confidence = []
        for ii in range(len(fld_data1[0])):
            Det = []
            Conf = []
            for idx, bbox in enumerate(fld_data1[0][ii]):

                width = abs(bbox[3] - bbox[1])
                height = abs(bbox[2] - bbox[0])

                if min(width, height) > 32:
                    Det.append(bbox)
                    Conf.append(fld_data1[1][ii][idx])

            fld_detect.append(Det)
            fld_confidence.append(Conf)

        """
        fld_detect = []
        for ii in range(len(fld_data1[0])):
            Det = []
            Det.extend(fld_data1[0][ii])
            fld_detect.append(Det)

        fld_confidence = []
        for ii in range(len(fld_data1[1])):
            Conf = []
            Conf.extend(fld_data1[1][ii])
            fld_confidence.append(Conf)
        """

        data[fld_name] = dict()

        data[fld_name]["detect"] = fld_detect
        data[fld_name]["confidence"] = fld_confidence
        data[fld_name]["mode"] = fps
        data[fld_name]["M"] = mode
        data[fld_name]["mean"] = mean

        ans = {}
        for j in range(len(fld_detect)):
            ans[fps * (j + 1)] = fld_detect[j]
        data[fld_name]["answer"] = ans

    model_y_new.cpu()
    torch.cuda.empty_cache()

    return data


def person_cls_new(detect, fld, fps, mode, classifier, bri=None):

    human_confidence = []

    person_set = []
    crop_imgs_set = []

    for kk in range(len(fld)):
        i = fps * (kk + 1)
        a = detect[i]

        crop_imgs_per_frame = crop(fld[kk], a)
        crop_imgs_set.extend(crop_imgs_per_frame)

        if len(crop_imgs_per_frame) == 0:
            person_set.extend([-1])
        else:
            person_set.extend([i for i in range(len(crop_imgs_per_frame))])

    dset = CLSDataset(crop_imgs_set, val_transform)
    dloader = torch.utils.data.DataLoader(
        dset, shuffle=False, batch_size=32, num_workers=4
    )

    score_list = []
    for j, d in enumerate(dloader):
        with torch.no_grad():
            score = torch.sigmoid(classifier(d.cuda())).cpu().numpy()
            score_list.extend(score.tolist())

    score_list_per_frame = []

    # person_set : 0,1,2, 0,1, 0, 0, 0,1,2, .... 0 (or 0,1,2)

    idx_score = 0
    for idx, obj in enumerate(person_set):

        if idx != 0 and obj != 0:

            if obj == -1:

                if len(score_list_per_frame) != 0:
                    human_confidence.append(score_list_per_frame)
                    score_list_per_frame = []

                human_confidence.append([])
                continue

            score_list_per_frame.extend(score_list[idx_score])

            if idx == len(person_set) - 1:
                human_confidence.append(score_list_per_frame)

            idx_score += 1

        elif idx != 0 and obj == 0:

            if len(score_list_per_frame) != 0:
                human_confidence.append(score_list_per_frame)

            score_list_per_frame = []
            score_list_per_frame.extend(score_list[idx_score])

            if idx == len(person_set) - 1:
                human_confidence.append(score_list_per_frame)

            idx_score += 1

        elif idx == 0 and obj != 0:
            human_confidence.append([])

        elif idx == 0 and obj == 0:
            score_list_per_frame = []
            score_list_per_frame.extend(score_list[idx_score])

            idx_score += 1

    return human_confidence


def falldown_cls_new(detect, fld, fps, mode, classifier):

    falldown_confidence = []
    person_set = []
    crop_imgs_set = []

    for kk in range(len(fld)):
        i = fps * (kk + 1)
        a = detect[i]

        crop_imgs_per_frame = crop(fld[kk], a)
        crop_imgs_set.extend(crop_imgs_per_frame)

        if len(crop_imgs_per_frame) == 0:
            person_set.extend([-1])
        else:
            person_set.extend([i for i in range(len(crop_imgs_per_frame))])

    dset = CLSDataset(crop_imgs_set, val_transform)
    dloader = torch.utils.data.DataLoader(
        dset, shuffle=False, batch_size=32, num_workers=4
    )

    score_list = []
    for j, d in enumerate(dloader):
        with torch.no_grad():
            score = torch.sigmoid(classifier(d.cuda())).cpu().numpy()
            score_list.extend(score.tolist())

    score_list_per_frame = []
    falldown_confidence = []

    idx_score = 0

    for idx, obj in enumerate(person_set):

        if idx != 0 and obj != 0:

            if obj == -1:

                if len(score_list_per_frame) != 0:
                    falldown_confidence.append(score_list_per_frame)
                    score_list_per_frame = []

                falldown_confidence.append([])
                continue

            score_list_per_frame.extend(score_list[idx_score])

            if idx == len(person_set) - 1:
                falldown_confidence.append(score_list_per_frame)

            idx_score += 1

        elif idx != 0 and obj == 0:

            if len(score_list_per_frame) != 0:
                falldown_confidence.append(score_list_per_frame)

            score_list_per_frame = []
            score_list_per_frame.extend(score_list[idx_score])

            if idx == len(person_set) - 1:
                falldown_confidence.append(score_list_per_frame)

            idx_score += 1

        elif idx == 0 and obj != 0:
            falldown_confidence.append([])

        elif idx == 0 and obj == 0:
            score_list_per_frame = []
            score_list_per_frame.extend(score_list[idx_score])

            idx_score += 1

    return falldown_confidence


def human_classification_process(data, classifier):
    class_data = deepcopy(data)
    # classifier.cuda()

    for key in class_data.keys():

        detect = class_data[key]["answer"]
        mode = class_data[key]["M"]

        new_confidence = []
        key_list = list(detect.keys())

        for key_ in key_list:
            new_conf = [1 for idx in range(len(detect[key_]))]
            new_confidence.append(new_conf)

        class_data[key]["human_confidence"] = new_confidence

    return class_data


def falldown_classification_process(data, classifier):
    class_data = deepcopy(data)
    classifier.cuda()

    for key in class_data.keys():

        img_fld = []
        fld = sorted(glob2.glob(key + "/*"))
        fps = class_data[key]["mode"]

        for j in range(len(fld)):
            if (j + 1) % fps == 0:
                img_fld.append(fld[j])

        detect = class_data[key]["answer"]
        mode = class_data[key]["M"]

        new_confidence = falldown_cls_new(
            detect, img_fld, fps, mode, classifier
        )
        class_data[key]["falldown_confidence"] = new_confidence

    classifier.cpu()
    torch.cuda.empty_cache()

    return class_data


def build_own_structure(data):
    data_after = {}

    for key in data:
        data_after[key] = {}
        frame_num_list = list(data[key]["answer"].keys())
        data_answer = data[key]["answer"]
        yolo_confidence = data[key]["confidence"]
        human_confidence = data[key]["human_confidence"]
        falldown_confidence = data[key]["falldown_confidence"]
        mode = data[key]["mode"]
        M = data[key]["M"]

        frame_list = []

        for idx, frame_num in enumerate(frame_num_list):

            try:
                bboxs_list_per_frame = data_answer[
                    frame_num
                ]  # [[x1,y1,x2,y2], ....]
                yolo_conf_list_per_frame = yolo_confidence[idx]
                human_conf_list_per_frame = human_confidence[idx]
                falldown_conf_list_per_frame = falldown_confidence[idx]

                data_after[key][frame_num] = {}

                data_after[key][frame_num]["bbox"] = bboxs_list_per_frame
                data_after[key][frame_num][
                    "yolo_confidence"
                ] = yolo_conf_list_per_frame
                data_after[key][frame_num][
                    "human_confidence"
                ] = human_conf_list_per_frame
                data_after[key][frame_num][
                    "falldown_confidence"
                ] = falldown_conf_list_per_frame

                frame_list.append(frame_num)

            except Exception:
                continue

        frame_list = sorted(frame_list)

        data_after[key]["mode"] = mode
        data_after[key]["M"] = M
        data_after[key]["frame_list"] = frame_list

    return data_after


def calibrate_confidence(
    confidence, bbox, activation="sigmoid", l1=1.0, l2=1.0
):

    wh_ratio = compute_ratio(bbox)
    wh_ratio_reverse = 1 / (wh_ratio + 0.001) + 0.1
    a_w = 0

    if activation == "sigmoid":
        a_w = (
            np.exp(wh_ratio_reverse - 1) / (1 + np.exp(wh_ratio_reverse - 1))
            - 0.5
        )

    elif activation == "tanh":
        a_w = (np.exp(wh_ratio_reverse - 1) - 1) / (
            1 + np.exp(wh_ratio_reverse - 1)
        )
    else:
        a_w = np.exp(wh_ratio_reverse) / (1 + np.exp(wh_ratio_reverse))

    new_confidence = l1 * confidence + confidence * a_w * l2

    return new_confidence


def data_handling(data, **kwargs):

    # threshold
    y_threshold = kwargs["yolo_threshold"]
    h_threshold = kwargs["human_threshold"]
    f_threshold = kwargs["falldown_threshold"]

    data_filtered = {}

    # filtering with human_classification

    for key in data.keys():  # key : video_dir

        data_filtered[key] = {}

        data_per_video = data[key]
        frame_list = data_per_video["frame_list"]

        for frame in frame_list:

            data_filtered[key][frame] = {}

            bboxs = data_per_video[frame]["bbox"]
            y_confs = data_per_video[frame]["yolo_confidence"]
            h_confs = data_per_video[frame]["human_confidence"]
            f_confs = data_per_video[frame]["falldown_confidence"]

            object_idx_list = list(range(0, len(bboxs)))  # 0,1,2,...bbox 갯수
            object_idx_list_filtered = []

            for idx in object_idx_list:

                # bbox 크기 조건
                bbox = bboxs[idx]
                width = abs(bbox[3] - bbox[1])
                height = abs(bbox[2] - bbox[0])

                if min(width, height) <= 32:
                    continue

                # filtering 조건
                if h_confs[idx] >= h_threshold and y_confs[idx] >= y_threshold:
                    if f_confs[idx] >= f_threshold:
                        # case 1 : y_conf, h_conf, f_conf 모두 조건 만족
                        object_idx_list_filtered.append(idx)

                    else:
                        # case 2 : f_conf 기준 미달시, bbox를 이용한 보정값 활용
                        f_confs_calibrated = calibrate_confidence(
                            f_confs[idx], bbox, activation="sigmoid", l2=0.8
                        )

                        if f_confs_calibrated >= f_threshold:
                            object_idx_list_filtered.append(idx)
                else:
                    # human confidence 값과 falldown 값이 높은 경우
                    if (
                        h_confs[idx] >= h_threshold
                        and f_confs[idx] >= f_threshold
                    ):

                        # y_confs를 보정
                        if calibrate_confidence(
                            y_confs[idx], bbox, "sigmoid", l1=1, l2=2
                        ):
                            object_idx_list_filtered.append(idx)
                        else:
                            # h_conf 와 f_conf가 굉장히 높게 나올 경우 일단 추가
                            if h_confs[idx] >= 0.85 and f_confs[idx] >= 0.85:
                                object_idx_list_filtered.append(idx)
                            else:
                                pass

                    elif (
                        f_confs[idx] >= f_threshold
                        and h_confs[idx] < h_threshold
                    ):

                        h_conf_cal = calibrate_confidence(
                            h_confs[idx], bbox, "sigmoid", l1=1, l2=2
                        )
                        y_conf_cal = calibrate_confidence(
                            y_confs[idx], bbox, "sigmoid", l1=1, l2=2
                        )

                        if (
                            y_conf_cal >= y_threshold
                            and h_conf_cal >= h_threshold
                        ):
                            object_idx_list_filtered.append(idx)

                        else:
                            pass

            if len(object_idx_list_filtered) == 0:
                data_filtered[key][frame]["bbox"] = []
                data_filtered[key][frame]["yolo_confidence"] = []
                data_filtered[key][frame]["human_confidence"] = []
                data_filtered[key][frame]["falldown_confidence"] = []

            elif len(object_idx_list_filtered) >= 1:

                bboxs_filtered = []
                y_confs_filtered = []
                h_confs_filtered = []
                f_confs_filtered = []

                for idx in object_idx_list_filtered:
                    bboxs_filtered.append(bboxs[idx])
                    y_confs_filtered.append(y_confs[idx])
                    h_confs_filtered.append(h_confs[idx])
                    f_confs_filtered.append(f_confs[idx])

                data_filtered[key][frame]["bbox"] = bboxs_filtered
                data_filtered[key][frame]["yolo_confidence"] = y_confs_filtered
                data_filtered[key][frame][
                    "human_confidence"
                ] = h_confs_filtered
                data_filtered[key][frame][
                    "falldown_confidence"
                ] = f_confs_filtered

        data_filtered[key]["frame_list"] = frame_list
        data_filtered[key]["mode"] = data[key]["mode"]
        data_filtered[key]["M"] = data[key]["M"]

    return data_filtered


def create_main_bbox(
    data, img_dir, threshold=0.9, length=15, iou_threshold=0.75
):

    frames = data[img_dir]["frame_list"]

    prepare_list = []
    y_confs_list = []
    h_confs_list = []
    f_confs_list = []
    detect = []

    for frame in frames:
        prepare_list.extend(data[img_dir][frame]["bbox"])
        y_confs_list.extend(data[img_dir][frame]["yolo_confidence"])
        h_confs_list.extend(data[img_dir][frame]["human_confidence"])
        f_confs_list.extend(data[img_dir][frame]["falldown_confidence"])

    iou_list = []
    for i in range(len(prepare_list)):
        s = 0
        for j in range(len(prepare_list)):
            iou = compute_iou2(prepare_list[i], prepare_list[j])
            if iou >= threshold:
                s += 1
        iou_list.append(s)

    conf_dict = {}
    for i in range(len(iou_list)):
        conf_dict[i] = (f_confs_list[i]) if iou_list[i] > length else 0

    sorted_conf = sorted(conf_dict.items(), key=lambda x: x[1], reverse=True)
    indexs = [sorted_conf[i][0] for i in range(len(sorted_conf))]

    keep = [True] * len(indexs)

    for i in range(len(indexs) - 1):
        keep_num = 1
        if keep[i] is True:
            for j in range(i + 1, len(indexs)):
                if (
                    compute_iou2(
                        prepare_list[indexs[i]], prepare_list[indexs[j]]
                    )
                    > iou_threshold
                ):
                    keep[j] = False
                    keep_num += 1

    sorted_conf_keep = np.array(sorted_conf)[keep]
    len_keep = len(sorted_conf_keep)

    main_bbox_index = [int(sorted_conf_keep[i][0]) for i in range(len_keep)]
    main_bbox = [
        prepare_list[int(sorted_conf_keep[i][0])] for i in range(len_keep)
    ]
    main_bbox_conf = [sorted_conf_keep[i][1] for i in range(len_keep)]

    return main_bbox, main_bbox_index, main_bbox_conf


def create_main_bbox_process(data, **kwargs):

    create_main_bbox_thres1 = kwargs["create_main_bbox_thres1"]
    create_main_bbox_thres2 = kwargs["create_main_bbox_thres2"]
    create_main_bbox_length = kwargs["create_main_bbox_length"]

    data_after = deepcopy(data)

    for key in data_after:
        data_after[key]["main_frames"] = {}

        main_bbox, main_bbox_idx, main_bbox_conf = create_main_bbox(
            data_after,
            key,
            threshold=create_main_bbox_thres1,
            length=create_main_bbox_length,
            iou_threshold=create_main_bbox_thres2,
        )

        if len(main_bbox) == 0:
            main_bbox, main_bbox_idx, main_bbox_conf = create_main_bbox(
                data_after,
                key,
                threshold=create_main_bbox_thres1 * 0.5,
                length=int(create_main_bbox_length * 0.5),
                iou_threshold=create_main_bbox_thres2 * 1.5,
            )

        data_after[key]["main_bbox"] = main_bbox
        data_after[key]["main_bbox_idx"] = main_bbox_idx
        data_after[key]["main_bbox_conf"] = main_bbox_conf

    return data_after


def compute_intersection_ratio(box_a, box_b):
    max_x = min(box_a[2], box_b[2])
    max_y = min(box_a[3], box_b[3])
    min_x = max(box_a[0], box_b[0])
    min_y = max(box_a[1], box_b[1])

    intersection = max(max_x - min_x, 0) * max(max_y - min_y, 0)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    return intersection / min(area_a, area_b)


def object_tracking(data, iou_threshold=0.5, frame_length_threshold=20, dt=10):

    for key in data.keys():
        frame_list = data[key]["frame_list"]
        main_bboxs = data[key]["main_bbox"]
        data[key]["object"] = {}

        # main_bboxs 가운데 구별 가능한 객체 선별
        object_list_bboxs = []

        for idx, main_bbox in enumerate(main_bboxs):

            is_conjugated = False

            for idx_, main_bbox_ in enumerate(main_bboxs):

                if idx == idx_ or idx > idx_:
                    continue

                iou = compute_iou2(main_bbox, main_bbox_)

                if iou > iou_threshold:
                    is_conjugated = True
                    break
                else:
                    continue

            if is_conjugated:
                continue
            elif not is_conjugated:
                object_list_bboxs.append(main_bbox)

        # object만 따로 식별한 bboxs

        if len(object_list_bboxs) == 0:
            data[key]["object"][0] = {}

            data[key]["object"][0]["object_frame_list"] = []
            data[key]["object"][0]["object_bbox"] = []

            data[key]["object"][0]["object_yolo_confidence"] = []
            data[key]["object"][0]["object_human_confidence"] = []
            data[key]["object"][0]["object_falldown_confidence"] = []

            continue

        else:
            object_idx_list = [idx for idx in range(0, len(object_list_bboxs))]

            for idx in object_idx_list:
                data[key]["object"][idx] = {}
                object_bbox = object_list_bboxs[idx]
                object_frame_list = []
                object_detected_bbox_list = []

                object_detected_yolo_list = []
                object_detected_human_list = []
                object_detected_falldown_list = []

                for frame in frame_list:

                    bboxs = data[key][frame]["bbox"]

                    y_confs = data[key][frame]["yolo_confidence"]
                    h_confs = data[key][frame]["human_confidence"]
                    f_confs = data[key][frame]["falldown_confidence"]

                    isDetected = False
                    isSemiDetected = False
                    idx_detected = 0

                    if len(bboxs) == 0:
                        continue

                    for idx_, bbox in enumerate(bboxs):
                        iou = compute_iou2(bbox, object_bbox)
                        intersection = compute_intersection_ratio(
                            bbox, object_bbox
                        )

                        condition1 = iou >= iou_threshold
                        condition2 = intersection >= min(iou_threshold, 0.8)

                        if condition1:
                            isDetected = True
                            idx_detected = idx_
                            break
                        elif condition2:
                            idx_detected = idx_
                            isSemiDetected = True
                            continue
                        else:
                            continue

                    if isDetected:
                        object_frame_list.append(frame)
                        object_detected_bbox_list.append(bboxs[idx_detected])

                        object_detected_yolo_list.append(y_confs[idx_detected])
                        object_detected_human_list.append(
                            h_confs[idx_detected]
                        )
                        object_detected_falldown_list.append(
                            f_confs[idx_detected]
                        )

                    else:
                        if isSemiDetected:
                            object_frame_list.append(frame)
                            object_detected_bbox_list.append(
                                bboxs[idx_detected]
                            )
                            object_detected_yolo_list.append(
                                y_confs[idx_detected]
                            )
                            object_detected_human_list.append(
                                h_confs[idx_detected]
                            )
                            object_detected_falldown_list.append(
                                f_confs[idx_detected]
                            )
                        else:
                            continue

                # 순서를 바꿔봄
                data[key]["object"][idx][
                    "object_frame_list"
                ] = object_frame_list
                data[key]["object"][idx][
                    "object_bbox"
                ] = object_detected_bbox_list

                data[key]["object"][idx][
                    "object_yolo_confidence"
                ] = object_detected_yolo_list
                data[key]["object"][idx][
                    "object_human_confidence"
                ] = object_detected_human_list
                data[key]["object"][idx][
                    "object_falldown_confidence"
                ] = object_detected_falldown_list

            for idx in object_idx_list:

                object_frame_list = data[key]["object"][idx][
                    "object_frame_list"
                ]

                if len(object_frame_list) > 2 and isinstance(
                    object_frame_list, list
                ):
                    dt_measured = (
                        object_frame_list[-1] - object_frame_list[0]
                    ) // 15
                    condition1 = (
                        len(object_frame_list) >= frame_length_threshold
                    )
                    condition2 = dt_measured >= dt

                else:
                    dt_measured = 0
                    condition1 = False
                    condition2 = False

                if condition1 and condition2:
                    pass
                else:
                    data[key]["object"][idx]["object_frame_list"] = []
                    data[key]["object"][idx]["object_bbox"] = []

                    data[key]["object"][idx]["object_yolo_confidence"] = []
                    data[key]["object"][idx]["object_human_confidence"] = []
                    data[key]["object"][idx]["object_falldown_confidence"] = []
    return data


# -


def interpolate_object_tracking_revised(
    data,
    inp_boundary_iou_threshold=0.3,
    inp_boundary_intersection_threshold=0.5,
    inp_boundary_time_threshold=12,
):

    data_object_tracking = deepcopy(data)

    for key in data_object_tracking.keys():

        fps = data_object_tracking[key]["mode"]
        original_frame_list = data_object_tracking[key]["frame_list"]

        for obj in data_object_tracking[key]["object"].keys():
            frame_list = data_object_tracking[key]["object"][obj][
                "object_frame_list"
            ]

            if len(frame_list) == 0:
                continue
            elif len(frame_list) == 1:
                continue

            # len(frame_list) > 1로 유효한 경우 interpolate

            frame_min = min(frame_list)
            frame_max = max(frame_list)

            if abs(frame_max - frame_min) < 2 * fps:
                continue

            frame_list_complete = list(range(frame_min, frame_max + 1, fps))

            obj_bbox_list_new = []
            obj_y_conf_list_new = []
            obj_h_conf_list_new = []
            obj_f_conf_list_new = []

            for idx in range(0, len(frame_list)):

                frame_i = frame_list[idx]

                if idx == len(frame_list) - 1:
                    frame_f = frame_i
                else:
                    frame_f = frame_list[idx + 1]

                # idx_i = int(frame_i / fps) - 1
                # idx_f = int(frame_f / fps) - 1

                obj_bbox_i = data_object_tracking[key]["object"][obj][
                    "object_bbox"
                ][idx]

                obj_y_conf_i = data_object_tracking[key]["object"][obj][
                    "object_yolo_confidence"
                ][idx]
                obj_h_conf_i = data_object_tracking[key]["object"][obj][
                    "object_human_confidence"
                ][idx]
                obj_f_conf_i = data_object_tracking[key]["object"][obj][
                    "object_falldown_confidence"
                ][idx]

                if frame_f - frame_i == fps:

                    obj_bbox_list_new.append(obj_bbox_i)
                    obj_y_conf_list_new.append(obj_y_conf_i)
                    obj_h_conf_list_new.append(obj_h_conf_i)
                    obj_f_conf_list_new.append(obj_f_conf_i)

                elif frame_f - frame_i > fps:
                    for _ in range(frame_i, frame_f, fps):
                        obj_bbox_list_new.append(obj_bbox_i)
                        obj_y_conf_list_new.append(obj_y_conf_i)
                        obj_h_conf_list_new.append(obj_h_conf_i)
                        obj_f_conf_list_new.append(obj_f_conf_i)

                elif frame_f == frame_i:
                    obj_bbox_list_new.append(obj_bbox_i)
                    obj_y_conf_list_new.append(obj_y_conf_i)
                    obj_h_conf_list_new.append(obj_h_conf_i)
                    obj_f_conf_list_new.append(obj_f_conf_i)

            # 중간에 연결되고 끊어진 경우, 그러나 실제 쓰러진 객체로서 식별될 경우
            # 실신된 객체로 판별하고 위치값이 변하지 않는다고 가정, 마지막 프레임까지 bbox를 구성해야 함

            condition1 = frame_list[-1] < original_frame_list[-1]
            condition2 = (
                frame_list[-1] - frame_list[0]
                >= inp_boundary_time_threshold * 15
            )

            if condition1 and condition2:
                for frame in range(
                    frame_list[-1], original_frame_list[-1] + fps, fps
                ):

                    bboxs = data_object_tracking[key][frame]["bbox"]
                    y_confs = data_object_tracking[key][frame][
                        "yolo_confidence"
                    ]
                    h_confs = data_object_tracking[key][frame][
                        "human_confidence"
                    ]
                    f_confs = data_object_tracking[key][frame][
                        "falldown_confidence"
                    ]

                    if len(bboxs) == 0:
                        isSuccess = False
                        frame_max = min(
                            original_frame_list[-1] + fps, frame_list[-1] + fps
                        )

                        for frame_ in range(frame + fps, frame_max, fps):
                            bboxs_ = data_object_tracking[key][frame_]["bbox"]
                            y_confs_ = data_object_tracking[key][frame_][
                                "yolo_confidence"
                            ]
                            h_confs_ = data_object_tracking[key][frame_][
                                "human_confidence"
                            ]
                            f_confs_ = data_object_tracking[key][frame_][
                                "falldown_confidence"
                            ]

                            if len(bboxs_) == 0:
                                continue
                            elif len(bboxs_) >= 1 and not bboxs_[0]:
                                continue

                            idx_list = []
                            for idx_, bbox_ in enumerate(bboxs_):
                                iou_ = compute_iou2(
                                    obj_bbox_list_new[-1], bbox_
                                )
                                intersection_ = compute_intersection_ratio(
                                    obj_bbox_list_new[-1], bbox_
                                )

                                condition1 = iou_ >= inp_boundary_iou_threshold
                                condition2 = (
                                    intersection_
                                    >= inp_boundary_intersection_threshold
                                )
                                condition3 = f_confs[idx] >= 0.8

                                if condition1 and condition2 and condition3:
                                    frame_list_complete.append(frame)
                                    obj_bbox_list_new.append(bbox_)
                                    obj_y_conf_list_new.append(y_confs_[idx_])
                                    obj_h_conf_list_new.append(h_confs_[idx_])
                                    obj_f_conf_list_new.append(f_confs_[idx_])
                                    isSuccess = True
                                    break
                                elif condition1:
                                    idx_list.append(idx)
                                elif condition2:
                                    idx_list.append(idx)
                                else:
                                    pass

                            if isSuccess:
                                break
                            elif len(idx_list) >= 1:
                                f_confs_target = [
                                    f_confs_[idx] for idx in idx_list
                                ]
                                arg_f_max = np.argmax(f_confs_target)

                                if f_confs_target[arg_f_max] >= 0.8:
                                    frame_list_complete.append(frame)
                                    obj_bbox_list_new.append(bboxs_[arg_f_max])
                                    obj_y_conf_list_new.append(
                                        y_confs_[arg_f_max]
                                    )
                                    obj_h_conf_list_new.append(
                                        h_confs_[arg_f_max]
                                    )
                                    obj_f_conf_list_new.append(
                                        f_confs_[arg_f_max]
                                    )
                                    isSuccess = True
                                    break

                        if isSuccess:
                            continue

                        else:
                            break

                    else:

                        idx_list = []
                        isSuccess = False

                        for idx, bbox in enumerate(bboxs):
                            iou = compute_iou2(obj_bbox_list_new[-1], bbox)
                            intersection = compute_intersection_ratio(
                                obj_bbox_list_new[-1], bbox
                            )

                            condition1 = iou >= inp_boundary_iou_threshold
                            condition2 = (
                                intersection
                                >= inp_boundary_intersection_threshold
                            )
                            condition3 = f_confs[idx] >= 0.8

                            if condition1 and condition2 and condition3:
                                frame_list_complete.append(frame)
                                obj_bbox_list_new.append(bboxs[idx])
                                obj_y_conf_list_new.append(y_confs[idx])
                                obj_h_conf_list_new.append(h_confs[idx])
                                obj_f_conf_list_new.append(f_confs[idx])
                                isSuccess = True
                                break
                            elif condition1:
                                idx_list.append(idx)
                            elif condition2:
                                idx_list.append(idx)
                            else:
                                pass

                        if isSuccess:
                            continue
                        elif len(idx_list) >= 1:
                            f_confs_target = [f_confs[idx] for idx in idx_list]
                            arg_f_max = np.argmax(f_confs_target)

                            if f_confs_target[arg_f_max] >= 0.8:
                                frame_list_complete.append(frame)
                                obj_bbox_list_new.append(bboxs[arg_f_max])
                                obj_y_conf_list_new.append(y_confs[arg_f_max])
                                obj_h_conf_list_new.append(h_confs[arg_f_max])
                                obj_f_conf_list_new.append(f_confs[arg_f_max])
                                continue
                            else:
                                break

                        else:
                            break

            else:
                pass

            data_object_tracking[key]["object"][obj][
                "object_frame_list"
            ] = frame_list_complete
            data_object_tracking[key]["object"][obj][
                "object_bbox"
            ] = obj_bbox_list_new
            data_object_tracking[key]["object"][obj][
                "object_yolo_confidence"
            ] = obj_y_conf_list_new
            data_object_tracking[key]["object"][obj][
                "object_human_confidence"
            ] = obj_h_conf_list_new
            data_object_tracking[key]["object"][obj][
                "object_falldown_confidence"
            ] = obj_f_conf_list_new

    return data_object_tracking


# +
def interpolate_object_tracking(
    data,
    inp_boundary_iou_thres=0.5,
    inp_boundary_intersection_thres=0.5,
    inp_boundary_time_thres=12,
):

    data_object_tracking = deepcopy(data)

    for key in data_object_tracking.keys():

        fps = data_object_tracking[key]["mode"]
        original_frame_list = data_object_tracking[key]["frame_list"]

        for obj in data_object_tracking[key]["object"].keys():
            frame_list = data_object_tracking[key]["object"][obj][
                "object_frame_list"
            ]

            if len(frame_list) == 0:
                continue
            elif len(frame_list) == 1:
                continue

            # len(frame_list) > 1로 유효한 경우 interpolate

            frame_min = min(frame_list)
            frame_max = max(frame_list)

            if abs(frame_max - frame_min) < 2 * fps:
                continue

            frame_list_complete = list(range(frame_min, frame_max + 1, fps))

            obj_bbox_list_new = []
            obj_y_conf_list_new = []
            obj_h_conf_list_new = []
            obj_f_conf_list_new = []

            for idx in range(0, len(frame_list)):

                frame_i = frame_list[idx]

                if idx == len(frame_list) - 1:
                    frame_f = frame_i
                else:
                    frame_f = frame_list[idx + 1]

                # idx_i = int(frame_i / fps) - 1
                # idx_f = int(frame_f / fps) - 1

                obj_bbox_i = data_object_tracking[key]["object"][obj][
                    "object_bbox"
                ][idx]

                obj_y_conf_i = data_object_tracking[key]["object"][obj][
                    "object_yolo_confidence"
                ][idx]
                obj_h_conf_i = data_object_tracking[key]["object"][obj][
                    "object_human_confidence"
                ][idx]
                obj_f_conf_i = data_object_tracking[key]["object"][obj][
                    "object_falldown_confidence"
                ][idx]

                if frame_f - frame_i == fps:

                    obj_bbox_list_new.append(obj_bbox_i)
                    obj_y_conf_list_new.append(obj_y_conf_i)
                    obj_h_conf_list_new.append(obj_h_conf_i)
                    obj_f_conf_list_new.append(obj_f_conf_i)

                elif frame_f - frame_i > fps:
                    for _ in range(frame_i, frame_f, fps):
                        obj_bbox_list_new.append(obj_bbox_i)
                        obj_y_conf_list_new.append(obj_y_conf_i)
                        obj_h_conf_list_new.append(obj_h_conf_i)
                        obj_f_conf_list_new.append(obj_f_conf_i)

                elif frame_f == frame_i:
                    obj_bbox_list_new.append(obj_bbox_i)
                    obj_y_conf_list_new.append(obj_y_conf_i)
                    obj_h_conf_list_new.append(obj_h_conf_i)
                    obj_f_conf_list_new.append(obj_f_conf_i)

            condition1 = frame_list[-1] < original_frame_list[-1]
            condition2 = (
                frame_list[-1] - frame_list[0] >= inp_boundary_time_thres * 15
            )

            if condition1 and condition2:
                for frame in range(
                    frame_list[-1], original_frame_list[-1] + fps, fps
                ):
                    frame_list_complete.append(frame)

                    bboxs = data_object_tracking[key][frame]["bbox"]
                    y_confs = data_object_tracking[key][frame][
                        "yolo_confidence"
                    ]
                    h_confs = data_object_tracking[key][frame][
                        "human_confidence"
                    ]
                    f_confs = data_object_tracking[key][frame][
                        "falldown_confidence"
                    ]

                    if len(bboxs) == 0:
                        isSuccess = False

                        if frame + 2 * fps <= original_frame_list[-1]:
                            frame_max = frame + 3 * fps
                        elif frame + 1 * fps <= original_frame_list[-1]:
                            frame_max = frame + 2 * fps
                        else:
                            frame_max = frame + fps  # 사실상 skip

                        for frame_ in range(frame + fps, frame_max, fps):
                            bboxs_ = data_object_tracking[key][frame_]["bbox"]
                            y_confs_ = data_object_tracking[key][frame_][
                                "yolo_confidence"
                            ]
                            h_confs_ = data_object_tracking[key][frame_][
                                "human_confidence"
                            ]
                            f_confs_ = data_object_tracking[key][frame_][
                                "falldown_confidence"
                            ]

                            if len(bboxs_) == 0:
                                continue

                            idx_list = []
                            for idx_, bbox_ in enumerate(bboxs_):
                                iou_ = compute_iou2(
                                    obj_bbox_list_new[-1], bbox_
                                )
                                intersection_ = compute_intersection_ratio(
                                    obj_bbox_list_new[-1], bbox_
                                )

                                condition1 = iou_ >= inp_boundary_iou_thres
                                condition2 = (
                                    intersection_
                                    >= inp_boundary_intersection_thres
                                )

                                if condition1 and condition2:
                                    obj_bbox_list_new.append(bbox_)
                                    obj_y_conf_list_new.append(y_confs_[idx_])
                                    obj_h_conf_list_new.append(h_confs_[idx_])
                                    obj_f_conf_list_new.append(f_confs_[idx_])
                                    isSuccess = True
                                    break
                                elif condition1:
                                    idx_list.append(idx)
                                elif condition2:
                                    idx_list.append(idx)
                                else:
                                    pass

                            if isSuccess:
                                break
                            elif len(idx_list) >= 1:
                                f_confs_target = [
                                    f_confs_[idx] for idx in idx_list
                                ]
                                arg_f_max = np.argmax(f_confs_target)

                                obj_bbox_list_new.append(bboxs_[arg_f_max])
                                obj_y_conf_list_new.append(y_confs_[arg_f_max])
                                obj_h_conf_list_new.append(h_confs_[arg_f_max])
                                obj_f_conf_list_new.append(f_confs_[arg_f_max])
                                isSuccess = True
                                break

                        if isSuccess:
                            continue

                        else:
                            obj_bbox_list_new.append(obj_bbox_list_new[-1])
                            obj_y_conf_list_new.append(obj_y_conf_list_new[-1])
                            obj_h_conf_list_new.append(obj_h_conf_list_new[-1])
                            obj_f_conf_list_new.append(obj_f_conf_list_new[-1])

                    else:

                        idx_list = []
                        isSuccess = False

                        for idx, bbox in enumerate(bboxs):
                            iou = compute_iou2(obj_bbox_list_new[-1], bbox)
                            intersection = compute_intersection_ratio(
                                obj_bbox_list_new[-1], bbox
                            )

                            condition1 = iou >= inp_boundary_iou_thres
                            condition2 = (
                                intersection >= inp_boundary_intersection_thres
                            )

                            if condition1 and condition2:
                                obj_bbox_list_new.append(bboxs[idx])
                                obj_y_conf_list_new.append(y_confs[idx])
                                obj_h_conf_list_new.append(h_confs[idx])
                                obj_f_conf_list_new.append(f_confs[idx])
                                isSuccess = True
                                break
                            elif condition1:
                                idx_list.append(idx)
                            elif condition2:
                                idx_list.append(idx)
                            else:
                                pass

                        if isSuccess:
                            continue
                        elif len(idx_list) >= 1:
                            f_confs_target = [f_confs[idx] for idx in idx_list]
                            arg_f_max = np.argmax(f_confs_target)

                            obj_bbox_list_new.append(bboxs[arg_f_max])
                            obj_y_conf_list_new.append(y_confs[arg_f_max])
                            obj_h_conf_list_new.append(h_confs[arg_f_max])
                            obj_f_conf_list_new.append(f_confs[arg_f_max])

                            continue

                        else:
                            obj_bbox_list_new.append(obj_bbox_list_new[-1])
                            obj_y_conf_list_new.append(obj_y_conf_list_new[-1])
                            obj_h_conf_list_new.append(obj_h_conf_list_new[-1])
                            obj_f_conf_list_new.append(obj_f_conf_list_new[-1])

                            continue

            else:
                pass

            data_object_tracking[key]["object"][obj][
                "object_frame_list"
            ] = frame_list_complete
            data_object_tracking[key]["object"][obj][
                "object_bbox"
            ] = obj_bbox_list_new
            data_object_tracking[key]["object"][obj][
                "object_yolo_confidence"
            ] = obj_y_conf_list_new
            data_object_tracking[key]["object"][obj][
                "object_human_confidence"
            ] = obj_h_conf_list_new
            data_object_tracking[key]["object"][obj][
                "object_falldown_confidence"
            ] = obj_f_conf_list_new

    return data_object_tracking


def handling_boundary_frame(data, **kwargs):

    data_object_tracking = deepcopy(data)

    falldown_threshold = kwargs["before_falldown_threshold"]
    remove_len_threshold = kwargs["before_falldown_remove_length"]

    for key in data_object_tracking.keys():

        fps = data_object_tracking[key]["mode"]

        for obj in data_object_tracking[key]["object"].keys():

            frame_list = list(
                reversed(
                    data_object_tracking[key]["object"][obj][
                        "object_frame_list"
                    ].copy()
                )
            )

            if len(frame_list) == 0:
                continue
            elif len(frame_list) == 1:
                continue

            bboxs_list = list(
                reversed(
                    data_object_tracking[key]["object"][obj][
                        "object_bbox"
                    ].copy()
                )
            )
            y_confs_list = list(
                reversed(
                    data_object_tracking[key]["object"][obj][
                        "object_yolo_confidence"
                    ].copy()
                )
            )
            h_confs_list = list(
                reversed(
                    data_object_tracking[key]["object"][obj][
                        "object_human_confidence"
                    ].copy()
                )
            )
            f_confs_list = list(
                reversed(
                    data_object_tracking[key]["object"][obj][
                        "object_falldown_confidence"
                    ].copy()
                )
            )

            remove_count = 0

            while True:

                if (
                    f_confs_list[-1] < falldown_threshold
                    and remove_count < remove_len_threshold
                ):

                    frame_list.pop()
                    bboxs_list.pop()
                    y_confs_list.pop()
                    h_confs_list.pop()
                    f_confs_list.pop()

                    remove_count += 1

                elif remove_count >= remove_len_threshold:
                    break

                else:
                    break

            data_object_tracking[key]["object"][obj][
                "object_frame_list"
            ] = list(reversed(frame_list))
            data_object_tracking[key]["object"][obj]["object_bbox"] = list(
                reversed(bboxs_list)
            )
            data_object_tracking[key]["object"][obj][
                "object_yolo_confidence"
            ] = list(reversed(y_confs_list))
            data_object_tracking[key]["object"][obj][
                "object_human_confidence"
            ] = list(reversed(h_confs_list))
            data_object_tracking[key]["object"][obj][
                "object_falldown_confidence"
            ] = list(reversed(f_confs_list))

    return data_object_tracking


def convert_original_structure(data):

    data_copy = deepcopy(data)

    for key in data_copy.keys():

        frame_list = data_copy[key]["frame_list"]

        for frame in frame_list:

            data_copy[key][frame] = {}
            data_copy[key][frame]["bbox"] = []
            data_copy[key][frame]["yolo_confidence"] = []
            data_copy[key][frame]["human_confidence"] = []
            data_copy[key][frame]["falldown_confidence"] = []

        # object 별로 bbox 추가

        for obj in data_copy[key]["object"].keys():

            object_bboxs = data_copy[key]["object"][obj]["object_bbox"]
            object_y_confs = data_copy[key]["object"][obj][
                "object_yolo_confidence"
            ]
            object_h_confs = data_copy[key]["object"][obj][
                "object_human_confidence"
            ]
            object_f_confs = data_copy[key]["object"][obj][
                "object_falldown_confidence"
            ]
            object_frame_list = data_copy[key]["object"][obj][
                "object_frame_list"
            ]

            for idx, object_frame in enumerate(object_frame_list):

                if len(object_bboxs) >= 1:
                    try:
                        obj_bbox = object_bboxs[idx]
                        obj_y_conf = object_y_confs[idx]
                        obj_h_conf = object_h_confs[idx]
                        obj_f_conf = object_f_confs[idx]
                    except Exception:
                        obj_bbox = []
                        obj_y_conf = []
                        obj_h_conf = []
                        obj_f_conf = []

                else:
                    obj_bbox = []
                    obj_y_conf = []
                    obj_h_conf = []
                    obj_f_conf = []

                data_copy[key][object_frame]["bbox"].append(obj_bbox)
                data_copy[key][object_frame]["yolo_confidence"].append(
                    obj_y_conf
                )
                data_copy[key][object_frame]["human_confidence"].append(
                    obj_h_conf
                )
                data_copy[key][object_frame]["falldown_confidence"].append(
                    obj_f_conf
                )

    return data_copy


def fill_result_all_frames(data_after):

    data_final = copy.deepcopy(data_after)

    for key in data_final.keys():

        flds = sorted(glob2.glob(key + "/*"))
        frame_list = data_final[key]["frame_list"]
        fps = data_final[key]["mode"]

        data_final[key]["annotations"] = {}

        for idx, fld in enumerate(flds):
            file_name = fld.split("/")[-1]
            data_final[key]["annotations"][file_name] = {}

            # isDetected_left = False
            # isDetected_right = False

            # dummy
            # if (idx + 1) >= fps:

            #     idx_l = int((idx + 1) // fps) * fps
            #     idx_r = int((idx + 1) // fps + 1) * fps

            #     isDetected_left = len(data_final[key][idx_l]["bbox"]) >= 1

            #     if idx_r <= len(flds):
            #         isDetected_right = len(data_final[key][idx_r]["bbox"]) >= 1

            if (idx + 1) % fps == 0:
                data_final[key]["annotations"][file_name]["bbox"] = data_final[
                    key
                ][(idx + 1)]["bbox"]
            else:
                if int((idx + 1) // fps + 1) * fps > len(flds):
                    data_final[key]["annotations"][file_name][
                        "bbox"
                    ] = data_final[key][int((idx + 1) // fps) * fps]["bbox"]
                else:
                    if idx + 1 < fps:
                        if len(data_final[key][fps]["bbox"]) >= 1:
                            data_final[key]["annotations"][file_name][
                                "bbox"
                            ] = data_final[key][fps]["bbox"]
                        else:
                            data_final[key]["annotations"][file_name][
                                "bbox"
                            ] = []
                    else:
                        data_final[key]["annotations"][file_name][
                            "bbox"
                        ] = data_final[key][int((idx + 1) // fps) * fps][
                            "bbox"
                        ]

    return data_final


def build_submission(data_final):
    submission = {}
    submission["annotations"] = []

    for key in data_final.keys():
        for file_name in data_final[key]["annotations"].keys():
            component = {}
            component["file_name"] = file_name
            component["box"] = []

            if len(data_final[key]["annotations"][file_name]["bbox"]) >= 1:
                for bbox in data_final[key]["annotations"][file_name]["bbox"]:
                    if len(bbox) >= 1:
                        pos = {"position": make_int(bbox)}
                        component["box"].append(pos)
            else:
                pass

            submission["annotations"].append(component)

    return submission


def build_submission_2021(data_final):
    submission = {}
    submission["answer"] = {}

    for key in data_final.keys():
        for file_name in data_final[key]["annotations"].keys():
            submission["answer"][file_name] = {}
            submission["answer"][file_name]["box"] = []

            if len(data_final[key]["annotations"][file_name]["bbox"]) >= 1:

                for bbox in data_final[key]["annotations"][file_name]["bbox"]:
                    submission["answer"][file_name]["box"].append(
                        make_int(bbox)
                    )

    return submission


def cal_area(box):
    return abs((box[2] - box[0]) * (box[3] - box[1]))


def cal_intersection(box_a, box_b):

    max_x = min(box_a[2], box_b[2])
    max_y = min(box_a[3], box_b[3])
    min_x = max(box_a[0], box_b[0])
    min_y = max(box_a[1], box_b[1])

    intersection = max(max_x - min_x, 0) * max(max_y - min_y, 0)

    return intersection


def concat_bbox_with_weight(box_a, box_b, weight_a, weight_b):

    new_box = [
        box_a[0] * weight_a + box_b[0] * weight_b,
        box_a[1] * weight_a + box_b[1] * weight_b,
        box_a[2] * weight_a + box_b[2] * weight_b,
        box_a[3] * weight_a + box_b[3] * weight_b,
    ]
    return new_box


def object_concat_per_bboxs(**kwargs):
    bbox1 = kwargs["bbox1"]
    bbox2 = kwargs["bbox2"]

    [x1_1, y1_1, x1_2, y1_2] = bbox1
    [x2_1, y2_1, x2_2, y2_2] = bbox2

    x1_m = 0.5 * (bbox1[0] + bbox1[2])
    y1_m = 0.5 * (bbox1[1] + bbox1[3])

    x2_m = 0.5 * (bbox2[0] + bbox2[2])
    y2_m = 0.5 * (bbox2[1] + bbox2[3])

    condition_1_in_2 = (
        (x1_2 <= x2_2) and (x1_1 >= x2_1) and (y1_2 >= y2_2) and (y1_1 <= y2_1)
    )
    condition_2_in_1 = (
        (x2_2 <= x1_2) and (x2_1 >= x1_1) and (y2_2 >= y1_2) and (y2_1 <= y1_1)
    )

    condition_1_close_2 = (
        (x1_m <= x2_2) and (x1_m >= x2_1) and (y1_m >= y2_2) and (y1_m <= y2_1)
    )
    condition_2_close_1 = (
        (x2_m <= x1_2) and (x2_m >= x1_1) and (y2_m >= y1_2) and (y2_m <= y1_1)
    )

    if condition_1_in_2:

        w1 = math.sqrt(2) / (1 + math.sqrt(2))
        w2 = 1 / (1 + math.sqrt(2))

        new_bbox = concat_bbox_with_weight(bbox1, bbox2, w1, w2)

    elif condition_2_in_1:

        w2 = math.sqrt(2) / (1 + math.sqrt(2))
        w1 = 1 / (1 + math.sqrt(2))

        new_bbox = concat_bbox_with_weight(bbox1, bbox2, w1, w2)

    else:
        if condition_1_close_2 and not condition_2_close_1:
            w1 = math.sqrt(2) / (1 + math.sqrt(2))
            w2 = 1 / (1 + math.sqrt(2))
            new_bbox = concat_bbox_with_weight(bbox1, bbox2, w1, w2)

        elif condition_2_close_1 and not condition_1_close_2:
            w2 = math.sqrt(2) / (1 + math.sqrt(2))
            w1 = 1 / (1 + math.sqrt(2))
            new_bbox = concat_bbox_with_weight(bbox1, bbox2, w1, w2)

        else:
            w1 = 0.5
            w2 = 0.5
            new_bbox = concat_bbox_with_weight(bbox1, bbox2, w1, w2)

    return new_bbox


def object_concat_per_frame(frame=0, **kwargs):

    bboxs = kwargs["bboxs"]
    y_confs = kwargs["yolo_confidence"]
    h_confs = kwargs["human_confidence"]
    f_confs = kwargs["falldown_confidence"]
    iou_threshold = kwargs["concat_iou_threshold"]
    intersection_threshold = kwargs["concat_intersection_threshold"]
    h_threshold = kwargs["concat_human_threshold"]
    f_threshold = kwargs["concat_falldown_threshold"]

    bboxs_new = []
    y_confs_new = []
    h_confs_new = []
    f_confs_new = []

    for idx in range(0, len(bboxs)):

        if True:
            bbox = bboxs[idx]
            max_index = -1
            max_iou = 0
            max_bbox = []

            for idx_ in range(idx + 1, len(bboxs)):

                bbox_ = bboxs[idx_]
                iou = compute_iou2(bbox, bbox_)
                intersection = compute_intersection_ratio(bbox, bbox_)

                condition_iou = iou >= iou_threshold
                condition_intersection = intersection >= intersection_threshold

                area1 = cal_area(bbox)
                area2 = cal_area(bbox_)

                area_ratio = min(area1 / area2, area2 / area1)

                condition_area = area_ratio > 0.25

                condition_h_conf = True
                condition_f_conf = True

                if (
                    (condition_iou or condition_intersection)
                    and max_iou < intersection
                    and condition_h_conf
                    and condition_f_conf
                    and condition_area
                ):

                    max_iou = intersection
                    max_index = idx_
                    max_bbox = bbox_

            if max_index > -1:
                bbox = object_concat_per_bboxs(bbox1=bbox, bbox2=max_bbox)

            bboxs_new.append(bbox)
            y_confs_new.append(y_confs[idx])
            h_confs_new.append(h_confs[idx])
            f_confs_new.append(f_confs[idx])

    return bboxs_new, y_confs_new, h_confs_new, f_confs_new


def object_concat(data, **kwargs):
    concat_iou_threshold = kwargs["concat_iou_threshold"]
    concat_intersection_threshold = kwargs["concat_intersection_threshold"]
    concat_human_threshold = kwargs["concat_human_threshold"]
    concat_falldown_threshold = kwargs["concat_falldown_threshold"]

    data_output = {}

    for key in data.keys():

        data_output[key] = {}
        frame_list = data[key]["frame_list"]

        for frame in frame_list:

            data_output[key][frame] = {}
            bboxs = data[key][frame]["bbox"]
            y_confs = data[key][frame]["yolo_confidence"]
            h_confs = data[key][frame]["human_confidence"]
            f_confs = data[key][frame]["falldown_confidence"]

            if len(bboxs) == 0:
                bboxs_new = []
                y_confs_new = []
                h_confs_new = []
                f_confs_new = []
            else:
                (
                    bboxs_new,
                    y_confs_new,
                    h_confs_new,
                    f_confs_new,
                ) = object_concat_per_frame(
                    bboxs=bboxs,
                    yolo_confidence=y_confs,
                    human_confidence=h_confs,
                    falldown_confidence=f_confs,
                    concat_iou_threshold=concat_iou_threshold,
                    concat_intersection_threshold=(
                        concat_intersection_threshold
                    ),
                    concat_human_threshold=concat_human_threshold,
                    concat_falldown_threshold=concat_falldown_threshold,
                    frame=frame,
                )

            data_output[key][frame]["bbox"] = bboxs_new
            data_output[key][frame]["yolo_confidence"] = y_confs_new
            data_output[key][frame]["human_confidence"] = h_confs_new
            data_output[key][frame]["falldown_confidence"] = f_confs_new

        data_output[key]["frame_list"] = frame_list
        data_output[key]["mode"] = data[key]["mode"]
        data_output[key]["M"] = data[key]["M"]
        data_output[key]["main_bbox"] = data[key]["main_bbox"]

    return data_output


def object_concat_next_step(data, iou_threshold=0, distance_threshold=20):

    data_output = {}

    for key in data.keys():

        data_output[key] = {}
        frame_list = data[key]["frame_list"]

        for frame in frame_list:

            data_output[key][frame] = {}
            bboxs = data[key][frame]["bbox"]
            y_confs = data[key][frame]["yolo_confidence"]
            h_confs = data[key][frame]["human_confidence"]
            f_confs = data[key][frame]["falldown_confidence"]

            bboxs_new = []
            y_confs_new = []
            h_confs_new = []
            f_confs_new = []

            bboxs_remove_list = [0 for idx in range(0, len(bboxs))]

            for idx in range(0, len(bboxs)):

                if bboxs_remove_list[idx] == 0:

                    for idx_ in range(idx + 1, len(bboxs)):

                        if bboxs_remove_list[idx_] == 0:
                            bbox = bboxs[idx]
                            bbox_ = bboxs[idx_]
                            iou = compute_iou2(bbox, bbox_)
                            condition_iou = iou > iou_threshold

                            box1 = bbox
                            box2 = bbox_

                            condition_distance = (
                                (
                                    box1[0] <= box2[0]
                                    and (
                                        box1[2] > box2[0]
                                        or box2[0] - box1[2]
                                        < distance_threshold
                                    )
                                )
                                or (
                                    box2[0] <= box1[0]
                                    and (
                                        box2[2] > box1[0]
                                        or box1[0] - box2[2]
                                        < distance_threshold
                                    )
                                )
                            ) and (
                                (
                                    box1[1] <= box2[1]
                                    and (
                                        box1[3] > box2[1]
                                        or box2[1] - box1[3]
                                        < distance_threshold
                                    )
                                )
                                or (
                                    box2[1] <= box1[1]
                                    and (
                                        box2[3] > box1[1]
                                        or box1[1] - box2[3]
                                        < distance_threshold
                                    )
                                )
                            )

                            if condition_iou or condition_distance:

                                bboxs_remove_list[idx_] = 1

                                x1 = min(bbox[0], bbox_[0])
                                y1 = min(bbox[1], bbox_[1])

                                x2 = max(bbox[2], bbox_[2])
                                y2 = max(bbox[3], bbox_[3])

                                bboxs[idx] = [x1, y1, x2, y2]

            for idx in range(0, len(bboxs)):

                if bboxs_remove_list[idx] == 0:
                    bboxs_new.append(bboxs[idx])
                    y_confs_new.append(y_confs[idx])
                    h_confs_new.append(h_confs[idx])
                    f_confs_new.append(f_confs[idx])

            data_output[key][frame]["bbox"] = bboxs_new
            data_output[key][frame]["yolo_confidence"] = y_confs_new
            data_output[key][frame]["human_confidence"] = h_confs_new
            data_output[key][frame]["falldown_confidence"] = f_confs_new

        data_output[key]["frame_list"] = frame_list
        data_output[key]["mode"] = data[key]["mode"]
        data_output[key]["M"] = data[key]["M"]
        data_output[key]["main_bbox"] = data[key]["main_bbox"]

    return data_output


def remove_conjugated_object(data, **kwargs):

    iou_threshold = kwargs["iou_threshold"]
    intersection_threshold = kwargs["intersection_threshold"]

    data_output = {}

    for key in data.keys():

        data_output[key] = {}
        frame_list = data[key]["frame_list"]

        for frame in frame_list:

            data_output[key][frame] = {}
            bboxs = data[key][frame]["bbox"]
            y_confs = data[key][frame]["yolo_confidence"]
            h_confs = data[key][frame]["human_confidence"]
            f_confs = data[key][frame]["falldown_confidence"]

            bboxs_new = []
            y_confs_new = []
            h_confs_new = []
            f_confs_new = []

            for idx in range(0, len(bboxs)):

                bbox = bboxs[idx]
                area = cal_area(bbox)

                isRemove = False

                for idx_ in range(0, len(bboxs)):

                    if idx == idx_:
                        continue

                    bbox_ = bboxs[idx_]
                    area_ = cal_area(bbox_)

                    iou = compute_iou2(bbox, bbox_)
                    intersection = compute_intersection_ratio(bbox, bbox_)

                    condition_iou = iou >= iou_threshold
                    condition_intersection = (
                        intersection >= intersection_threshold
                    )

                    if (
                        condition_iou or condition_intersection
                    ) and area + 0.0001 < area_:
                        isRemove = True
                        break

                if not isRemove:
                    bboxs_new.append(bbox)

            data_output[key][frame]["bbox"] = bboxs_new
            data_output[key][frame]["yolo_confidence"] = y_confs_new
            data_output[key][frame]["human_confidence"] = h_confs_new
            data_output[key][frame]["falldown_confidence"] = f_confs_new

        data_output[key]["frame_list"] = frame_list
        data_output[key]["mode"] = data[key]["mode"]
        data_output[key]["M"] = data[key]["M"]
        data_output[key]["main_bbox"] = data[key]["main_bbox"]

    return data_output


def remove_duplicated(data, iou_threshold=0.99):

    data_output = {}

    for key in data.keys():

        data_output[key] = {}
        frame_list = data[key]["frame_list"]

        for frame in frame_list:

            data_output[key][frame] = {}
            bboxs = data[key][frame]["bbox"]
            bboxs_new = []

            for idx in range(0, len(bboxs)):

                bbox = bboxs[idx]
                area = cal_area(bbox)

                isRemove = False

                for idx_ in range(idx + 1, len(bboxs)):

                    bbox_ = bboxs[idx_]

                    iou = compute_iou2(bbox, bbox_)
                    condition_iou = iou >= iou_threshold

                    if condition_iou:
                        isRemove = True
                        break

                if not isRemove:
                    bboxs_new.append(bbox)
            data_output[key][frame]["bbox"] = bboxs_new

        data_output[key]["frame_list"] = frame_list
        data_output[key]["mode"] = data[key]["mode"]
        data_output[key]["M"] = data[key]["M"]
        data_output[key]["main_bbox"] = data[key]["main_bbox"]

    return data_output


from utils.datasets import create_dataloader, LoadImages
from pathlib import Path
import cv2
from copy import deepcopy

if __name__ == "__main__":

    fps = args["fps"]
    # argument : yolo detection
    conf_thres = args["yolo_conf_thres"]
    max_det = args["yolo_max_det"]

    # argument : data handling
    yolo_threshold = args["yolo_threshold"]
    human_threshold = args["human_threshold"]
    falldown_threshold = args["falldown_threshold"]
    calibrate_confidence_l1 = args["calibrate_confidence_l1"]
    calibrate_confidence_l2 = args["calibrate_confidence_l2"]

    # argument : create main_bbox for object tracking
    create_main_bbox_thres1 = args["create_main_bbox_threshold1"]
    create_main_bbox_thres2 = args["create_main_bbox_threshold2"]
    create_main_bbox_length = args["create_main_bbox_length"]

    # argument : object tracking
    tracking_iou_threshold = args["tracking_iou_threshold"]
    tracking_frame_length = args["tracking_frame_length"]
    tracking_time_threshold = args["tracking_time_threshold"]

    # argument : false positive한 Falldown 처리
    before_falldown_threshold = args["before_falldown_threshold"]
    before_falldown_remove_length = args["before_falldown_remove_length"]

    # argument : object concat
    concat_iou_threshold = args["concat_iou_threshold"]
    concat_intersection_threshold = args["concat_intersection_threshold"]
    concat_human_threshold = args["concat_human_threshold"]
    concat_falldown_threshold = args["concat_falldown_threshold"]

    # argument : object interpolate
    inp_boundary_iou_thres = args["inp_boundary_iou_threshold"]
    inp_boundary_intersection_thres = args[
        "inp_boundary_intersection_threshold"
    ]
    inp_boundary_time_thres = args["inp_boundary_time_threshold"]

    # time measure
    start_time = time.time()

    dataset_dir = args["dataset_dir"]
    folder = sorted(glob2.glob(dataset_dir + "*"))

    # yolo process
    data = yolo_process(
        folder,
        model_y_new,
        device,
        fps=fps,
        img_row=img_row,
        conf_thres=conf_thres,
        max_det=max_det,
    )

    # human_cls
    data = human_classification_process(data, classifier=None)

    # falldown_cls
    data = falldown_classification_process(
        data, classifier_list["falldown_classification"]
    )

    # build new structure
    data_branch1 = build_own_structure(data)

    # data handling
    data_branch2 = data_handling(
        data_branch1,
        yolo_threshold=yolo_threshold,
        human_threshold=human_threshold,
        falldown_threshold=falldown_threshold,
        calibrate_confidence_l1=calibrate_confidence_l1,
        calibrate_confidence_l2=calibrate_confidence_l2,
    )

    # create main bbox
    data_branch2 = create_main_bbox_process(
        data_branch2,
        create_main_bbox_thres1=create_main_bbox_thres1,
        create_main_bbox_thres2=create_main_bbox_thres2,
        create_main_bbox_length=create_main_bbox_length,
    )

    # object tracking
    data_branch2 = object_tracking(
        data_branch2,
        dt=tracking_time_threshold,
        iou_threshold=tracking_iou_threshold,
        frame_length_threshold=tracking_frame_length,
    )

    # object frame 누락 요소 찾기
    # data_branch3 = interpolate_object_tracking(
    #     data_branch2,
    #     inp_boundary_iou_thres=inp_boundary_iou_thres,
    #     inp_boundary_intersection_thres=inp_boundary_intersection_thres,
    #     inp_boundary_time_thres=inp_boundary_time_thres,
    # )

    data_branch3 = interpolate_object_tracking_revised(
        data_branch2,
        inp_boundary_iou_threshold=inp_boundary_iou_thres,
        inp_boundary_intersection_threshold=inp_boundary_intersection_thres,
        inp_boundary_time_threshold=inp_boundary_time_thres,
    )

    # boundary frame processing
    data_branch4 = handling_boundary_frame(
        data_branch3,
        before_falldown_threshold=before_falldown_threshold,
        before_falldown_remove_length=before_falldown_remove_length,
    )

    # convert original structure
    data_branch4 = convert_original_structure(data_branch4)

    # object concat
    data_branch5 = object_concat(
        data_branch4,
        concat_iou_threshold=concat_iou_threshold,
        concat_intersection_threshold=concat_intersection_threshold,
        concat_human_threshold=concat_human_threshold,
        concat_falldown_threshold=concat_falldown_threshold,
    )

    data_branch5_ = object_concat_next_step(
        data_branch5, iou_threshold=0.001, distance_threshold=10
    )

    data_branch6 = remove_conjugated_object(
        data_branch5_, iou_threshold=0.1, intersection_threshold=0.1
    )
    data_branch7 = remove_duplicated(data_branch6)

    # frame 간격 모두 채우기
    data_final = fill_result_all_frames(data_branch7)

    # submisison
    submission = build_submission_2021(data_final)

    with open(file_names["result"], "w") as f:
        json.dump(submission, f, ensure_ascii=False, indent=4)
