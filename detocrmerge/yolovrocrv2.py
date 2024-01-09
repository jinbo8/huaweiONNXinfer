import onnxruntime
import numpy as np
import cv2
import copy
import os
import argparse
from PIL import Image, ImageDraw, ImageFont
import time

plateName = r"#京沪津渝冀晋蒙辽吉黑苏浙皖闽赣鲁豫鄂湘粤桂琼川贵云藏陕甘青宁新学警港澳挂使领民航危0123456789ABCDEFGHJKLMNPQRSTUVWXYZ险品"
mean_value, std_value = ((0.588, 0.193))  # 识别模型均值标准差

def allFilePath(rootPath, allFIleList):  # 遍历文件
    # jwang 0
    fileList = os.listdir(rootPath)
    for temp in fileList:
        if os.path.isfile(os.path.join(rootPath, temp)):
            allFIleList.append(os.path.join(rootPath, temp))
        else:
            allFilePath(os.path.join(rootPath, temp), allFIleList)


def detect_pre_precessing(img, img_size):  # 检测前处理
    # jwang1

    img, r, left, top = my_letter_box(img, img_size)
    # cv2.imwrite("1.jpg",img)
    img = img[:, :, ::-1].transpose(2, 0, 1).copy().astype(np.float32)
    img = img / 255
    img = img.reshape(1, *img.shape)
    return img, r, left, top


def my_letter_box(img, size=(640, 640)):
    #jwang2
    h, w, c = img.shape
    r = min(size[0] / h, size[1] / w)
    new_h, new_w = int(h * r), int(w * r)
    top = int((size[0] - new_h) / 2)
    left = int((size[1] - new_w) / 2)

    bottom = size[0] - new_h - top
    right = size[1] - new_w - left
    img_resize = cv2.resize(img, (new_w, new_h))
    img = cv2.copyMakeBorder(img_resize, top, bottom, left, right, borderType=cv2.BORDER_CONSTANT,
                             value=(114, 114, 114))
    return img, r, left, top

def decodePlate(preds):  # 识别后处理
    #jwang14
    pre = 0
    newPreds = []
    for i in range(len(preds)):
        if preds[i] != 0 and preds[i] != pre:
            newPreds.append(preds[i])
        pre = preds[i]
    plate = ""
    for i in newPreds:
        plate += plateName[int(i)]
    return plate
    # return newPreds


def rec_pre_precessing(img, size=(48, 168)):  # 识别前处理
    # jwang13
    img = cv2.resize(img, (168, 48))
    img = img.astype(np.float32)
    img = (img / 255 - mean_value) / std_value  # 归一化 减均值 除标准差
    img = img.transpose(2, 0, 1)  # h,w,c 转为 c,h,w
    img = img.reshape(1, *img.shape)  # channel,height,width转为batch,channel,height,channel
    return img



def four_point_transform(image, pts):  # 透视变换得到矫正后的图像，方便识别
    #jwang3
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    # return the warped image
    return warped

def draw_result(orgimg, dict_list):
    #jwang4
    result_str = ""
    for result in dict_list:
        rect_area = result['bbox_x1y1x2y2']

        x, y, w, h = rect_area[0], rect_area[1], rect_area[2] - rect_area[0], rect_area[3] - rect_area[1]
        padding_w = 0.05 * w
        padding_h = 0.11 * h
        rect_area[0] = max(0, int(x - padding_w))
        rect_area[1] = min(orgimg.shape[1], int(y - padding_h))
        rect_area[2] = max(0, int(rect_area[2] + padding_w))
        rect_area[3] = min(orgimg.shape[0], int(rect_area[3] + padding_h))

        height_area = result['roi_height']
        landmarks = result['landmarks']
        result = result['plate_number']
        result_str += result + " "
        for i in range(4):  # 关键点
            cv2.circle(orgimg, (int(landmarks[i][0]), int(landmarks[i][1])), 5, clors[i], -1)
        cv2.rectangle(orgimg, (rect_area[0], rect_area[1]), (rect_area[2], rect_area[3]), (0, 0, 255), 2)  # 画框
        if len(result) >= 1:
            orgimg = cv2ImgAddText(orgimg, result, rect_area[0] - height_area, rect_area[1] - height_area - 10,
                                   (255, 0, 0), height_area)
    # print(result_str)
    return orgimg

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):  # 将识别结果画在图上
    # jwang5
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontText = ImageFont.truetype(
        "fonts/platech.ttf", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontText)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

def post_precessing(dets, r, left, top, conf_thresh=0.3, iou_thresh=0.45):  # 检测后处理
    #jwang 6
    num_cls = 2
    choice = dets[:, :, 4] > conf_thresh
    dets = dets[choice]
    dets[:, 5:5 + num_cls] *= dets[:, 4:5]  # 5::7是2分类类别分数
    box = dets[:, :4]
    boxes = xywh2xyxy(box)
    score = np.max(dets[:, 5:5 + num_cls], axis=-1, keepdims=True)
    index = np.argmax(dets[:, 5:5 + num_cls], axis=-1).reshape(-1, 1)
    kpt_b = 5 + num_cls
    landmarks = dets[:, [kpt_b, kpt_b + 1, kpt_b + 3, kpt_b + 4, kpt_b + 6, kpt_b + 7, kpt_b + 9,
                         kpt_b + 10]]  # yolov7关键有三个数，x,y,score，这里我们只需要x,y
    output = np.concatenate((boxes, score, landmarks, index), axis=1)
    reserve_ = my_nms(output, iou_thresh)
    output = output[reserve_]
    output = restore_box(output, r, left, top)
    return output



def get_split_merge(img):  # 双层车牌进行分割后识别
    #jwang7
    h, w, c = img.shape
    img_upper = img[0:int(5 / 12 * h), :]
    img_lower = img[int(1 / 3 * h):, :]
    img_upper = cv2.resize(img_upper, (img_lower.shape[1], img_lower.shape[0]))
    new_img = np.hstack((img_upper, img_lower))
    return new_img


def order_points(pts):  # 关键点排列 按照（左上，右上，右下，左下）的顺序排列
    #jwang8
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect


def get_plate_result(img, session_rec):  # 识别后处理
    #jwang9
    img = rec_pre_precessing(img)
    # print(f"img:{img.shape}")

    y_onnx = session_rec.run([session_rec.get_outputs()[0].name], {session_rec.get_inputs()[0].name: img})[0]
    # print(y_onnx[0])
    # print(y_onnx)
    index = np.argmax(y_onnx[0], axis=1)  # 找出概率最大的那个字符的序号
    # print(y_onnx[0])
    plate_no = decodePlate(index)
    # plate_no = decodePlate(y_onnx[0])
    return plate_no

def xywh2xyxy(boxes):  # xywh坐标变为 左上 ，右下坐标 x1,y1  x2,y2
    #jwang10
    xywh = copy.deepcopy(boxes)
    xywh[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
    xywh[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
    xywh[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
    xywh[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
    return xywh


def my_nms(boxes, iou_thresh):  # nms
    #jwang11
    index = np.argsort(boxes[:, 4])[::-1]
    keep = []
    while index.size > 0:
        i = index[0]
        keep.append(i)
        x1 = np.maximum(boxes[i, 0], boxes[index[1:], 0])
        y1 = np.maximum(boxes[i, 1], boxes[index[1:], 1])
        x2 = np.minimum(boxes[i, 2], boxes[index[1:], 2])
        y2 = np.minimum(boxes[i, 3], boxes[index[1:], 3])

        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)

        inter_area = w * h
        union_area = (boxes[i, 2] - boxes[i, 0]) * (boxes[i, 3] - boxes[i, 1]) + (
                    boxes[index[1:], 2] - boxes[index[1:], 0]) * (boxes[index[1:], 3] - boxes[index[1:], 1])
        iou = inter_area / (union_area - inter_area)
        idx = np.where(iou <= iou_thresh)[0]
        index = index[idx + 1]
    return keep


def restore_box(boxes, r, left, top):  # 返回原图上面的坐标
    # jwang12
    boxes[:, [0, 2, 5, 7, 9, 11]] -= left
    boxes[:, [1, 3, 6, 8, 10, 12]] -= top

    boxes[:, [0, 2, 5, 7, 9, 11]] /= r
    boxes[:, [1, 3, 6, 8, 10, 12]] /= r
    return boxes


#------------------------------------------------------ocr---------------------------------------------------------
'''
测试转出的onnx模型
'''
import cv2
import numpy
import numpy as np

import torch
import onnxruntime as rt
import math
import os


class TestOnnx:
    def __init__(self, onnx_file, character_dict_path, use_space_char=True):
        self.sess = rt.InferenceSession(onnx_file)
        # 获取输入节点名称
        self.input_names = [input.name for input in self.sess.get_inputs()]
        # 获取输出节点名称
        self.output_names = [output.name for output in self.sess.get_outputs()]

        self.character = []
        self.character.append("blank")
        with open(character_dict_path, "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                self.character.append(line)
        if use_space_char:
            self.character.append(" ")

    def resize_norm_img(self, img, image_shape=[3, 48, 168]):
        imgC, imgH, imgW = image_shape
        h = img.shape[0]
        w = img.shape[1]
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
        resized_image = resized_image.astype('float32')
        if image_shape[0] == 1:
            resized_image = resized_image / 255
            resized_image = resized_image[np.newaxis, :]
        else:
            resized_image = resized_image.transpose((2, 0, 1)) / 255
        resized_image -= 0.5
        resized_image /= 0.5
        padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
        padding_im[:, :, 0:resized_w] = resized_image
        return padding_im

    # # 准备模型运行的feed_dict
    def process(self, input_names, image):
        feed_dict = dict()
        for input_name in input_names:
            feed_dict[input_name] = image

        return feed_dict

    def get_ignored_tokens(self):
        return [0]

    def decode(self, text_index, text_prob=None, is_remove_duplicate=False):
        """ convert text-index into text-label. """
        result_list = []
        ignored_tokens = self.get_ignored_tokens()
        batch_size = len(text_index)
        for batch_idx in range(batch_size):
            selection = np.ones(len(text_index[batch_idx]), dtype=bool)
            if is_remove_duplicate:
                selection[1:] = text_index[batch_idx][1:] != text_index[
                                                                 batch_idx][:-1]
            for ignored_token in ignored_tokens:
                selection &= text_index[batch_idx] != ignored_token

            char_list = [
                self.character[int(text_id)].replace('\n', '')
                for text_id in text_index[batch_idx][selection]
            ]
            if text_prob is not None:
                conf_list = text_prob[batch_idx][selection]
            else:
                conf_list = [1] * len(selection)
            if len(conf_list) == 0:
                conf_list = [0]

            text = ''.join(char_list)
            result_list.append((text, np.mean(conf_list).tolist()))

        return result_list

    def test(self, image_path):
        img_onnx = cv2.imread(image_path)
        # img_onnx = cv2.resize(img_onnx, (320, 32))
        # img_onnx = img_onnx.transpose((2, 0, 1)) / 255
        img_onnx = self.resize_norm_img(img_onnx)
        onnx_indata = img_onnx[np.newaxis, :, :, :]
        onnx_indata = torch.from_numpy(onnx_indata)
        # print('diff:', onnx_indata - input_data)
        # print('image shape: ', onnx_indata.shape)
        onnx_indata = np.array(onnx_indata, dtype=np.float32)
        feed_dict = self.process(self.input_names, onnx_indata)

        output_onnx = self.sess.run(self.output_names, feed_dict)
        # print('output_onnx[0].shape: ', output_onnx[0].shape)
        # print(' output_onnx[0]: ', output_onnx[0])

        output_onnx = numpy.asarray(output_onnx[0])

        preds_idx = output_onnx.argmax(axis=2)
        preds_prob = output_onnx.max(axis=2)
        post_result = self.decode(preds_idx, preds_prob, is_remove_duplicate=True)
        # print(f"post_result:{post_result}")
        if isinstance(post_result, dict):
            rec_info = dict()
            for key in post_result:
                if len(post_result[key][0]) >= 2:
                    rec_info[key] = {
                        "label": post_result[key][0][0],
                        "score": float(post_result[key][0][1]),
                    }
            print(f"image1:{image_path}", rec_info)
        else:
            if len(post_result[0]) >= 2:
                # info = post_result[0][0] + "\t" + str(post_result[0][1])
                info = post_result[0][0]
                info_conf = post_result[0][1]
            print(f"image2:",image_path, info, info_conf)
            return info, info_conf



if __name__ == "__main__":

    # https://blog.csdn.net/qq_22764813/article/details/133787584?spm=1001.2101.3001.6650.2&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-2-133787584-blog-115270800.235%5Ev38%5Epc_relevant_sort_base3&utm_relevant_index=5
    parser = argparse.ArgumentParser()
    parser.add_argument('--character_dict_path', type=str, default= '/home/dell/桌面/huaweiDetOCR/ONNX_infer_image_jwang/20240109ocr_onnx_infer_script/chinese_plate_dict.txt', help='ocr dict path(s)')  # 检测模型
    parser.add_argument('--detect_model', type=str, default='/home/dell/桌面/huaweiDetOCR/ONNX_infer_image_jwang/lpDet/yolov7-lite-s.onnx', help='model.pt path(s)')  # 检测模型
    parser.add_argument('--rec_model', type=str, default='/home/dell/桌面/huaweiDetOCR/ONNX_infer_image_jwang/20240109ocr_onnx_infer_script/ocr_rec111302.onnx', help='model.pt path(s)')  # 识别模型
    parser.add_argument('--image_path', type=str, default='/home/dell/桌面/huaweiDetOCR/images/img', help='source')
    parser.add_argument('--img_size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--output', type=str, default='/home/dell/桌面/huaweiDetOCR/ONNX_infer_image_jwang/drawed', help='source')
    opt = parser.parse_args()
    file_list = []
    allFilePath(opt.image_path, file_list)
    providers = ['CPUExecutionProvider']
    clors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    img_size = (768, 1280)
    session_detect = onnxruntime.InferenceSession(opt.detect_model, providers=providers)
    session_rec = onnxruntime.InferenceSession(opt.rec_model, providers=providers)
    if not os.path.exists(opt.output):
        os.mkdir(opt.output)
    save_path = opt.output
    count = 0
    begin = time.time()
    for pic_ in file_list:
        count += 1
        print(count, pic_, end=" ")
        img = cv2.imread(pic_)
        # print(f"img:{img}")
        img0 = copy.deepcopy(img)
        img, r, left, top = detect_pre_precessing(img, img_size)  # 检测前处理
        # print(img.shape)
        y_onnx = session_detect.run([session_detect.get_outputs()[0].name], {session_detect.get_inputs()[0].name: img})[0]
        outputs = post_precessing(y_onnx, r, left, top)  # 检测后处理
        # print(f"outputs:{outputs.shape}")
        # print(f"outputs:{outputs}")

        # -----ocr----
        testobj = TestOnnx(opt.rec_model, opt.character_dict_path)
        img2 = cv2.imread(pic_)
        dict_list = []
        for boxlp in outputs:
            print(f"boxlp:{boxlp}")
            # print(f"boxlp:{boxlp}")
            # print(f"boxlp0:{boxlp[0]}")
            # print(f"boxlp1:{boxlp[1]}")
            # print(f"boxlp2:{boxlp[2]}")
            # print(f"boxlp3:{boxlp[3]}")
            # lp_bbox_data = img2[int(boxlp[0]):int(boxlp[2]), int(boxlp[1]):int(boxlp[3])]
            lp_bbox_data = img2[int(boxlp[1]):int(boxlp[3]), int(boxlp[0]):int(boxlp[2])]
            save_img_path = os.path.join('/home/dell/桌面/huaweiDetOCR/ONNX_infer_image_jwang/20240109ocr_onnx_infer_script/res', str(boxlp[1])+'.jpg')
            result, lpconf = testobj.test(save_img_path)
            print(f'res0109:{save_img_path}:{result}, {lpconf}')
            # cv2.imwrite(save_img_path, lp_bbox_data)
            result_dict = {}
            rect = boxlp[:4].tolist()
            land_marks = boxlp[5:13].reshape(4, 2)
            roi_img = four_point_transform(img0, land_marks)
            # print(f"roi_img:{roi_img.shape}")

            label = int(boxlp[-1])
            score = boxlp[4]
            if label == 1:  # 代表是双层车牌
                roi_img = get_split_merge(roi_img)
            plate_no = get_plate_result(roi_img, session_rec)
            result_dict['bbox_x1y1x2y2'] = rect
            result_dict['landmarks'] = land_marks.tolist()
            result_dict['landmarks_conf'] = label
            result_dict['plate_number'] = result
            result_dict['plate_number_conf'] = lpconf
            result_dict['roi_height'] = roi_img.shape[0]
            dict_list.append(result_dict)

        print(f"dict_list:{dict_list}")

        # 检测识别结果绘制到图像上
        ori_img = draw_result(img0, dict_list)
        img_name = os.path.basename(pic_)
        save_img_path = os.path.join(save_path, img_name)
        cv2.imwrite(save_img_path, ori_img)
    print(f"总共耗时{time.time() - begin} s")


