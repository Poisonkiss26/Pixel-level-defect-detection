import copy
import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import numpy.fft as fft
import pickle
from skimage.feature import graycomatrix, graycoprops


def detect_pixel(img, rect_list, label):  # 像素级别的检测
    mask = np.zeros_like(img)
    for rect in rect_list:  # 生成掩膜mask
        r1, r2, c1, c2 = rect[0][1], rect[1][1], rect[0][0], rect[1][0]
        mask[r1:r2, c1:c2] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (30, 30)))

    overlap = np.bitwise_and(mask, label)
    overlap_rate = np.sum(overlap) / np.sum(label)
    img_masked = copy.deepcopy(img)
    img_masked[mask == 0] = 0
    img = cv2.GaussianBlur(img, (3, 3), 0.5)
    img_grad = cv2.Laplacian(img, 0)
    th, img_grad = cv2.threshold(img_grad, 10, 255, cv2.THRESH_BINARY)
    k1 = np.array([[1, 1]])
    k2 = np.array([[1], [1]])
    img_grad = cv2.morphologyEx(img_grad, cv2.MORPH_OPEN, k2)
    img_grad = cv2.morphologyEx(img_grad, cv2.MORPH_OPEN, k1)
    img_grad = cv2.dilate(img_grad, k2)
    img_grad = cv2.dilate(img_grad, k1)
    img_grad[mask == 0] = 0
    pixel_overlap = np.bitwise_and(img_grad, label)
    pixel_overlap_rate = np.sum(pixel_overlap) / np.sum(label)
    # cv2.imshow('原图', img)
    # cv2.imshow('真实的标签', label)
    # cv2.imshow('套上掩膜的图像', img_masked)
    # cv2.imshow('检测出来的缺陷', img_grad)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    return overlap_rate, pixel_overlap_rate


def gen_dataset():  # 加载并整理测数据
    NG_img = []
    NG_labels = []
    NG_img_path = 'data/defect/NG/imagenormal/'
    label_path = 'data/defect/NG/imagedrawn/'
    label_names = os.listdir(label_path)
    for i in range(len(label_names)):
        NG_img.append(cv2.imread(f'{NG_img_path}{i}.bmp', 0))
        NG_labels.append(cv2.imread(f'{label_path}{i}.png', 0))

    OK_path = 'data/defect/OK/'
    OK_img = []
    OK_img_names = os.listdir(OK_path)
    for i in range(len(OK_img_names)):
        OK_img.append(cv2.imread(f'{OK_path}{i}.bmp', 0))

    return OK_img, NG_img, NG_labels


def light_balance(image, num_blocks_c, num_blocks_r, dst_avg=None):  # 对两个方向分别做光照均衡，稍微有点用
    mask = (image != 0).astype(np.int32)
    image = image.astype(np.double)
    if dst_avg is None:
        avg = np.mean(image)
    else:
        avg = dst_avg
    r, c = image.shape[:2]
    block_size_c = int(c / num_blocks_c)
    block_size_r = int(r / num_blocks_r)

    l2 = []
    for j in range(num_blocks_c):
        mask_2 = np.zeros_like(image, np.int32)
        c1 = j * block_size_c
        c2 = c1 + block_size_c
        mask_2[:, c1:c2] = 1
        mask_2[mask == 0] = 0
        if np.sum(mask_2) == 0:
            l2.append(0)
            continue
        block_image = image[mask_2 > 0]
        l2.append(np.mean(block_image))
    light = np.array(l2) - avg
    light = cv2.resize(light, (c, r), interpolation=cv2.INTER_CUBIC)
    res = image - light
    res[res < 0] = 0
    res[mask == 0] = 0

    l2 = []
    for j in range(num_blocks_r):
        mask_2 = np.zeros_like(image, np.int32)
        r1 = j * block_size_r
        r2 = r1 + block_size_r
        mask_2[r1:r2, :] = 1
        mask_2[mask == 0] = 0
        if np.sum(mask_2) == 0:
            l2.append(0)
            continue
        block_image = image[mask_2 > 0]
        l2.append(np.mean(block_image))
    light = np.array(l2) - avg
    light = cv2.resize(light, (c, r), interpolation=cv2.INTER_CUBIC)
    res = image - light
    res[res < 0] = 0
    res[mask == 0] = 0
    res = res.astype(np.uint8)
    return res, avg


def calculate_glcm(image, distances, angles):  # 灰度共生矩阵计算
    image[image < 0] = 0
    glcm = graycomatrix(image, distances, angles, levels=256, symmetric=True, normed=True)
    return glcm


def extract_texture_features(glcm):
    contrast = graycoprops(glcm, 'contrast')
    energy = graycoprops(glcm, 'energy')
    correlation = graycoprops(glcm, 'correlation')
    homogeneity = graycoprops(glcm, 'homogeneity')
    return contrast, energy, correlation, homogeneity


def GLCM(image, num_blocks_r, num_blocks_c, threshold):  # 分块做灰度共生矩阵分析，并将感兴趣的区域筛选出来
    img_copy = copy.deepcopy(image).copy()
    image = image.astype(np.int32)
    r, c = image.shape[:2]
    block_size_c = int(c / num_blocks_c)
    block_size_r = int(r / num_blocks_r)
    res_rect_list = []
    for i in np.arange(0, num_blocks_r):
        for j in np.arange(0, num_blocks_c):
            if i == 0 and j == 0 or i == 0 and j == num_blocks_c - 1 or i == num_blocks_r - 1 and j == 0 or i == num_blocks_r - 1 and j == num_blocks_c - 1:
                continue
            r1 = i * block_size_r
            r2 = r1 + block_size_r
            c1 = j * block_size_c
            c2 = c1 + block_size_c
            block_image = image[r1:r2, c1:c2]
            if not np.any(block_image):
                continue
            block_image[block_image == 0] = np.mean(block_image[block_image != 0])
            g = calculate_glcm(block_image, [1], [0, np.pi / 2, np.pi / 4, np.pi * 3 / 4])
            contrast, energy, correlation, homogeneity = extract_texture_features(g)
            if np.max(contrast) > threshold:
                img_copy = cv2.rectangle(img_copy, (c1, r1), (c2, r2), (255, 255, 255), 2)
                res_rect_list.append([[c1, r1], [c2, r2]])
            cv2.putText(img_copy,
                        f'{np.round(np.max(contrast), 1)}',
                        (int(0.75 * c1 + 0.25 * c2), int(0.5 * r1 + 0.5 * r2)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.25, (255, 255, 255), 1)
    return img_copy, res_rect_list


def subpixel_edge(grayImage):  # 亚像素边缘检测
    kernels_Num = 8
    kernels = ['_' for i in range(kernels_Num)]
    kernels[0] = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=int)
    kernels[1] = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype=int)
    kernels[2] = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=int)
    kernels[3] = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype=int)
    kernels[4] = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=int)
    kernels[5] = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]], dtype=int)
    kernels[6] = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=int)
    kernels[7] = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]], dtype=int)

    gradients = ['_' for i in range(kernels_Num)]
    for i in range(kernels_Num):
        gradients[i] = cv2.filter2D(grayImage, cv2.CV_16S, kernels[i])

    angle_list = [270, 315, 0, 45, 90, 135, 180, 225]
    amplitude = np.full(grayImage.shape, 0)
    angle = np.full(grayImage.shape, -64)

    for r in range(grayImage.shape[0]):
        pAmp = amplitude[r]
        pAng = angle[r]

        pGrad = ['_' for i in range(kernels_Num)]
        for i in range(kernels_Num):
            pGrad[i] = gradients[i][r]  # 不同方向（8个方向）上同一行的梯度
        for c in range(grayImage.shape[1]):
            for i in range(kernels_Num):  # 每个方向的梯度比较
                if (pAmp[c] < pGrad[i][c]):
                    pAmp[c] = pGrad[i][c]
                    pAng[c] = angle_list[i]

    edge = np.full(grayImage.shape, 0)
    edge.astype('uint8')
    thres = 100  # 阈值  设置最小幅度值
    for r in range(1, grayImage.shape[0] - 1):
        pAmp1 = amplitude[r - 1]
        pAmp2 = amplitude[r]
        pAmp3 = amplitude[r + 1]

        pAng = angle[r]
        pEdge = edge[r]
        for c in range(1, grayImage.shape[1] - 1):

            if (pAmp2[c] < thres):
                continue
            if pAng[c] == 270:
                if pAmp2[c] > pAmp1[c] and pAmp2[c] >= pAmp3[c]:
                    pEdge[c] = 255
            elif pAng[c] == 90:
                if pAmp2[c] >= pAmp1[c] and pAmp2[c] > pAmp3[c]:
                    pEdge[c] = 255
            elif pAng[c] == 315:
                if pAmp2[c] > pAmp1[c - 1] and pAmp2[c] >= pAmp3[c + 1]:
                    pEdge[c] = 255
            elif pAng[c] == 135:
                if pAmp2[c] >= pAmp1[c - 1] and pAmp2[c] > pAmp3[c + 1]:
                    pEdge[c] = 255
            elif pAng[c] == 0:
                if pAmp2[c] > pAmp2[c - 1] and pAmp2[c] >= pAmp2[c + 1]:
                    pEdge[c] = 255
            elif pAng[c] == 180:
                if pAmp2[c] >= pAmp2[c - 1] and pAmp2[c] > pAmp2[c + 1]:
                    pEdge[c] = 255
            elif pAng[c] == 45:
                if pAmp2[c] >= pAmp1[c + 1] and pAmp2[c] > pAmp3[c - 1]:
                    pEdge[c] = 255
            elif pAng[c] == 225:
                if pAmp2[c] > pAmp1[c + 1] and pAmp2[c] >= pAmp3[c - 1]:
                    pEdge[c] = 255

    edge = cv2.convertScaleAbs(edge)
    root2 = np.sqrt(2.0)
    tri_list = [[0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0]]
    for i in range(kernels_Num):
        tri_list[0][i] = np.cos(angle_list[i] * np.pi / 180.0)
        tri_list[1][i] = -np.sin(angle_list[i] * np.pi / 180.0)
    vPts = []

    for r in range(1, grayImage.shape[0] - 1):
        pAmp1 = amplitude[r - 1]
        pAmp2 = amplitude[r]
        pAmp3 = amplitude[r + 1]

        pAng = angle[r]
        pEdge = edge[r]
        for c in range(1, grayImage.shape[1] - 1):
            if (pEdge[c]):
                nAngTmp = 0
                dTmp = 0
                if pAng[c] == 270:
                    nAngTmp = 0
                    dTmp = (pAmp1[c] - pAmp3[c]) / (pAmp1[c] + pAmp3[c] - 2 * pAmp2[c]) * 0.5
                elif pAng[c] == 90:
                    nAngTmp = 4
                    dTmp = -(pAmp1[c] - pAmp3[c]) / (pAmp1[c] + pAmp3[c] - 2 * pAmp2[c]) * 0.5
                elif pAng[c] == 315:
                    nAngTmp = 1
                    dTmp = (pAmp1[c - 1] - pAmp3[c + 1]) / (pAmp1[c - 1] + pAmp3[c + 1] - 2 * pAmp2[c]) * root2 * 0.5
                elif pAng[c] == 135:
                    nAngTmp = 5
                    dTmp = -(pAmp1[c - 1] - pAmp3[c + 1]) / (pAmp1[c - 1] + pAmp3[c + 1] - 2 * pAmp2[c]) * root2 * 0.5
                elif pAng[c] == 0:
                    nAngTmp = 2
                    dTmp = (pAmp2[c - 1] - pAmp2[c + 1]) / (pAmp2[c - 1] + pAmp2[c + 1] - 2 * pAmp2[c]) * 0.5
                elif pAng[c] == 180:
                    nAngTmp = 6
                    dTmp = -(pAmp2[c - 1] - pAmp2[c + 1]) / (pAmp2[c - 1] + pAmp2[c + 1] - 2 * pAmp2[c]) * 0.5
                elif pAng[c] == 45:
                    nAngTmp = 3
                    dTmp = (pAmp3[c - 1] - pAmp1[c + 1]) / (pAmp1[c + 1] + pAmp3[c - 1] - 2 * pAmp2[c]) * root2 * 0.5
                elif pAng[c] == 225:
                    nAngTmp = 7
                    dTmp = -(pAmp3[c - 1] - pAmp1[c + 1]) / (pAmp1[c + 1] + pAmp3[c - 1] - 2 * pAmp2[c]) * root2 * 0.5

                x = c + dTmp * tri_list[0][nAngTmp]
                y = r + dTmp * tri_list[1][nAngTmp]
                vPts.append([x, y])
    tmpImg = np.zeros(grayImage.shape, dtype=np.uint8)
    for x, y in vPts:
        tmpImg[int(y), int(x)] = 255
    return tmpImg


def tile_correction(img, imgRaw):  # 倾斜矫正1
    img = np.pad(img, pad_width=((0, 0), (128, 128)))
    imgRaw = np.pad(imgRaw, pad_width=((0, 0), (128, 128)))
    angleList = np.round(np.arange(80, 100, 0.1), 1)
    resList = []
    for angle in angleList:
        mat = cv2.getRotationMatrix2D((640, 640), angle, 1)
        imgRot = cv2.warpAffine(img, mat, (1280, 1280))
        imgSum = np.any(imgRot, axis=0)
        resList.append(np.sum(imgSum > 0))
    expectedAngle = angleList[np.argmin(resList)]
    mat = cv2.getRotationMatrix2D((640, 640), expectedAngle, 1)
    imgCorrect = cv2.warpAffine(imgRaw, mat, (1280, 1280))
    imgCorrect = subpixel_edge(imgCorrect[:, 200:800])
    imgCorrect = cv2.morphologyEx(imgCorrect, cv2.MORPH_CLOSE, np.ones((3, 3)))
    return imgCorrect, expectedAngle


def tile_correction_2(thres, imgRaw, label=None):  # 倾斜校正2，给缺陷检测任务用的，和上面的大致差不多
    r, c = thres.shape[0], thres.shape[1]
    dst_size = 1100
    pad_width_c = int((dst_size - c) / 2)
    pad_width_r = int((dst_size - r) / 2)
    thres = np.pad(thres, pad_width=((pad_width_r, pad_width_r), (pad_width_c, pad_width_c)))
    label = np.pad(label,
                   pad_width=((pad_width_r, pad_width_r), (pad_width_c, pad_width_c))) if label is not None else None
    imgRaw = np.pad(imgRaw, pad_width=((pad_width_r, pad_width_r), (pad_width_c, pad_width_c)))
    angleList = np.round(np.arange(80, 100, 0.1), 1)
    resList = []
    for angle in angleList:
        mat = cv2.getRotationMatrix2D((481, 372), angle, 1)
        imgRot = cv2.warpAffine(thres, mat, (1100, 1100))
        imgAny = np.any(imgRot, axis=0)
        resList.append(np.sum(imgAny))
    expectedAngle = angleList[np.argmin(resList)]
    mat = cv2.getRotationMatrix2D((550, 550), expectedAngle, 1)
    thresCorrect = cv2.warpAffine(thres, mat, (1100, 1100))
    th, thresCorrect = cv2.threshold(thresCorrect, 127, 255, cv2.THRESH_BINARY)
    thresCorrect = thresCorrect[pad_width_c:-pad_width_c, pad_width_r:-pad_width_r]
    inds = np.where(thresCorrect)
    inds = np.stack([inds[0], inds[1]]).T
    # a = 35
    r_top_clip = inds[:, 0].min()
    r_button_clip = inds[:, 0].max()
    c_left_clip = inds[:, 1].min()
    c_right_clip = inds[:, 1].max()

    imgCorrect = cv2.warpAffine(imgRaw, mat, (1100, 1100))
    imgCorrect = imgCorrect[pad_width_c:-pad_width_c, pad_width_r:-pad_width_r]
    labelCorrect = cv2.warpAffine(label, mat, (1100, 1100)) if label is not None else None
    labelCorrect = labelCorrect[pad_width_c:-pad_width_c, pad_width_r:-pad_width_r] if label is not None else None
    # cv2.imshow('裁剪之前的图', imgCorrect)
    imgCorrect = imgCorrect[:r_button_clip]
    imgCorrect = imgCorrect[r_top_clip:]
    imgCorrect = imgCorrect[:, :c_right_clip]
    imgCorrect = imgCorrect[:, c_left_clip:]
    # cv2.imshow('label', labelCorrect)
    # cv2.imshow('correctimg', imgCorrect)
    # cv2.waitKey(-1)
    # cv2.destroyAllWindows()
    if label is not None:
        labelCorrect = labelCorrect[:r_button_clip]
        labelCorrect = labelCorrect[r_top_clip:]
        labelCorrect = labelCorrect[:, :c_right_clip]
        labelCorrect = labelCorrect[:, c_left_clip:]
        # cv2.imshow('label', labelCorrect)
        # cv2.imshow('correctimg', imgCorrect)
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()
        return imgCorrect, labelCorrect, expectedAngle
    return imgCorrect, expectedAngle


def gen_contours(inds, img):  # 顺时针生成正确顺序的边缘
    first_ind = copy.deepcopy(inds[0, :])
    res_list = [first_ind]
    present_ind = copy.deepcopy(first_ind)
    inds = inds[1:, :]
    operate_list = np.array(
        [[0, -1], [1, 1], [0, 1], [0, 1], [1, -1], [1, -1], [0, -1], [0, -1], [0, -1], [1, 1], [1, 1],
         [1, 1], [0, 1], [0, 1], [0, 1], [0, 1], [1, -1], [1, -1], [1, -1], [1, -1], [0, -1], [0, -1],
         [0, -1], [0, -1]])
    while True:  # 按顺时针往8+16个方向找下一个邻接的像素边缘
        sign = 0
        for p in operate_list:
            present_ind[p[0]] += p[1]
            n = np.argwhere(np.all(present_ind == inds, axis=1))
            if len(n):
                if np.all(present_ind == first_ind):
                    break
                inds = np.delete(inds, n, axis=0)
                res_list.append(copy.deepcopy(present_ind))
                sign = 1
                break
        if not sign:
            break  # 如果8个邻域都没有找到像素点，就提前结束
    res_list.append(first_ind)
    res_list = np.array(res_list)
    return res_list


def thinner_edge(inds, img):  # 为了尽量保证subpixel边缘的连续性，对边缘做了闭运算，这一定程度上加粗了边缘，所以需要重新细化
    j = 0
    for i in range(inds.shape[0]):
        ind = inds[j]
        r = ind[0]
        c = ind[1]
        piece = [img[r, c], img[r - 1, c], img[r, c + 1]]
        if np.all(piece):
            inds = np.delete(inds, j, axis=0)
            img[r, c] = 0
            j -= 1
            continue
        piece = [img[r, c], img[r - 1, c], img[r, c - 1]]
        if np.all(piece):
            inds = np.delete(inds, j, axis=0)
            img[r, c] = 0
            j -= 1
            continue
        piece = [img[r, c], img[r + 1, c], img[r, c + 1]]
        if np.all(piece):
            inds = np.delete(inds, j, axis=0)
            img[r, c] = 0
            j -= 1
            continue
        piece = [img[r, c], img[r + 1, c], img[r, c - 1]]
        if np.all(piece):
            inds = np.delete(inds, j, axis=0)
            img[r, c] = 0
            j -= 1
            continue
        j += 1
    return inds, img


def eliminate_thin_curve(ok_img):  # 在频域中，对于水平和垂直的分量，只保留低频部分
    img_f = fft.fftshift(fft.fft2(ok_img))
    cr, cc = int(img_f.shape[0] / 2), int(img_f.shape[1] / 2)
    k = 2
    j = 1
    # plt.figure()
    # plt.subplot(121)
    # plt.imshow(-np.log10(np.abs(img_f)+1),cmap='gray')
    img_f[cr - k:cr + k, :cc - j] /= np.arange(1, img_f[cr - k:cr + k, :cc - j].shape[1] + 1)[::-1]
    img_f[cr - k:cr + k, cc + j:] /= np.arange(1, img_f[cr - k:cr + k, cc + j:].shape[1] + 1)
    # img_f[cr - k:cr + k, cc + j:] /= np.arange(1, img_f[cr - k:cr + k, cc + j:].shape[1] + 1)
    # plt.subplot(122)
    # plt.imshow(-np.log10(np.abs(img_f) + 1),cmap='gray')
    # plt.show()
    img_recon_1 = np.abs(fft.ifft2(fft.ifftshift(img_f))).astype(np.uint8)
    th, img_recon_1_thres = cv2.threshold(img_recon_1, 125, 255, cv2.THRESH_BINARY)
    img_recon_1_thres[:17] = 255
    img_recon_1_thres[583:] = 255
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 5))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    img_recon_1_thres = cv2.morphologyEx(img_recon_1_thres, cv2.MORPH_OPEN, kernel1)
    img_recon_1_thres = cv2.morphologyEx(img_recon_1_thres, cv2.MORPH_CLOSE, kernel2)
    img_recon_1[img_recon_1_thres > 0] = 0
    img_f = fft.fftshift(fft.fft2(img_recon_1))
    img_f[cr - k:cr + k, :cc - j] = 0
    img_f[cr - k:cr + k, cc + j:] = 0
    img_f[:cr - j, cc - k:cc + k] = 0
    img_f[cr - j:, cc - k:cc + k] = 0
    img_recon_2 = np.abs(fft.ifft2(fft.ifftshift(img_f))).astype(np.uint8)
    img_recon_1_thres = cv2.dilate(img_recon_1_thres, np.ones((9, 9)))
    img_recon_2[img_recon_1_thres > 0] = 0
    return img_recon_1, img_recon_2, img_recon_1_thres


def save(file_path, lst):
    with open(file_path, 'wb') as file:
        pickle.dump(lst, file)


def load(file_path):
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    return data
