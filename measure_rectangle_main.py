from utils import *

if not os.path.exists('./data/measure_correctedEdge'):
    os.mkdir('./data/measure_correctedEdge')
    imgFileNames = os.listdir('./data/measure/')
    for name in imgFileNames:
        path = './data/measure/' + name
        img = cv2.imread(path, 0)
        img = 255 - img
        img[:200, :] = 0
        img[800:, :] = 0
        img[img < 220] = 0  # 去噪
        imgCopy = copy.deepcopy(img)
        # cv2.imshow('', cv2.resize(img, dsize=(640, 512)))
        # cv2.waitKey(-1)
        # cv2.destroyAllWindows()
        imgCorrect, rot_angle = tile_correction(img, imgCopy)  # 纠正倾斜，然后用亚像素检测边缘
        cv2.imwrite(f'data/measure_correctedEdge/{name}', imgCorrect)
        print(f'为纠正图像{name}的倾斜,\t将其旋转{rot_angle}度')

imgFileNames = os.listdir('./data/measure_correctedEdge/')
rectRateList = []
heightList = []
widthList = []
# bad_list = [21, 43]
for i, name in enumerate(imgFileNames):
    path = './data/measure_correctedEdge/' + name
    imgEdge = cv2.imread(path, 0)
    inds = np.where(imgEdge)  # 所有边缘的坐标
    inds = np.stack([inds[0], inds[1]], axis=1)
    inds, imgEdge = thinner_edge(inds, imgEdge)
    inds = gen_contours(inds, imgEdge)
    rMin = np.min(inds[:, 0])  # 最小外接矩形的顶点坐标
    rMax = np.max(inds[:, 0])
    cMin = np.min(inds[:, 1])
    cMax = np.max(inds[:, 1])
    imgEdge = cv2.drawContours(imgEdge, (np.expand_dims(inds[:, ::-1], 1)), -1, color=(128, 128, 128), thickness=1)
    heightList.append(rMax - rMin)
    widthList.append(cMax - cMin)
    tempArea = (rMax - rMin) * (cMax - cMin)  # 最大外接矩形的面具
    ctArea = cv2.contourArea(np.expand_dims(inds, 1))  # 提取到的边缘的面积
    rectangleRate = ctArea / tempArea  # 矩形度计算
    print(f"图像{i}\t{name}\t的矩形度:\t{rectangleRate}")
    rectRateList.append(rectangleRate)
mean = np.mean(rectRateList)
variance = np.var(rectRateList)

plt.figure()
plt.scatter(rectRateList, np.ones_like(rectRateList))
plt.title(
    f'Distribution of Rectangle Rate  mean={np.round(np.mean(rectRateList), 5)}  variance={np.round(np.var(rectRateList), 8)}')
plt.figure()
plt.scatter(heightList, np.ones_like(heightList))
plt.title(f'Height  mean={np.round(np.mean(heightList), 5)}  variance={np.round(np.var(heightList), 5)})')
plt.figure()
plt.scatter(widthList, np.ones_like(widthList))
plt.title(f'Width  mean={np.round(np.mean(widthList), 5)}  variance={np.round(np.var(widthList), 5)}')
plt.show()
print(f'矩形度的均值:{mean}\n矩形度的方差:{variance}')
