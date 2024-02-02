from tools import saveRed
import cv2
import numpy as np
import tools

# 把图片中红色的“对号”和“错号”保存成一张一张的小图片
def getRedRightWrong(imagePath, index = 0):
    localimage_orign = cv2.imread(imagePath)
    localimage_gray = cv2.cvtColor(localimage_orign, cv2.COLOR_BGR2GRAY)
    localimage_savered = saveRed(localimage_orign)

    kernel = np.ones((3,3),np.uint8)
    localimage = cv2.erode(localimage_savered,kernel,iterations = 1)

    # 膨胀
    kernel = np.ones((25,25),np.uint8)
    localimage = cv2.dilate(localimage,kernel,iterations = 1)

    # kernel = np.ones((10,10),np.uint8)
    # localimage = cv2.erode(localimage,kernel,iterations = 1)
    # cellImage = cv2.morphologyEx(localimage, cv2.MORPH_, kernel)
    cv2.imwrite('out.jpg', localimage)

    contours, hierarchy = cv2.findContours(localimage, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    # 创建一个全黑的掩膜
    # mask = np.zeros_like(localimage_orign)

    # # 使用白色填充轮廓，生成掩膜
    # cv2.drawContours(mask, contours, -1, (255,255,255), thickness=cv2.FILLED)
    # cv2.imwrite('out.jpg', mask)

    # # 使用掩膜扣出轮廓内的图像
    # result = cv2.bitwise_and(localimage_orign, mask)
    # cv2.imwrite('out.jpg', result)

    for cnt in contours:
        (x, y, w, h) = cv2.boundingRect(cnt)
        # area_nocontours = cv2.contourArea(cnt)            
        # area = imageHelper.realContourArea(image_erode, cnt)           
        area = w * h
        if area > 200 and (w > 80 or h > 80):
        # rect = cv2.minAreaRect(cnt)
            # 创建一个全黑的掩膜
            mask = np.zeros_like(localimage_gray)

            # 使用白色填充轮廓，生成掩膜
            cv2.drawContours(mask, [cnt], -1, (255,255,255), thickness=cv2.FILLED)
            # 只截取cnt的图片
            image_cnt = localimage_savered[y:y + h, x:x + w]
            image_cnt_mask = mask[y:y + h, x:x + w]


            # 使用掩膜扣出轮廓内的图像
            result = cv2.bitwise_and(image_cnt, image_cnt_mask)
            # 填充result的边框让长和宽一样，填充黑色
            # 计算需要添加的边框的大小
            if h > w:
                top = bottom = 0
                left = right = (h - w) // 2
            else:
                top = bottom = (w - h) // 2
                left = right = 0

            # 添加边框
            result = cv2.copyMakeBorder(result, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])

            imageName = 'result/_' + str(index) + '.jpg'
            cv2.imwrite(imageName, result)
            index += 1


# getRedRightWrong("6.jpg", 34)
# 生成训练数据txt文件
tools.print_directory_contents('dataset/test/1', '1')