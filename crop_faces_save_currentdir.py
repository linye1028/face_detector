#date:20181122
#检测当前目录下的jpg和JPG文件中的人脸，并保存到faces_separated文件夹中，该文件夹需要新建

import dlib         # 人脸识别的库dlib
import dlib         # 人脸识别的库dlib
import numpy as np  # 数据处理的库numpy
import cv2          # 图像处理的库OpenCv
import os

# 读取图像的路径，当前目录下
path_read = "./"
filename=[]
for fn in os.listdir(path_read):
	if fn[-4:]=='.jpg'  or fn[-4:]=='.JPG':
		filename.append(fn)
		print(fn)
# 用来存储生成的单张人脸的路径
path_save = "faces_separated/"

# Delete old images
def clear_images():
    imgs = os.listdir(path_save)

    for img in imgs:
        os.remove(path_save + img)

    print("clean finish", '\n')


clear_images()

# Dlib 正向人脸检测器
detector = dlib.get_frontal_face_detector()
fn_i=0 #人脸编号
for fn in filename :
	img = cv2.imread(path_read+fn)
	# Dlib 检测
	faces = detector(img, 1)
	
	print("人脸数：", len(faces), '\n')

	for k, d in enumerate(faces):
		
		# 计算矩形大小
		# (x,y), (宽度width, 高度height)
		pos_start = tuple([d.left(), d.top()])
		pos_end = tuple([d.right(), d.bottom()])

		# 计算矩形框大小
		height = d.bottom()-d.top()
		width = d.right()-d.left()

		# 根据人脸大小生成空的图像
		img_blank = np.zeros((height, width, 3), np.uint8)

		for i in range(height):
			for j in range(width):
					img_blank[i][j] = img[d.top()+i][d.left()+j]

		# cv2.imshow("face_"+str(k+1), img_blank)

		# 存在本地
		fn_i=fn_i+1
		print("Save to:", path_save+"img_face_"+str(fn_i)+".jpg")
		cv2.imwrite(path_save+"img_face_"+str(fn_i)+".jpg", img_blank)
