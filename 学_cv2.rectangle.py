import cv2
img = cv2.imread(r"E:\Picture_bin\2022_4\10re.png")
print(img.shape)  # 图片大小__(750, 1200, 3)
cv2.rectangle(img, (240, 0), (480, 375), 255, -1)
cv2.imshow("fff", img)

cv2.waitKey(0)
cv2.destroyAllWindows()
