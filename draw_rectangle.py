import cv2
img = cv2.imread("/media/eap/FAA45FC7A45F84D3/Git/test/Image_test/coreless_battery00003754.jpg")
print(img.shape)
cv2.rectangle(img, (394, 567), (541, 654), (0, 0, 0), thickness=2)
cv2.rectangle(img, (1065, 1010), (1192, 1032), (0, 0, 0), thickness=2)
# cv2.rectangle(img, (559, 1016), (731, 1041), (128, 0, 0), thickness=2)
# cv2.rectangle(img, (109, 734), (304, 880), (128, 0, 0), thickness=2)
cv2.imwrite("test.jpg", img)