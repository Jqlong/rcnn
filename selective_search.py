import sys
import cv2
import matplotlib.pyplot as plt

def get_selective_search(img, strategy='q'):
    gs = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()  # 使用opencv初始化选择性搜索分割对象gs
    gs.setBaseImage(img)

    if strategy == 's':
        gs.switchToSingleStrategy()
    elif strategy == 'f':
        gs.switchToSelectiveSearchFast()
    elif strategy == 'q':
        gs.switchToSelectiveSearchQuality()
    else:
        print(__doc__)
        sys.exit(1)

    rects = gs.process()
    rects[:, 2] += rects[:, 0]
    rects[:, 3] += rects[:, 1]

    return rects

filepath = 'C:\\Users\\22357\\Pictures\\header.jpg'
img = cv2.imread(filepath, cv2.IMREAD_COLOR)
rects = get_selective_search(img, strategy='s')
for rect in rects:
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 255, 0), 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
cv2.imwrite('header.jpg', img)
plt.show()
