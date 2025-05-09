import cv2
import numpy as np

widthimg=700
heightimg=700
questions=5
choices=5
ans = [1,2,0,1,3]
webcamFeed=True
cameraNo=0

cap = cv2.VideoCapture(cameraNo)
cap.set(10,150)

while True:
    if webcamFeed: success,img=cap.read()
    else: img = cv2.imread("IMG-1555.jpg")

    img1=cv2.resize(img,(widthimg,heightimg))
    imgcontours=img1.copy()
    imgBiggestContours=img1.copy()
    imggray=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    imgblur=cv2.GaussianBlur(imggray,(5,5),1)
    imgcanny=cv2.Canny(imgblur,10,50)

    def stackImages(imgArray, scale, labels=[]):
        max_width = max(img.shape[1] for row in imgArray for img in row)
        max_height = max(img.shape[0] for row in imgArray for img in row)

        rows = len(imgArray)
        cols = len(imgArray[0])
        rowsAvailable = isinstance(imgArray[0], list)

        if rowsAvailable:
            for x in range(0, rows):
                for y in range(0, cols):
                    if len(imgArray[x][y].shape) == 2:
                        imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (max_width, max_height), None, scale, scale)

            imageBlank = np.zeros((max_height, max_width, 3), np.uint8)
            hor = [imageBlank] * rows
            hor_con = [imageBlank] * rows
            for x in range(0, rows):
                hor[x] = np.hstack(imgArray[x])
                hor_con[x] = np.concatenate(imgArray[x])
            ver = np.vstack(hor)
            ver_con = np.concatenate(hor)
        else:
            for x in range(0, rows):
                if len(imgArray[x].shape) == 2:
                    imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
                imgArray[x] = cv2.resize(imgArray[x], (max_width, max_height), None, scale, scale)

            hor = np.hstack(imgArray)
            hor_con = np.concatenate(imgArray)
            ver = hor

        if len(labels) != 0:

          eachImgWidth = int(ver.shape[1] / cols)
          eachImgHeight = int(ver.shape[0] / rows)
          for d in range(0, rows):
            for c in range(0, cols):
               (text_width, text_height), _ = cv2.getTextSize(labels[d][c], cv2.FONT_HERSHEY_COMPLEX, 1, 1)
               rect_start = (c * eachImgWidth, eachImgHeight * d)
               rect_end = (c * eachImgWidth + text_width + 10, text_height + 10 + eachImgHeight * d)

               cv2.rectangle(
                    ver,
                    rect_start,
                    rect_end,
                    (255, 255, 255),
                    cv2.FILLED,
                )

               cv2.putText(
                    ver,
                    labels[d][c],
                    (eachImgWidth * c + 10, eachImgHeight * d + 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    1,
                    (255, 0, 255),
                    1,
                )
        return ver

    def rectCountour(coutours):
        rectCon= []
        for i in coutours:
            area= cv2.contourArea(i)
            if area>50:
              peri = cv2.arcLength(i,True)
              approx = cv2.approxPolyDP(i,0.02*peri,True)
              if len (approx)==4:
                rectCon.append(i)
        rectCon = sorted(rectCon,key=cv2.contourArea,reverse=True)
        return rectCon

    def getCornerPoints(cont):
        peri=cv2.arcLength(cont,True)
        approx=cv2.approxPolyDP(cont,0.02*peri,True)
        return approx.reshape(-1,2)

    def reorder(myPoints):
        myPoints = myPoints.reshape((4, 2))
        myPointsNew = np.zeros((4, 1, 2), np.int32)
        sums = myPoints.sum(axis=1)
        myPointsNew[0] = myPoints[np.argmin(sums)]
        myPointsNew[3] = myPoints[np.argmax(sums)]
        diffs = np.diff(myPoints, axis=1)
        myPointsNew[1] = myPoints[np.argmin(diffs)]
        myPointsNew[2] = myPoints[np.argmax(diffs)]
        return myPointsNew

    def splitBoxes(img):
        rows = np.vsplit(img, 5)
        boxes = []

        for r in rows:
            cols = np.hsplit(r, 5)
            for box in cols:
                boxes.append(box)

        return boxes

    def showAnswers(img, myIndex, grading, ans, questions, choices):
        secW = int(img.shape[1] / questions)
        secH = int(img.shape[0] / choices)
        for X in range(0, questions):
            myAns = myIndex[X]
            cX = (myAns * secW) + secW // 2
            cY = (X * secH) + secH // 2

            if grading[X] ==1:
              myColor = (0,255,0)
            else:
              myColor=(0,0,255)
              correctAns = ans[X]
              cv2.circle(img, ((correctAns*secW)+secW//2,(X*secH)+secH//2), 20, (0,255,0), cv2.FILLED)

            cv2.circle(img, (cX, cY), 50, myColor, cv2.FILLED)
        return img

    try:
        contours,heirarchy=cv2.findContours(imgcanny,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(imgcontours,contours,-1,(0,255,0),10)

        rectCon=rectCountour(contours)
        biggestContour = getCornerPoints(rectCon[0])
        gradePoints= getCornerPoints(rectCon[1])

        if biggestContour.size != 0 and gradePoints.size != 0:
            cv2.drawContours(imgBiggestContours, [biggestContour], -1, (0, 0, 255), 2)
            cv2.drawContours(imgBiggestContours, [gradePoints], -1, (255, 0, 0), 2)

            biggestContour=reorder(biggestContour)
            gradePoints=reorder(gradePoints)

            pt1 = np.float32(biggestContour)
            pt2 = np.float32([[0,0],[widthimg,0],[0,heightimg],[widthimg,heightimg]])
            matrix = cv2.getPerspectiveTransform(pt1,pt2)
            imgWarpColored = cv2.warpPerspective(imgBiggestContours,matrix,(widthimg,heightimg))

            ptG1 = np.float32(gradePoints)
            ptG2 = np.float32([[0, 0], [325, 0], [0, 150], [325, 150]])
            matrixG = cv2.getPerspectiveTransform(ptG1, ptG2)
            imgGradeDisplay = cv2.warpPerspective(imgBiggestContours, matrixG, (325, 150))

            imgWarpGray = cv2.cvtColor(imgWarpColored, cv2.COLOR_BGR2GRAY)
            imgThresh = cv2.threshold(imgWarpGray, 150, 255, cv2.THRESH_BINARY_INV)[1]

            boxes = splitBoxes(imgThresh)

            myPixelVal = np.zeros((questions, choices))
            countC = 0
            countR = 0

            for image in boxes:
                totalPixels = cv2.countNonZero(image)
                myPixelVal[countR][countC] = totalPixels
                countC += 1
                if (countC == choices): countR += 1; countC = 0
            print(myPixelVal)

            myIndex = []
            for x in range(0, questions):
                arr = myPixelVal[x]
                myIndexVal = np.where(arr == np.amax(arr))
                myIndex.append(myIndexVal[0][0])
            print(myIndex)

            grading = []
            for x in range(0, questions):
                if ans[x] == myIndex[x]:
                    grading.append(1)
                else:
                    grading.append(0)
            print(grading)

            score = sum(grading) / questions * 100

            imgResult = imgWarpColored.copy()
            imgResult = showAnswers(imgResult, myIndex, grading, ans, questions, choices)
            imRawDrawing = np.zeros_like(imgWarpColored)
            imRawDrawing = showAnswers(imRawDrawing, myIndex, grading, ans, questions, choices)
            invMatrix = cv2.getPerspectiveTransform(pt2, pt1)
            imgInvWarp = cv2.warpPerspective(imRawDrawing, invMatrix, (widthimg, heightimg))
            imgRawGrade = np.zeros_like(imgGradeDisplay)
            cv2.putText(imgRawGrade, str(int(score)) + '%', (70, 105), cv2.FONT_HERSHEY_COMPLEX, 3, (0, 255, 255), 3)
            invMatrixG = cv2.getPerspectiveTransform(ptG2, ptG1)
            imgInvGradeDisplay = cv2.warpPerspective(imgRawGrade, invMatrixG, (widthimg, heightimg))

            imgFinal = img1.copy()
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvWarp, 1, 0)
            imgFinal = cv2.addWeighted(imgFinal, 1, imgInvGradeDisplay, 1, 0)

        imgBlank=np.zeros_like(img1)
        imageArray=([[img1,imgFinal,imgblur,imgcanny],[imgcontours,imgBiggestContours,imgWarpColored,imgThresh],[imgResult,imRawDrawing,imgInvWarp,imggray]])

    except:
        imgBlank = np.zeros_like(img1)
        imageArray = ([[img1, imggray, imgblur, imgcanny], [imgBlank, imgBlank, imgBlank, imgBlank],
                       [imgBlank, imgBlank, imgBlank, imgBlank]])
    labels = [['Original','Final','Blur','Canny'],['Contours','Biggest Contours','Warp','Threshold'],['Result','Raw Drawing','Inv Warp','gray']]
    imgStacked=stackImages(imageArray,0.4,labels)

    cv2.imshow("Stacked Images",imgStacked)
    if cv2.waitKey(1) & 0xFF == ord('s'):
        cv2.imwrite('FinalResult.jpg',imgFinal)
        cv2.waitKey(300)