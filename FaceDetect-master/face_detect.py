import cv2

# Get user supplied values
imagePath = ['beatles.png', "abba.png",
"instagram.png", "finery.png"]
cascPath = "haarcascade_frontalface_default.xml"
scaleFactor = 1.1
minNeighbors=5
minSize=(30, 30)


def face_dection(imagePath, cascPath, scaleFactor, minNeighbors, minSize):
    detection_result={}
    for i in imagePath:
    # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)
        # Read the image
        image = cv2.imread(i)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        #print("images",image)
        #print("colorconversion",gray)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags = cv2.CASCADE_SCALE_IMAGE
        )
        #print("faces",faces)
        #print("image",image)

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)


        detection_result[i]=len(faces)
        print("Found {0} faces!".format(len(faces)))

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        imS = cv2.resize(image, (1280, 720))                    # Resize image
        cv2.imshow("output", imS)                            # Show image
        cv2.waitKey(0)

    print(detection_result)

face_dection(imagePath, cascPath, scaleFactor, minNeighbors, minSize)

###output only result without picture###
def face_result(imagePath, cascPath, scaleFactor, minNeighbors, minSize):
    detection_result={}
    for i in imagePath:
    # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)
        # Read the image
        image = cv2.imread(i)
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

        # Detect faces in the image
        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=scaleFactor,
            minNeighbors=minNeighbors,
            minSize=minSize,
            flags = cv2.CASCADE_SCALE_IMAGE
        )

        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

        detection_result[i]=len(faces)
    return(detection_result)

face_result(imagePath, cascPath, scaleFactor, minNeighbors, minSize)
