import cv2
import os
import glob

###rename files###
def main():
    i = 0

    for filename in os.listdir("images_set/inst/"):
        dst = str(i) + ".jpg"
        src ='images_set/inst/'+ filename
        dst ='images_set/inst/'+ dst

        # rename() function will
        # rename all the files
        os.rename(src, dst)
        i += 1

# Driver Code
if __name__ == '__main__':

    # Calling main() function
    main()

# Get user supplied values
imagePath = "images_set/multi_person/"
#images = [cv2.imread(file) for file in glob.glob(imagePath+'*.[pj][np][gg]*')]
cascPath = "haarcascade_frontalface_default.xml"
scaleFactor = 1.1
minNeighbors=5
minSize=(30, 30)


#imagePath = [cv2.imread(file) for file in glob.glob('images_set/multi_person/*.[pj][np][gg]*')]
def face_dection(imagePath, cascPath, scaleFactor, minNeighbors, minSize):
    detection_result={}
    images = [cv2.imread(file) for file in glob.glob(imagePath+'*.[pj][np][gg]*')]
    #print(len(images))
    for num,i in enumerate(images):
    # Create the haar cascade
        faceCascade = cv2.CascadeClassifier(cascPath)
        # Read the image
        #image = cv2.imread(i)
        gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
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
            cv2.rectangle(i, (x, y), (x+w, y+h), (0, 255, 0), 2)


        detection_result[num]=len(faces)
        print("Found {0} faces!".format(len(faces)))

        cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        imS = cv2.resize(i, (1280, 720))                    # Resize image
        cv2.imshow("output", imS)                            # Show image
        cv2.waitKey(0)

    print(detection_result)

face_dection(imagePath, cascPath, scaleFactor, minNeighbors, minSize)

###output only result without picture###
# Get user supplied values
imagePath = "images_set/inst/"
#images = [cv2.imread(file) for file in glob.glob(imagePath+'*.[pj][np][gg]*')]
#cascPath = "haarcascade_frontalface_default.xml"
#cascPath = "data/haarcascades/haarcascade_fullbody.xml"
#cascPath = "data/haarcascades/haarcascade_frontalcatface.xml"
#cascPath = "data/haarcascades/haarcascade_frontalface_alt.xml"
#cascPath = "data/haarcascades/haarcascade_frontalface_alt2.xml"
#cascPath = "data/haarcascades/haarcascade_lowerbody.xml"
#cascPath = "data/haarcascades/haarcascade_upperbody.xml"
#cascPath = "data/haarcascades/haarcascade_eye.xml"
#cascPath = "data/lbpcascades/lbpcascade_frontalface_improved.xml" #try tune this one more
cascPath = "data/lbpcascades/lbpcascade_frontalface.xml"
#cascPath = "data/lbpcascades/lbpcascade_frontalcatface.xml"
#cascPath = "data/lbpcascades/lbpcascade_profileface.xml"
#with lbpcascade_frontalface.xml, s:1.044, min:3 size:72,72
#cascPath = "data/haarcascades/haarcascade_eye.xml"
#cascPath = "data/haarcascades/haarcascade_frontalcatface_extended.xml"
#cascPath = "data/haarcascades/haarcascade_mcs_upperbody.xml"

scaleFactor = 1.05
minNeighbors=3
minSize=(75, 75)

def face_result(imagePath, cascPath, scaleFactor, minNeighbors, minSize):
    detection_result={}
    images = [cv2.imread(file) for file in glob.glob(imagePath+'*.[pj][np][gg]*')[:5]]
    for num, i in enumerate(images):
    # Create the haar cascadex`
        faceCascade = cv2.CascadeClassifier(cascPath)
        # Read the image
        #image = cv2.imread(i)
        gray = cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)

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
            cv2.rectangle(i, (x, y), (x+w, y+h), (0, 255, 0), 2)

        detection_result[num]=len(faces)
        cv2.namedWindow("output", cv2.WINDOW_NORMAL)        # Create window with freedom of dimensions
        imS = cv2.resize(i, (1280, 720))                    # Resize image
        cv2.imshow("output", imS)                            # Show image
        cv2.waitKey(0)
    return(detection_result)

res = face_result(imagePath, cascPath, scaleFactor, minNeighbors, minSize)
res
sum(res.values())
