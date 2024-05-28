import numpy as np
import cv2
import os
import pickle

# Constants for image capture frequency, resize factor, and number of training images
FREQ_DIV = 5   
RESIZE_FACTOR = 4
NUM_TRAINING = 20

class TrainEigenFaces:
    """
    Class for training a face recognition model using Eigenfaces in OpenCV.
    """
    def __init__(self):
        cascPath = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        self.face_name = input("Name of the person face: ")
        self.path = os.path.join(self.face_dir, self.face_name)
        print(self.path)
        if not os.path.isdir(self.path):
            print("creating path")
            os.mkdir(self.path)
        self.count_captures = 0
        self.count_timer = 0

    def capture_training_images(self):
        """
        Capture training images from the webcam.
        """
        video_capture = cv2.VideoCapture(0)
        while True:
            self.count_timer += 1
            ret, frame = video_capture.read()
            inImg = np.array(frame)
            outImg = self.process_image(inImg)
            cv2.imshow('Video', outImg)

            # When everything is done, release the capture on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                return


    def process_image(self, inImg):
        """
        Process an image: flip it, convert to grascale, resize and detect faces.
        """
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (92, 112)        
        if self.count_captures < NUM_TRAINING:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            gray_resized = cv2.resize(gray, (int(gray.shape[0]/RESIZE_FACTOR),int(gray.shape[1]/RESIZE_FACTOR)))        
            faces = self.face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                )
            if len(faces) > 0:
                areas = []
                for (x, y, w, h) in faces: 
                    areas.append(w*h)
                max_area, idx = max([(val,idx) for idx,val in enumerate(areas)])
                face_sel = faces[idx]
            
                x = face_sel[0] * RESIZE_FACTOR
                y = face_sel[1] * RESIZE_FACTOR
                w = face_sel[2] * RESIZE_FACTOR
                h = face_sel[3] * RESIZE_FACTOR

                # Ensure the face region is within image boundaries
                y_start = max(0, y-h//2-30)
                y_end = min(gray.shape[0], y+h//2+30)
                x_start = max(0, x+80)
                x_end = min(gray.shape[1], x+w+80)

                face = gray[y_start:y_end, x_start:x_end]
                face_resized = cv2.resize(face, (resized_width, resized_height))
                img_no = sorted([int(fn[:fn.find('.')]) for fn in os.listdir(self.path) if fn[0]!='.' ]+[0])[-1] + 1
                
                if self.count_timer%FREQ_DIV == 0:
                    if face.size != 0:
                        cv2.imwrite('%s/%s.png' % (self.path, img_no), face_resized)
                        self.count_captures += 1
                        print("Captured image: ", self.count_captures)

                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
                cv2.putText(frame, self.face_name, (x - 10, y - 10), cv2.FONT_HERSHEY_PLAIN, 1,(0, 255, 0))
        elif self.count_captures == NUM_TRAINING:
            print("Training data captured. Press 'q' to exit.")
            self.count_captures += 1

        return frame
    
    def form_face_matrix(self, data, labels, tag, num_images = 20):
        """
        Form a matrix of face images for a given tag.
        
        Parameters:
        - data: Images data.
        - labels: Labels for the images.
        - tag: Tag identifying the person.
        - num_images: Number of images to consider.
        
        Returns:
        - face_images: Matrix of face images for the given tag.
        """
        face_indices = np.where(labels == tag)[0]
        face_images = data[face_indices].reshape(num_images, -1).T
        return face_images

    def compute_eigenfaces(self):
        """
        Compute eigenfaces for face recognition.
        
        Returns:
        - faces_svd_component: Singular value decomposition components of face matrices.
        """
        imgs = []
        tags = []
        index = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                img_path = os.path.join(self.face_dir, subdir)
                for fn in os.listdir(img_path):
                    path = img_path + '/' + fn
                    tag = index
                    imgs.append(cv2.imread(path, 0))
                    tags.append(int(tag))
                index += 1
        (imgs, tags) = [np.array(item) for item in [imgs, tags]]
        
        face_matrices = [self.form_face_matrix(imgs, tags, person) for person in range(len([name for name in os.listdir(self.face_dir) if os.path.isdir(os.path.join(self.face_dir, name))]))]
        faces_svd_component = [np.linalg.svd(matrix, full_matrices=False)[0] for matrix in face_matrices]
        with open('faces_svd', 'wb') as fp:
            pickle.dump(faces_svd_component, fp)
        return faces_svd_component
    

trainer = TrainEigenFaces()
trainer.capture_training_images()
trainer.compute_eigenfaces()
class RecogEigenFaces():
    """
    Class for recognizing faces using Eigenfaces.
    """
    def __init__(self):
        cascPath = "haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascPath)
        self.face_dir = 'face_data'
        self.face_names = []
        with open('faces_svd', 'rb') as fp:
            self.U = pickle.load(fp)
    
    def load_trained_data(self):
        """
        Load the names of individuals from the training dataset.
        """
        names = {}
        key = 0
        for (subdirs, dirs, files) in os.walk(self.face_dir):
            for subdir in dirs:
                names[key] = subdir
                key += 1
        self.names = names 

    def show_video(self):
        """
        Capture video stream from the webcam and recognize faces in real-time.
        """
        video_capture = cv2.VideoCapture(0)
        while True:
            ret, frame = video_capture.read()
            inImg = np.array(frame)
            outImg, self.face_names = self.process_image(inImg)
            cv2.imshow('Video', outImg)

            # When everything is done, release the capture on pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                video_capture.release()
                cv2.destroyAllWindows()
                return
            
    def classify_image(self, z, Us):
        """
        Classify a face image by finding the closest match using eigenfaces.
        
        Parameters:
        - z: Eigenface representation of the input face image.
        - Us: List of eigenface spaces.
        
        Returns:
        - label: Index of the closest match in the eigenfaces list.
        - min_distance: Minimum distance between the input image and the closest eigenface space.
        """
        min_distance = float("inf")
        label = None
        
        for n, U in enumerate(Us):
            projection = U @ (U.T @ z.reshape(-1))
            distance = np.linalg.norm(z.reshape(-1)-projection)
            
            if distance < min_distance:
                min_distance = distance
                label = n
        print(label, min_distance)
        return label, min_distance

    def process_image(self, inImg):
        """
        Detect faces in the input image, recognize them, and annotate the image accordingly.
        
        Parameters:
        - inImg: Input image frame from the video stream.
        
        Returns:
        - frame: Annotated image with recognized faces.
        - persons: List of names corresponding to the recognized faces.
        """
        frame = cv2.flip(inImg,1)
        resized_width, resized_height = (92, 112)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)        
        gray_resized = cv2.resize(gray, (int(gray.shape[0]/RESIZE_FACTOR),int(gray.shape[1]/RESIZE_FACTOR)))
        faces = self.face_cascade.detectMultiScale(
                gray_resized,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
                )
        persons = []
        for i in range(len(faces)):
            face_i = faces[i]
            x = face_i[0] * RESIZE_FACTOR
            y = face_i[1] * RESIZE_FACTOR
            w = face_i[2] * RESIZE_FACTOR
            h = face_i[3] * RESIZE_FACTOR
            
            # Ensure the face region is within image boundaries
            y_start = max(0, y-h//2-30)
            y_end = min(gray.shape[0], y+h//2+30)
            x_start = max(0, x+80)
            x_end = min(gray.shape[1], x+w+80)
                
            face = gray[y_start:y_end, x_start:x_end]
            face_resized = cv2.resize(face, (resized_width, resized_height))
            confidence = self.classify_image(face_resized, self.U)
            if confidence[1]<5000:
                person = self.names[confidence[0]]
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 3)
                cv2.putText(frame, '%s - %.0f' % (person, confidence[1]), (x+w-20, y+h-40), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            else:
                person = 'Unknown'
                cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 3)
                cv2.putText(frame, person, (x+w-20, y+h-40), cv2.FONT_HERSHEY_PLAIN,1,(0, 255, 0))
            persons.append(person)
        return (frame, persons)

recognizer = RecogEigenFaces()
recognizer.load_trained_data()
print("Press 'q' to quit video")
recognizer.show_video()
