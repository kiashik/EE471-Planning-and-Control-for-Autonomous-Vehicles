# (c) 2026 S. Farzan, Electrical Engineering Department, Cal Poly
# EE 471 (SP26): Planning and Control for Autonomous Vehicles
"""
lab04_ee471.py

Script for EE 471 Lab 4:
Camera calibration along with line detection for QCar.
Please review the Lab 4 PDF document on Canvas.
"""
# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : File Description and Imports
from pal.products.qcar import QCarCameras, QCarRealSense, IS_PHYSICAL_QCAR
import time
import numpy as np
import cv2
#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : ImageInterpretation Class Setup

class ImageInterpretation():

    def __init__(self,
            imageSize,
            frameRate,
            streamInfo,
            gridDims,
            boxSize):

        # Camera calibration constants:
        self.NUMBER_IMAGES = 15

        if not IS_PHYSICAL_QCAR:
            self.NUMBER_IMAGES = 5

        # List of variables given by students
        self.imageSize      = imageSize
        self.chessboardDim  = [gridDims[0], gridDims[1]]
        self.frameRate      = frameRate
        self.boxSize        = boxSize
        self.sampleRate     = 1/self.frameRate
        self.calibFinished  = False

        # List of camera intrinsic properties:
        self.CSICamIntrinsics = np.eye(3, 3, dtype=np.float32)
        # CSI camera intrinsic matrix at resolution [820, 410] is approximately:
        # [[318.86    0.00  401.34]
        #  [  0.00  312.14  201.50]
        #  [  0.00    0.00    1.00]]
        self.CSIDistParam = np.ones((1, 5), dtype=np.float32)
        # CSI camera distortion parameters at resolution [820, 410] are approximately:
        # [[-0.9033  1.5314 -0.0173 0.0080 -1.1659]]

        self.d435CamIntrinsics = np.eye(3, 3, dtype=np.float32)
        # D435 RGB camera intrinsic matrix at resolution [640, 480] is approximately:
        # [[455.20    0.00  308.53]
        #  [  0.00  459.43  213.56]
        #  [  0.00    0.00    1.00]]
        self.d435DistParam = np.ones((1, 5), dtype=np.float32)
        # D435 RGB camera distortion parameters at resolution [640, 480] are approximately:
        # [[-5.1135e-01  5.4549 -2.2593e-02 -6.2131e-03 -2.0190e+01]]

        # Final image streamed by CSI or D435 camera
        self.streamD435 = np.zeros((self.imageSize[1][0], self.imageSize[1][1]))
        self.streamCSI  = np.zeros((self.imageSize[0][0], self.imageSize[0][1]))

        # Information for interfacing with front CSI camera
        enableCameras = [False, False, False, False]
        enableCameras[streamInfo[0]] = True

        self.frontCSI = QCarCameras(
            frameWidth  = self.imageSize[0][0],
            frameHeight = self.imageSize[0][1],
            frameRate   = self.frameRate[0],
            enableRight = enableCameras[0],
            enableBack  = enableCameras[1],
            enableLeft  = enableCameras[2],
            enableFront = enableCameras[3]
        )

        # Information for interfacing with the RealSense camera
        self.d435Color = QCarRealSense(
            mode           = streamInfo[1],
            frameWidthRGB  = self.imageSize[1][0],
            frameHeightRGB = self.imageSize[1][1],
            frameRateRGB   = self.frameRate[1]
        )

        self.SimulationTime = 15

    def _calibrate_from_images(self, savedImages):
        """Run camera calibration on a list of grayscale chessboard images.
        Returns (K, dist, rms_reprojection_error). Students complete the
        three TODO blocks below."""

        # 3D object points for the internal-corner grid (planar target, z = 0).
        nx, ny = self.chessboardDim[0], self.chessboardDim[1]
        objp = np.zeros((nx * ny, 3), np.float32)
        objp[:, :2] = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2) * self.boxSize

        # Subpixel-refinement criteria for cv2.cornerSubPix.
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)

        objPoints = []   # 3D points in real-world space
        imgPoints = []   # 2D points in the image plane

        # Detect and refine chessboard corners in every image.
        lastImage = None
        dummy_i=0
        for img in savedImages:
            # ----- TODO (Part A3, step 1) -----
            # Detect the chessboard internal corners in `img` using cv2.
            # The pattern size is (nx, ny). Assign:
            #   found    -> bool indicating success
            #   corners  -> the detected corner array
            found, corners = cv2.findChessboardCorners(img, (nx,ny), None)
            cv2.imwrite(f"real_cal_image{dummy_i}.jpg",img)
            dummy_i+=1
            

            if found:
                # ----- TODO (Part A3, step 2) -----
                # Refine `corners` to subpixel accuracy using cv2.cornerSubPix
                # with a search window of (11, 11), no zero-zone (-1, -1),
                # and the `criteria` tuple defined above. Reassign the result
                # back into `corners`.
                corners = cv2.cornerSubPix(img, corners, (11, 11), (-1, -1), criteria)
                objPoints.append(objp)
                imgPoints.append(corners)
                lastImage = img

        if not objPoints:
            raise RuntimeError(
                "No chessboard corners detected. Check gridDims (internal "
                "corners, not squares) and image quality."
            )

        # Solve for the intrinsic matrix and distortion coefficients.
        # Image size for cv2.calibrateCamera is (width, height).
        imageSize = lastImage.shape[::-1]

        # ----- TODO (Part A3, step 3) -----
        # Call cv2.calibrateCamera with (objPoints, imgPoints, imageSize, ...)
        # to estimate the intrinsic matrix K and distortion coefficients
        # `dist`. Store the RMS reprojection error in `rms`.
        rms = 0.0
        K    = np.eye(3, dtype=np.float32)
        dist = np.zeros((1, 5), dtype=np.float32)

        rms, K, dist, _, _ = cv2.calibrateCamera(objPoints, imgPoints, imageSize, None, None)

        return K, dist, rms

    def camera_calibration(self):

        savedImages = []
        imageCount  = 0
        cameraType  = "csi"

        while True:
            startTime = time.time()

            # Read RGB information for front CSI first, then D435 RGB.
            if cameraType == "csi":
                self.frontCSI.readAll()
                endTime = time.time()
                image   = self.frontCSI.csiFront.imageData
                computationTime = endTime - startTime
                sleepTime = self.sampleRate[0] \
                    - (computationTime % self.sampleRate[0])

            if cameraType == "D435":
                self.d435Color.read_RGB()
                endTime = time.time()
                image   = self.d435Color.imageBufferRGB
                computationTime = endTime - startTime
                sleepTime = self.sampleRate[1] \
                    - (computationTime % self.sampleRate[1])

            # Use cv2 to display the current image
            cv2.imshow("Camera Feed", image)

            msSleepTime = int(1000 * sleepTime)
            if msSleepTime <= 0:
                msSleepTime = 1
            if cv2.waitKey(msSleepTime) & 0xFF == ord('q'):
                imageCount += 1
                print("saving Image #: ", imageCount)
                grayImage = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                savedImages.append(grayImage)

                if imageCount == self.NUMBER_IMAGES and cameraType == "csi":
                    print("Implement calibration for CSI camera images: ")

                    # ============== Part A3 - CSI Camera Parameter Estimation ==============
                    print("Camera calibration for front csi")
                    K, dist, rms = self._calibrate_from_images(savedImages)
                    self.CSICamIntrinsics = K
                    self.CSIDistParam     = dist
                    print("CSI calibration RMS reprojection error:", rms)

                    # Printed output for students
                    text = "CSI camera intrinsic matrix at resolution {} is:"
                    print(text.format(self.imageSize[0][:]))
                    print(self.CSICamIntrinsics)

                    text = ("CSI camera distortion parameters "
                        + "at resolution {} are: ")
                    print(text.format(self.imageSize[0][:]))
                    print(self.CSIDistParam)

                    cameraType  = "D435"
                    savedImages = []
                    imageCount  = 0

                if imageCount == self.NUMBER_IMAGES and cameraType == "D435":
                    print("Implement calibration for "
                        + "realsense D435 camera images:")

                    # ============== Part A3 - D435 Camera Parameter Estimation ==============
                    print("Camera calibration for D435 RGB camera")
                    K, dist, rms = self._calibrate_from_images(savedImages)
                    self.d435CamIntrinsics = K
                    self.d435DistParam     = dist
                    print("D435 calibration RMS reprojection error:", rms)

                    # Printed output for students
                    text = ("D435 RGB camera intrinsic matrix "
                        + "at resolution {} is:")
                    print(text.format(self.imageSize[1][:]))
                    print(self.d435CamIntrinsics)

                    text = ("D435 RGB camera distortion parameters "
                        + "at resolution {} are: ")
                    print(text.format(self.imageSize[1][:]))
                    print(self.d435DistParam)

                    # Quick visual sanity check: undistort each saved D435
                    print("Press any key in the image window to advance to the next "
                        "rectified image (or 'q' to skip the rest).")
                    for idx, distImg in enumerate(savedImages):
                        undist = cv2.undistort(
                            distImg,
                            self.d435CamIntrinsics,
                            self.d435DistParam
                        )
                        # cv2.imwrite(f"real_RectifiedImages{idx}.jpg", undist)
                        cv2.imshow("RectifiedImages", undist)
                        print(f"  showing rectified image {idx+1}/{len(savedImages)}")
                        key = cv2.waitKey(0) & 0xFF   # wait indefinitely for a key
                        if key == ord('q'):
                            break
                    break

        print("Both Cameras calibrated!")

        self.calibFinished = True
        cv2.destroyAllWindows()

    def line_detection(self, cameraType):

        currentTime = 0
        t0 = time.time()

        while currentTime < self.SimulationTime:
            LoopStartTime = time.time()
            currentTime   = time.time() - t0

            # Choose which stream is used for line detection.
            if cameraType == "csi":
                self.frontCSI.readAll()
                endTime = time.time()
                image   = self.frontCSI.csiFront.imageData
                computationTime = endTime - LoopStartTime
                sleepTime = self.sampleRate[0] \
                    - (computationTime % self.sampleRate[0])
                cameraIntrinsics = self.CSICamIntrinsics
                cameraDistortion = self.CSIDistParam

            if cameraType == "D435":
                self.d435Color.read_RGB()
                endTime = time.time()
                image   = self.d435Color.imageBufferRGB
                computationTime = endTime - LoopStartTime
                sleepTime = self.sampleRate[1] \
                    - (computationTime % self.sampleRate[1])
                cameraIntrinsics = self.d435CamIntrinsics
                cameraDistortion = self.d435DistParam

            # ============== Part B2 - Image Correction ====================
            print("Implement image correction for raw camera image... ")
            # YOUR CODE HERE
            # Use cv2.undistort with cameraIntrinsics and cameraDistortion
            # to remove lens distortion. Store the result in undistortedImage.
            cv2.imwrite("Braw.jpg",image)
            undistortedImage = cv2.undistort(image, cameraIntrinsics, cameraDistortion)
            cv2.imwrite("Bundistored.jpg",undistortedImage)

            # ============== Part B3 - Image Filtering ====================
            print("Implement image filter on distortion corrected image... ")
            # YOUR CODE HERE
            # Build a small filtering pipeline that emphasizes line structure:
            #   - Convert the image to grayscale with cv2.cvtColor.
            #   - Smooth with cv2.GaussianBlur to reduce sensor noise.
            #   - Detect edges with cv2.Canny.
            # Tune the blur kernel size and the Canny low/high thresholds
            # for your scene. Store the binary edge map in filteredImage.
            gray = cv2.cvtColor(undistortedImage, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(gray, (5,5), 0)
            edges = cv2.Canny(gaussian, 50, 150)
            # thresh = cv2.threshold(edges, 100, 255, cv2.THRESH_BINARY)
            filteredImage = edges
            cv2.imwrite("Bfiltered.jpg",filteredImage)

            # ============= Part B4 - Feature Extraction ===================
            print("Extract line information from filtered image... ")
            # YOUR CODE HERE
            # Use cv2.HoughLinesP on filteredImage to extract line segments.
            # For each segment, draw it onto a copy of undistortedImage
            # using cv2.line. Tune rho, theta, threshold, minLineLength,
            # and maxLineGap for the scene. Store the annotated image in
            # linesImage and the raw line data in lines.
            linesImage = np.copy(undistortedImage)
            houghlines = cv2.HoughLinesP(filteredImage, 1.0, np.pi/180, 30, minLineLength=25, maxLineGap=10)
            print(houghlines)
            if houghlines is not None:
                for line in houghlines:
                    x1, y1, x2, y2 = line[0]
                    cv2.line(linesImage, (x1,y1), (x2,y2), (0,255,0), 2)

            print("Display image with lines found... ")
            imageDisplayed = linesImage

            # Use cv2 to display the current image
            cv2.imwrite("LinesImage.jpg", imageDisplayed)
            cv2.imshow("Lines Image", imageDisplayed)
            msSleepTime = int(1000 * sleepTime)
            if msSleepTime <= 0:
                msSleepTime = 1

            cv2.waitKey(msSleepTime)

    def stop_cameras(self):
        # Stop the image feed for both cameras.
        self.frontCSI.terminate()
        self.d435Color.terminate()

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Main
def main():
    try:
        '''
        INPUTS:
        imageSize           = [[L,W],[L,W]] 2x2 array specifying the
                                resolution for the CSI camera and the D435.
        frameRate           = [CSIframeRate, D435frameRate] 2x1 array of
                                image frame rates for the CSI and D435.
        streamInfo          = [CSIIndex, D435Stream] 2x1 array specifying
                                the CSI camera index and the D435 stream type.
        gridDims            = Number of internal corners along each side of
                                the chessboard used for calibration.
        boxSize             = Float value specifying the size of one cell
                                in the chessboard, in meters.
        '''

        # ======== Part A2/B1 - Student Inputs for Image Interpretation ===========
        cameraInterfacingLab = ImageInterpretation(
            imageSize  = [[820, 410], [640, 480]],
            frameRate  = np.array([30, 30]),
            streamInfo = [3, "RGB"],
            gridDims   = (6, 6),
            boxSize    = 1.25
        )

        ''' TODO: Select the specific activity for the lab.
        List of current activities:
        - Calibrate
        - Line Detect
        '''
        camMode = "Line Detect"

        # ========== Part B1 - Camera Intrinsics and Distortion Coefficients ==========
        # TODO: Use your Part A calibration results for the camera you select below.
        if not IS_PHYSICAL_QCAR:    # CSI CAM MATRIX
            cameraMatrix = np.array([[264.32954759,   0.0,         395.45757263],
                [  0.0,         236.01661515, 161.52664939],
                [  0.0,           0.0,           1.0        ]])
            
            
            distortionCoefficients = np.array([-0.39224848,  0.15605065, -0.04902927, -0.02187179, -0.03189701])
            
        else:
            cameraMatrix = np.array([[707.31703579,   0.0,         391.84000212],
            [  0.0,         408.13345552, 232.91585798],
            [  0.0,           0.0,           1.0        ]])

            distortionCoefficients = np.array([ -2.59274561,  14.25220831,  -0.03946918,   0.02746133, -26.34466779])



        if camMode == "Calibrate":
            try:
                cameraInterfacingLab.camera_calibration()
                if cameraInterfacingLab.calibFinished == True \
                        and camMode == "Calibrate":
                    print("calibration process done, stopping cameras...")
                    cameraInterfacingLab.stop_cameras()

            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()

        if camMode == "Line Detect":
            try:
                text = "Specify the camera used for line detection (csi/D435): "
                cameraType = input(text)
                if cameraType == "csi":
                    cameraInterfacingLab.CSICamIntrinsics = cameraMatrix
                    cameraInterfacingLab.CSIDistParam     = distortionCoefficients
                    cameraInterfacingLab.line_detection(cameraType)

                elif cameraType == "D435":
                    cameraInterfacingLab.d435CamIntrinsics = cameraMatrix
                    cameraInterfacingLab.d435DistParam     = distortionCoefficients
                    cameraInterfacingLab.line_detection(cameraType)
                else:
                    print("Invalid camera type")

            except KeyboardInterrupt:
                cameraInterfacingLab.stop_cameras()
    finally:
        if not IS_PHYSICAL_QCAR:
            import qlabs_setup
            qlabs_setup.terminate()
        input('Experiment complete. Press any key to exit...')

#endregion

# -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- -- --

#region : Run
if __name__ == '__main__':
    main()
#endregion
