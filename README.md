1. Clone the Repository:
   git clone https://github.com/Dharshini-V26/Hand-Sign-Language-detection.git
   cd Sign-Language-detection

2. Create a Virtual Environment:
   python -m venv venv
   venv\Scripts\activate  # On Windows

3. Install Required Packages:
   pip install -r requirements.txt
   
   Eg:
   pip install opencv-python
   pip install cvzone
   pip install numpy
   pip install mediapipe
   pip install tensorflow

4. Run Scripts:
   python datacollection.py

   The webcam will open.
   Show your hand sign gesture.
   Press "S" to save the current frame/image for that hand gesture.

5. Save and Download Trained Model:
   After training:
   Download the "keras_model.h5" model file and "labels.txt" file as a .zip.
   Extract the .zip file.
   Place both files inside your project directory under Model/ folder.
   
6. Test Real-time Detection:
   python test.py

   The webcam will open and detect the hand signs in real time using the trained model.
   
