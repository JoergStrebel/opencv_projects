#include <opencv2/opencv.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video.hpp>

using namespace cv;
using namespace std;

int main(int argc, char* argv[])
{
 //Open the default video camera
 VideoCapture cap;

 cap.open(0, CAP_V4L);

 // if not success, exit program
 if (cap.isOpened() == false)  
 {
  cout << "Cannot open the video camera" << endl;
  cin.get(); //wait for any key press
  return -1;
 } 

// Available resolutions
// 640x480
// 352x288
// 320x240
// 176x144
// 160x120
// 1280x720
// 1280x960
//Data transfer speed is fixed at ca. 15 MB/s -  see usbtop

 double dWidth = 640; //get the width of frames of the video
 double dHeight = 480; //get the height of frames of the video
 
 cap.set(CAP_PROP_FRAME_WIDTH, dWidth); //get the width of frames of the video
 cap.set(CAP_PROP_FRAME_HEIGHT, dHeight); //get the height of frames of the video

 cout << "Resolution of the video : " << dWidth << " x " << dHeight << endl;
 
 // Create some random colors
 vector<Scalar> colors;
 RNG rng;
 for(int i = 0; i < 100; i++)
 {
    int r = rng.uniform(0, 256);
    int g = rng.uniform(0, 256);
    int b = rng.uniform(0, 256);
    colors.push_back(Scalar(r,g,b));
 }
 Mat old_frame, old_gray;
 vector<Point2f> p0, p1;
 
 // Take first frame and find corners in it
 cap >> old_frame;
 cvtColor(old_frame, old_gray, COLOR_BGR2GRAY);
 goodFeaturesToTrack(old_gray, p0, 100, 0.3, 7, Mat(), 7, false, 0.04);
 
 // Create a mask image for drawing purposes
 Mat mask = Mat::zeros(old_frame.size(), old_frame.type());


 string window_name = "My Camera Feed";
 namedWindow(window_name); //create a window called "My Camera Feed"
 
 while (true)
 {  
   Mat frame, frame_gray;
   bool bSuccess = cap.read(frame); // read a new frame from video 

   //Breaking the while loop if the frames cannot be captured
   if (bSuccess == false) 
   {
     cout << "Video camera is disconnected" << endl;
     cin.get(); //Wait for any key press
     break;
   }
  
   cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
   // calculate optical flow
   vector<uchar> status;
   vector<float> err;
   TermCriteria criteria = TermCriteria((TermCriteria::COUNT) + (TermCriteria::EPS), 10, 0.03);
   calcOpticalFlowPyrLK(old_gray, frame_gray, p0, p1, status, err, Size(15,15), 2, criteria);
   
   vector<Point2f> good_new;
   for(uint i = 0; i < p0.size(); i++)
   {
    // Select good points
    if(status[i] == 1) {
      good_new.push_back(p1[i]);
      // draw the tracks
      line(mask,p1[i], p0[i], colors[i], 2);
      circle(frame, p1[i], 5, colors[i], -1);
     }
    }
    Mat img;
    add(frame, mask, img);
    
   //continuously show frame rate
   // Always shows 7.5 fps for GStreamer, no matter what the resolution
   // Always shows 30 fps for V4L, no matter what the resolution
   //cout<< cap.get(CAP_PROP_FPS) << " " <<  endl;
   //show the frame in the created window
   imshow(window_name, img);

   //wait for for 1 ms until any key is pressed.  
   //If the 'Esc' key is pressed, break the while loop.
   //If the any other key is pressed, continue the loop 
   //If any key is not pressed withing 1 ms, continue the loop 
   if (waitKey(1) == 27)
   {
     cout << "Esc key is pressed by user. Stoppig the video" << endl;
     break;
   }
   // Now update the previous frame and previous points
   old_gray = frame_gray.clone();
   p0 = good_new;
 }

 return 0;

}


