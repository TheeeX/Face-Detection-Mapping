#include "opencv2/opencv.hpp"
#include<opencv/highgui.h>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include<opencv2/objdetect/objdetect.hpp>

using namespace cv;
using namespace std;

/** Function Headers */
void detectAndDisplay(Mat frame);

/** Global variables */
String face_cascade_name = "C:\\Ceemple\\OpenCV4VS\\samples\\data\\haarcascades\\haarcascade_frontalface_alt.xml";
String eyes_cascade_name = "C:\\Ceemple\\OpenCV4VS\\samples\\data\\haarcascades\\haarcascade_eye_tree_eyeglasses.xml";
CascadeClassifier face_cascade;
CascadeClassifier eyes_cascade;
string window_name = "Capture - Face detection";
RNG rng(12345);

/** @function main */
int main(int argc, const char** argv)
{
	VideoCapture cap;
	cap.open(0);
	Mat frame;

	if (!face_cascade.load(face_cascade_name)){ printf("--(!)Error loading\n"); return -2; };
	if (!eyes_cascade.load(eyes_cascade_name)){ printf("--(!)Error loading\n"); return -1; };

	while (1)
	{
		cap >> frame;

		detectAndDisplay(frame);

		int c = waitKey(10);
		if ((char)c == 'c') { break; }
	}

	return 0;
}

void detectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;

	cvtColor(frame, frame_gray, CV_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	//-- Detect faces
	face_cascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
	/*
	cout << "No. of faces :" << faces.size() << endl << "\n";
	for (size_t i = 0; i < faces.size(); i++){
	cout << "\tfaces[i].x : " << (faces[i].x) << endl;
	cout << "\tfaces[i].y : " << (faces[i].y) << endl;
	cout << "\tfaces[i].width : " << (faces[i].width) << endl;
	cout << "\tfaces[i].height : " << (faces[i].height) << endl;
	cout << "\tX : " << (faces[i].x + faces[i].width*0.5) << endl;
	cout << "\tY : " << (faces[i].y + faces[i].height*0.5) << endl;
	cout << "\n";
	}
	*/

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width*0.5, faces[i].y + faces[i].height*0.5);
		/*-- Draws ellipse ellipse(Mat& img, Point center, Size axes, double angle, double startAngle, double endAngle, const Scalar& color, int thickness=1, int lineType=8, int shift=0)--*/
		ellipse(frame, center, Size(faces[i].width*0.5, faces[i].height*0.5), 0, 0, 360, Scalar(255, 0, 255), 4, 8, 0);

		Mat faceROI = frame_gray(faces[i]);
		std::vector<Rect> eyes;

		//-- In each face, detect eyes
		eyes_cascade.detectMultiScale(faceROI, eyes, 1.1, 2, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));

		//imshow("hulala", faceROI);

		for (size_t j = 0; j < eyes.size(); j++)
		{
			Point center(faces[i].x + eyes[j].x + eyes[j].width*0.5, faces[i].y + eyes[j].y + eyes[j].height*0.5);
			Point p1(faces[i].x + eyes[j].x, faces[i].y + eyes[j].y);
			Point p2(faces[i].x + eyes[j].x + eyes[j].width, faces[i].y + eyes[j].y + eyes[j].height);
			int radius = cvRound((eyes[j].width + eyes[j].height)*0.25);
			/*-- Draws Circle circle(Mat& img, Point center, int radius, const Scalar& color, int thickness=1, int lineType=8, int shift=0) --*/
			circle(frame, center, radius, Scalar(255, 0, 0), 1, 8, 0);
			rectangle(frame, p1, p2, Scalar(255, 0, 0), 1, 8, 0);

		}
	}
	//-- Show what you got
	imshow(window_name, frame);
}