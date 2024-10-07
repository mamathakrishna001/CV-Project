#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>

using namespace cv;
using namespace std;

void detectHarrisCorners(const Mat& img, Mat& harrisCorners, Mat& harrisScaled) {
    Mat harrisNormalized;
    harrisCorners = Mat::zeros(img.size(), CV_32FC1);

    // Detect Harris corners
    cornerHarris(img, harrisCorners, 2, 3, 0.04);

    // Normalize and convert to an 8-bit image
    normalize(harrisCorners, harrisNormalized, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    convertScaleAbs(harrisNormalized, harrisScaled);

    // Draw circles at detected corners
    for (int i = 0; i < harrisNormalized.rows; i++) {
        for (int j = 0; j < harrisNormalized.cols; j++) {
            if ((int)harrisNormalized.at<float>(i, j) > 200) {
                circle(harrisScaled, Point(j, i), 5, Scalar(255), 2);
            }
        }
    }
}

void detectSIFTFeatures(const Mat& img, vector<KeyPoint>& keypoints, Mat& descriptors) {
    Ptr<SIFT> sift = SIFT::create();
    sift->detectAndCompute(img, noArray(), keypoints, descriptors);
}

vector<DMatch> matchFeatures(const Mat& descriptors1, const Mat& descriptors2) {
    BFMatcher matcher(NORM_L2);
    vector<vector<DMatch>> knnMatches;
    vector<DMatch> goodMatches;

    matcher.knnMatch(descriptors1, descriptors2, knnMatches, 2);

    // Apply Lowe's ratio test to keep good matches
    for (size_t i = 0; i < knnMatches.size(); i++) {
        if (knnMatches[i][0].distance < 0.75 * knnMatches[i][1].distance) {
            goodMatches.push_back(knnMatches[i][0]);
        }
    }

    return goodMatches;
}

int main() {
    // Load the first image
    Mat img1 = imread("img1.png", IMREAD_GRAYSCALE);
    if (img1.empty()) {
        cerr << "Error: Could not load image img1.png" << endl;
        return -1;
    }

    // Harris corner detection
    Mat harrisCorners, harrisScaled;
    detectHarrisCorners(img1, harrisCorners, harrisScaled);

    // SIFT feature detection for the first image
    vector<KeyPoint> keypoints1;
    Mat descriptors1;
    detectSIFTFeatures(img1, keypoints1, descriptors1);

    // Load the second image
    Mat img2 = imread("img2.png", IMREAD_GRAYSCALE);
    if (img2.empty()) {
        cerr << "Error: Could not load image img2.png" << endl;
        return -1;
    }

    // SIFT feature detection for the second image
    vector<KeyPoint> keypoints2;
    Mat descriptors2;
    detectSIFTFeatures(img2, keypoints2, descriptors2);

    // Match features between the two images
    vector<DMatch> goodMatches = matchFeatures(descriptors1, descriptors2);

    // Draw and show the results
    Mat imgKeypoints1, imgMatches;
    drawKeypoints(img1, keypoints1, imgKeypoints1, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawMatches(img1, keypoints1, img2, keypoints2, goodMatches, imgMatches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    // Show results
    imshow("Harris Corner", harrisScaled);
    imshow("SIFT Keypoints", imgKeypoints1);
    imshow("Feature Matches", imgMatches);

    waitKey(0);
    return 0;
}
