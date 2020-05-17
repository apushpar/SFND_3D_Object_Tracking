#include <numeric>
#include "matching2D.hpp"
#include <iostream>
#include <fstream>

using namespace std;

// Find best matches for keypoints in two camera images based on several matching methods
void matchDescriptors(std::vector<cv::KeyPoint> &kPtsSource, std::vector<cv::KeyPoint> &kPtsRef, cv::Mat &descSource, cv::Mat &descRef,
                      std::vector<cv::DMatch> &matches, std::string descriptorType, std::string matcherType, std::string selectorType)
{
    // configure matcher
    bool crossCheck = false;
    cv::Ptr<cv::DescriptorMatcher> matcher;

    if (matcherType.compare("MAT_BF") == 0)
    {
        int normType = cv::NORM_HAMMING;
        matcher = cv::BFMatcher::create(normType, crossCheck);
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        // ...
        if (descSource.type() != CV_32F)
        {
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::FlannBasedMatcher::create();
    }

    // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)

        vector<vector<cv::DMatch>> knnMatches;
        int k = 2;
        double compareRatio = 0.8;
        matcher->knnMatch(descSource, descRef, knnMatches, k);

        double ratio;
        for (vector<cv::DMatch> matchArr : knnMatches)
        {
            ratio = matchArr[0].distance / matchArr[1].distance;
            if (ratio >= compareRatio)
            {
                matches.push_back(matchArr[0]);
            }
        }

    }
}

// Use one of several types of state-of-art descriptors to uniquely identify keypoints
void descKeypoints(vector<cv::KeyPoint> &keypoints, cv::Mat &img, cv::Mat &descriptors, string descriptorType)
{
    // select appropriate descriptor
    cv::Ptr<cv::DescriptorExtractor> extractor;
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        int nfeat = 500;
        float scaleFactor = 1.2f;
        int nLevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int wta_k =2;
        // cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;
        extractor = cv::ORB::create(nfeat, scaleFactor, nLevels, edgeThreshold, firstLevel, wta_k, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
    }
    else if (descriptorType.compare("FREAK") == 0)
    {
        bool orientationNormalized = true;
        bool scaleNormalized = true;
        float patternScale = 22.0f;
        int nOctaves = 4;
        extractor = cv::xfeatures2d::FREAK::create(orientationNormalized, scaleNormalized, patternScale, nOctaves);
    }
    else if (descriptorType.compare("AKAZE") == 0)
    {
        int descriptorSize = 0;
        int descriptorChannels = 3;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        extractor = cv::AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, descriptorSize, descriptorChannels, threshold, nOctaves, nOctaveLayers, cv::KAZE::DIFF_PM_G2);

    }
    else
    {
        // SIFT
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        extractor = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    
    // perform feature description
    double t = (double)cv::getTickCount();
    extractor->compute(img, keypoints, descriptors);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << descriptorType << " descriptor extraction in " << 1000 * t / 1.0 << " ms" << endl;
    // cout << 1000 * t / 1.0 << ",";
}

void visualizeResults(cv::Mat img, vector<cv::KeyPoint> &keypoints, string name)
{
    cv::Mat visImage = img.clone();
    cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    string windowName = name + " Corner Detector Results";
    cv::namedWindow(windowName, 6);
    imshow(windowName, visImage);
    cv::waitKey(0);
}

// Detect keypoints in image using the traditional Shi-Thomasi detector
void detKeypointsShiTomasi(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // compute detector parameters based on image size
    int blockSize = 4;       //  size of an average block for computing a derivative covariation matrix over each pixel neighborhood
    double maxOverlap = 0.0; // max. permissible overlap between two features in %
    double minDistance = (1.0 - maxOverlap) * blockSize;
    int maxCorners = img.rows * img.cols / max(1.0, minDistance); // max. num. of keypoints

    double qualityLevel = 0.01; // minimal accepted quality of image corners
    double k = 0.04;

    // Apply corner detection
    double t = (double)cv::getTickCount();
    vector<cv::Point2f> corners;
    cv::goodFeaturesToTrack(img, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, false, k);

    // add corners to result vector
    for (auto it = corners.begin(); it != corners.end(); ++it)
    {

        cv::KeyPoint newKeyPoint;
        newKeyPoint.pt = cv::Point2f((*it).x, (*it).y);
        newKeyPoint.size = blockSize;
        keypoints.push_back(newKeyPoint);
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Shi-Tomasi codetection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // cout << 1000 * t / 1.0 << ",";
    // visualize results
    if (bVis)
    {
        visualizeResults(img, keypoints, "Shi-Tomasi");
        
    }
}

void detKeypointsHarris(vector<cv::KeyPoint> &keypoints, cv::Mat &img, bool bVis)
{
    // Detector parameters
    int blockSize = 2; // for every pixel, a blockSize × blockSize neighborhood is considered
    int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
    int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
    double k = 0.04; // Harris parameter (see equation for details)

    double t = (double)cv::getTickCount();
    // Detect Harris corners and normalize output
    cv::Mat dst, dst_norm, dst_norm_scaled;
    dst = cv::Mat::zeros(img.size(), CV_32FC1 );
    cv::cornerHarris( img, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT ); 
    cv::normalize( dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat() );
    cv::convertScaleAbs( dst_norm, dst_norm_scaled );

    // vector<cv::KeyPoint> kpts;
    double maxOverlap = 0;
    for(size_t j = 0; j < dst_norm.rows; j++)
    {
        for(size_t i = 0; i < dst_norm.cols; i++)
        {
            int response = dst_norm.at<float>(j, i);
            if (response > minResponse)
            {
                cv::KeyPoint newKpt;
                newKpt.pt = cv::Point2f(i, j);
                newKpt.size = 2 * apertureSize;
                newKpt.response = response;

                // nms over neighborhood
                bool bOverlap = false;
                for (auto it = keypoints.begin(); it != keypoints.end(); ++it)
                {
                    double kptOverlap = cv::KeyPoint::overlap(newKpt, *it);
                    if (kptOverlap > maxOverlap)
                    {
                        bOverlap = true;
                        if (newKpt.response > (*it).response)
                        {
                            *it = newKpt;
                            break;
                        }
                    }
                }
                if (!bOverlap)
                {
                    keypoints.push_back(newKpt);
                }
            }
        }
    }
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << "Harris detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // cout << 1000 * t / 1.0 << ",";
    // visualize results
    if (bVis)
    {
        visualizeResults(img, keypoints, "Harris");
    }
}

void detKeypointsModern(vector<cv::KeyPoint> &keypoints, cv::Mat &img, string detectorType, bool bVis)
{   
    double t = (double)cv::getTickCount();
    cv::Ptr<cv::FeatureDetector> detector;
    if (detectorType.compare("FAST") == 0)
    {
        int threshold = 30;
        bool nms = true;
        // cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create(threshold, nms, cv::FastFeatureDetector::TYPE_9_16);
        detector = cv::FastFeatureDetector::create(threshold, nms, cv::FastFeatureDetector::TYPE_9_16);
        // detector->detect(img, keypoints);
        // cv::FAST(img, keypoints, threshold, nms, cv::FastFeatureDetector::TYPE_9_16);
        
    }
    else if (detectorType.compare("BRISK") == 0)
    {
        int threshold = 30;
        int octaves = 3;
        float patternScale = 1.0f;
        detector = cv::BRISK::create(threshold, octaves, patternScale);

    }
    else if (detectorType.compare("ORB") == 0)
    {
        int nfeat = 500;
        float scaleFactor = 1.2f;
        int nLevels = 8;
        int edgeThreshold = 31;
        int firstLevel = 0;
        int wta_k =2;
        // cv::ORB::ScoreType scoreType = cv::ORB::HARRIS_SCORE;
        int patchSize = 31;
        int fastThreshold = 20;
        detector = cv::ORB::create(nfeat, scaleFactor, nLevels, edgeThreshold, firstLevel, wta_k, cv::ORB::HARRIS_SCORE, patchSize, fastThreshold);
    }
    else if (detectorType.compare("AKAZE") == 0)
    {
        bool extended = false;
        bool upright = false;
        float threshold = 0.001f;
        int nOctaves = 4;
        int nOctaveLayers = 4;
        detector = cv::KAZE::create(extended, upright, threshold, nOctaves, nOctaveLayers); // CHECK
    }
    else
    {
        // SIFT
        int nfeatures = 0;
        int nOctaveLayers = 3;
        double contrastThreshold = 0.04;
        double edgeThreshold = 10;
        double sigma = 1.6;
        detector = cv::xfeatures2d::SIFT::create(nfeatures, nOctaveLayers, contrastThreshold, edgeThreshold, sigma);
    }
    detector->detect(img, keypoints);
    t = ((double)cv::getTickCount() - t) / cv::getTickFrequency();
    cout << detectorType << " detection with n=" << keypoints.size() << " keypoints in " << 1000 * t / 1.0 << " ms" << endl;
    // cout << 1000 * t / 1.0 << ",";
    // visualize results
    if (bVis)
    {
        visualizeResults(img, keypoints, detectorType);
    }
}