
#include <iostream>
#include <algorithm>
#include <numeric>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "camFusion.hpp"
#include "dataStructures.h"
#include <queue>
using namespace std;


// Create groups of Lidar points whose projection into the camera falls into the same bounding box
void clusterLidarWithROI(std::vector<BoundingBox> &boundingBoxes, std::vector<LidarPoint> &lidarPoints, float shrinkFactor, cv::Mat &P_rect_xx, cv::Mat &R_rect_xx, cv::Mat &RT)
{
    // loop over all Lidar points and associate them to a 2D bounding box
    cv::Mat X(4, 1, cv::DataType<double>::type);
    cv::Mat Y(3, 1, cv::DataType<double>::type);

    for (auto it1 = lidarPoints.begin(); it1 != lidarPoints.end(); ++it1)
    {
        // assemble vector for matrix-vector-multiplication
        X.at<double>(0, 0) = it1->x;
        X.at<double>(1, 0) = it1->y;
        X.at<double>(2, 0) = it1->z;
        X.at<double>(3, 0) = 1;

        // project Lidar point into camera
        Y = P_rect_xx * R_rect_xx * RT * X;
        cv::Point pt;
        pt.x = Y.at<double>(0, 0) / Y.at<double>(0, 2); // pixel coordinates
        pt.y = Y.at<double>(1, 0) / Y.at<double>(0, 2);

        vector<vector<BoundingBox>::iterator> enclosingBoxes; // pointers to all bounding boxes which enclose the current Lidar point
        for (vector<BoundingBox>::iterator it2 = boundingBoxes.begin(); it2 != boundingBoxes.end(); ++it2)
        {
            // shrink current bounding box slightly to avoid having too many outlier points around the edges
            cv::Rect smallerBox;
            smallerBox.x = (*it2).roi.x + shrinkFactor * (*it2).roi.width / 2.0;
            smallerBox.y = (*it2).roi.y + shrinkFactor * (*it2).roi.height / 2.0;
            smallerBox.width = (*it2).roi.width * (1 - shrinkFactor);
            smallerBox.height = (*it2).roi.height * (1 - shrinkFactor);

            // check wether point is within current bounding box
            if (smallerBox.contains(pt))
            {
                enclosingBoxes.push_back(it2);
            }

        } // eof loop over all bounding boxes

        // check wether point has been enclosed by one or by multiple boxes
        if (enclosingBoxes.size() == 1)
        { 
            // add Lidar point to bounding box
            enclosingBoxes[0]->lidarPoints.push_back(*it1);
        }

    } // eof loop over all Lidar points
}


// void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait, string imgName)
{
    // create topview image
    cv::Mat topviewImg(imageSize, CV_8UC3, cv::Scalar(255, 255, 255));

    for(auto it1=boundingBoxes.begin(); it1!=boundingBoxes.end(); ++it1)
    {
        // create randomized color for current 3D object
        cv::RNG rng(it1->boxID);
        cv::Scalar currColor = cv::Scalar(rng.uniform(0,150), rng.uniform(0, 150), rng.uniform(0, 150));

        // plot Lidar points into top view image
        int top=1e8, left=1e8, bottom=0.0, right=0.0; 
        float xwmin=1e8, ywmin=1e8, ywmax=-1e8;
        for (auto it2 = it1->lidarPoints.begin(); it2 != it1->lidarPoints.end(); ++it2)
        {
            // world coordinates
            float xw = (*it2).x; // world position in m with x facing forward from sensor
            float yw = (*it2).y; // world position in m with y facing left from sensor
            xwmin = xwmin<xw ? xwmin : xw;
            ywmin = ywmin<yw ? ywmin : yw;
            ywmax = ywmax>yw ? ywmax : yw;

            // top-view coordinates
            int y = (-xw * imageSize.height / worldSize.height) + imageSize.height;
            int x = (-yw * imageSize.width / worldSize.width) + imageSize.width / 2;

            // find enclosing rectangle
            top = top<y ? top : y;
            left = left<x ? left : x;
            bottom = bottom>y ? bottom : y;
            right = right>x ? right : x;

            // draw individual point
            cv::circle(topviewImg, cv::Point(x, y), 4, currColor, -1);
        }

        // draw enclosing rectangle
        cv::rectangle(topviewImg, cv::Point(left, top), cv::Point(right, bottom),cv::Scalar(0,0,0), 2);

        // augment object with some key data
        char str1[200], str2[200];
        sprintf(str1, "id=%d, #pts=%d", it1->boxID, (int)it1->lidarPoints.size());
        putText(topviewImg, str1, cv::Point2f(left-250, bottom+50), cv::FONT_ITALIC, 2, currColor);
        sprintf(str2, "xmin=%2.2f m, yw=%2.2f m", xwmin, ywmax-ywmin);
        putText(topviewImg, str2, cv::Point2f(left-250, bottom+125), cv::FONT_ITALIC, 2, currColor);
        cout << "show3DPoints: " << it1->boxID << ", " <<  (int)it1->lidarPoints.size() << endl;
    }

    // plot distance markers
    float lineSpacing = 2.0; // gap between distance markers
    int nMarkers = floor(worldSize.height / lineSpacing);
    for (size_t i = 0; i < nMarkers; ++i)
    {
        int y = (-(i * lineSpacing) * imageSize.height / worldSize.height) + imageSize.height;
        cv::line(topviewImg, cv::Point(0, y), cv::Point(imageSize.width, y), cv::Scalar(255, 0, 0));
    }

    // display image
    string windowName = "3D Objects";
    cv::namedWindow(windowName, 1);
    cv::imshow(windowName, topviewImg);
    // // TESTING
    string saveFolder = "/home/workspace/akshay/SFND_3D_Object_Tracking/result/FP5/";
    string saveName = saveFolder + imgName;
    cv::imwrite(saveName, topviewImg);
    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}

void getKeyPointDistanceRatios(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches, vector<double> &distRatios)
{
    // compute distance ratios between all matched keypoints
    for (auto it1 = kptMatches.begin(); it1 != kptMatches.end() - 1; ++it1)
    { // outer kpt. loop
        // get current keypoint and its matched partner in the prev. frame
        cv::KeyPoint kpOuterCurr = kptsCurr.at(it1->trainIdx);
        cv::KeyPoint kpOuterPrev = kptsPrev.at(it1->queryIdx);
        for (auto it2 = kptMatches.begin() + 1; it2 != kptMatches.end(); ++it2)
        { // inner kpt.-loop

            double minDist = 100.0; // min. required distance
            double maxDist = 160.0;
            // get next keypoint and its matched partner in the prev. frame
            cv::KeyPoint kpInnerCurr = kptsCurr.at(it2->trainIdx);
            cv::KeyPoint kpInnerPrev = kptsPrev.at(it2->queryIdx);

            // compute distances and distance ratios
            double distCurr = cv::norm(kpOuterCurr.pt - kpInnerCurr.pt);
            double distPrev = cv::norm(kpOuterPrev.pt - kpInnerPrev.pt);
            
            if (distPrev > std::numeric_limits<double>::epsilon() && distCurr >= minDist && distCurr <= maxDist)
            { // avoid division by zero

                double distRatio = distCurr / distPrev;
                // cout << "distCurr: " << distCurr << " distPrev: " << distPrev << " distRatio: " << distRatio << endl;
                distRatios.push_back(distRatio);
            }
        } // eof inner loop over all matched kpts
    }     // eof outer loop over all matched kpts
}

// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
    std::vector<cv::DMatch> matchesForBB;
    std::vector<double> eucDistances;
    float shrinkFactor = -0.10;
    cv::Rect smallerBox;
    smallerBox.x = boundingBox.roi.x + shrinkFactor * boundingBox.roi.width / 2.0;
    smallerBox.y = boundingBox.roi.y + shrinkFactor * boundingBox.roi.height / 2.0;
    smallerBox.width = boundingBox.roi.width * (1 - shrinkFactor);
    smallerBox.height = boundingBox.roi.height * (1 - shrinkFactor);
    for (cv::DMatch match: kptMatches)
    {
        cv::KeyPoint prevKpt = kptsPrev.at(match.queryIdx);
        cv::KeyPoint currKpt = kptsCurr.at(match.trainIdx);
        if (smallerBox.contains(currKpt.pt) && smallerBox.contains(prevKpt.pt))
        // if (boundingBox.roi.contains(currKpt.pt) && boundingBox.roi.contains(prevKpt.pt))
        {
            matchesForBB.push_back(match);
            double dist = cv::norm(currKpt.pt - prevKpt.pt);
            eucDistances.push_back(dist);
        }
    }
    // boundingBox.kptMatches = matchesForBB;
    cout << matchesForBB.size() << ", ";
    // cout << "org bb matches count: " << matchesForBB.size() << endl;

    // for outlier removal (https://www.khanacademy.org/math/statistics-probability/summarizing-quantitative-data/box-whisker-plots/a/identifying-outliers-iqr-rule)
    // vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame    
    // getKeyPointDistanceRatios(kptsPrev, kptsCurr, matchesForBB, distRatios);
    // for (auto it1 = matchesForBB.begin(); it1 != matchesForBB.end() - 1; ++it1)
    // {

    // }
    // cout << "Euclidean distances: ";
    // for (double dist: eucDistances)
    //     cout << dist << ", ";
    // cout << endl;
    double q1Dist = getMedianFromVector(eucDistances, 0, eucDistances.size()/2 - 1);
    double q3Dist = getMedianFromVector(eucDistances, eucDistances.size()/2 + 1, eucDistances.size() - 1);
    double meanDist = std::accumulate(eucDistances.begin(), eucDistances.end(), 0.0) / eucDistances.size();
    double medianDist = getMedianFromVector(eucDistances, 0, eucDistances.size() - 1);
    double iqr = q3Dist - q1Dist;
    double iqrFactor = 1.2;
    // cout << "MeanDist: " << meanDist << endl;
    // cout << "MedianDist: " << medianDist << endl;
    // cout << "IQR: " << iqr << endl;
    // cout << "IQR lower bound: " << q1Dist - iqrFactor*iqr << endl;
    // cout << "IQR upper bound: " << q3Dist + iqrFactor*iqr << endl;

    double rangeFactor = 2.5;
    for (cv::DMatch match: matchesForBB)
    {
        cv::KeyPoint prevKpt = kptsPrev.at(match.queryIdx);
        cv::KeyPoint currKpt = kptsCurr.at(match.trainIdx);
        double dist = cv::norm(currKpt.pt - prevKpt.pt);
        if (dist <= (medianDist + rangeFactor*medianDist) && dist >= (medianDist - rangeFactor*medianDist))
        {
            boundingBox.kptMatches.push_back(match);
        }
    }
    // cout << "final bb count: " << boundingBox.kptMatches.size() << endl;
    cout << boundingBox.kptMatches.size() << ", ";

}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
    vector<double> distRatios; // stores the distance ratios for all keypoints between curr. and prev. frame    
    getKeyPointDistanceRatios(kptsPrev, kptsCurr, kptMatches, distRatios);
    // only continue if list of distance ratios is not empty
    if (distRatios.size() == 0)
    {
        TTC = NAN;
        return;
    }

    // compute camera-based TTC from distance ratios
    double meanDistRatio = std::accumulate(distRatios.begin(), distRatios.end(), 0.0) / distRatios.size();
    // std::sort(distRatios.begin(), distRatios.end());
    double medianDistRatio = getMedianFromVector(distRatios, 0, distRatios.size()-1);
    cout << "meanDistanceRatio: " << meanDistRatio << endl;
    cout << "medianDistanceRatio: " << medianDistRatio << endl;
    double dT = 1 / frameRate;
    TTC = -dT / (1 - medianDistRatio);

    // STUDENT TASK (replacement for meanDistRatio)
}

double getMedianFromVector(vector<double> vec, int start, int end)
{
    std::sort(vec.begin(), vec.end());
    int size = end - start + 1;
    int mid = start + (end - start) / 2;
    double median = 0.0;
    if (size % 2 == 1)
    {
        median =  vec[mid];
    }
    else
    {
        median = (vec[mid] + vec[mid-1]) / 2;
    }
    return median;
}


float getMedianFromQueue(priority_queue<float> q)
{
    // if (q.size() == 0)
    //     // throw exception?
    int mid = int(q.size() / 2);
    int i = 0;
    while (i < mid)
    {
        q.pop();
        i ++;
    }
    float result;
    if (q.size() % 2 == 0)
    {
        float p1 = q.top();
        q.pop();
        float p2 = q.top();
        result = (p1 + p2) / 2.0;
    }
    else
    {
        result = q.top();
    }
    return result;
}

void computeTTCLidar(std::vector<LidarPoint> &lidarPointsPrev,
                     std::vector<LidarPoint> &lidarPointsCurr, double frameRate, double &TTC)
{
    // ...
    bool useMedian = false;
    priority_queue <float> prevMinXQueue;
    priority_queue <float> currMinXQueue;
    float prevxwMin = 1e8, currxwMin = 1e8;
    float prevMinXValueRobust, currMinXValueRobust;
    int queue_size = 5;
    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1)
    {
        float xw = (*it1).x;
        prevxwMin = prevxwMin < xw ? prevxwMin : xw;
        prevMinXQueue.push(xw);
        if (prevMinXQueue.size() > queue_size)
            prevMinXQueue.pop();
    }

    for (auto it2 = lidarPointsCurr.begin(); it2 != lidarPointsCurr.end(); ++it2)
    {
        float xw = (*it2).x;
        currxwMin = currxwMin < xw ? currxwMin : xw;
        currMinXQueue.push(xw);
        if (currMinXQueue.size() > queue_size)
            currMinXQueue.pop();
    }

    if (useMedian)
    {
        prevMinXValueRobust = getMedianFromQueue(prevMinXQueue);
        currMinXValueRobust = getMedianFromQueue(currMinXQueue);
    }
    else
    {
        prevMinXValueRobust = prevxwMin;
        currMinXValueRobust = currxwMin;
    }
    double dT = 1 / frameRate;
    TTC = currMinXValueRobust * dT / (prevMinXValueRobust - currMinXValueRobust);
    // cout << useMedian << ", " << queue_size << ", " << frameRate << ", " << prevMinXValueRobust << ", " << currMinXValueRobust << ", "<< TTC << endl;
}


void matchBoundingBoxes(std::vector<cv::DMatch> &matches, std::map<int, int> &bbBestMatches, DataFrame &prevFrame, DataFrame &currFrame)
{
    // ...
    // vector<cv::KeyPoint> prevKpts = prevFrame.keypoints;
    // vector<cv::KeyPoint> currKpts = currFrame.keypoints;
    map<int, map<int, int>> boxIDMap;
    map<int, map<int, int>> reverseboxIDMap; // so that no more than one boxid from prev frame matches boxid from current frame
    for (cv::DMatch match: matches)
    {
        cv::KeyPoint prevKpt = prevFrame.keypoints.at(match.queryIdx);
        cv::KeyPoint currKpt = currFrame.keypoints.at(match.trainIdx);
        // cv::KeyPoint prevKpt = prevFrame.keypoints[match.trainIdx];
        // cv::KeyPoint currKpt = currFrame.keypoints[match.queryIdx];
        vector<int> prevBoxIDs;
        vector<int> currBoxIDs;
        for (vector<BoundingBox>::iterator it1 = prevFrame.boundingBoxes.begin(); it1 != prevFrame.boundingBoxes.end(); ++it1)
        {
            if(it1->roi.contains(prevKpt.pt))
            {
                prevBoxIDs.push_back(it1->boxID);
            }
        }

        for (vector<BoundingBox>::iterator it2 = currFrame.boundingBoxes.begin(); it2 != currFrame.boundingBoxes.end(); ++it2)
        {
            if(it2->roi.contains(currKpt.pt))
            {
                currBoxIDs.push_back(it2->boxID);
            }
        }

        if (prevBoxIDs.size() > 0 && currBoxIDs.size() > 0)
        {
            for (int prevBoxID: prevBoxIDs)
            {
                for (int currBoxID: currBoxIDs)
                {
                    boxIDMap[prevBoxID][currBoxID] ++;
                }
            }
        }
    }

    for (auto outerBoxMap: boxIDMap)
    {
        int maxValue = -1;
        int maxID = -1;
        for (auto innerBoxMap: outerBoxMap.second)
        {
            if (innerBoxMap.second > maxValue)
            {
                maxValue = innerBoxMap.second;
                maxID = innerBoxMap.first;
            }
        }
        if (maxID != -1)
            // bbBestMatches[outerBoxMap.first] = maxID;
            reverseboxIDMap[maxID][outerBoxMap.first] = maxValue;
        
    }

    // generate final result
    for (auto outerMap: reverseboxIDMap)
    {
        int maxValue = -1;
        int maxID = -1;
        for (auto innerMap: outerMap.second)
        {
            if (innerMap.second > maxValue)
            {
                maxValue = innerMap.second;
                maxID = innerMap.first;
            }
        }
        if (maxID != -1)
            bbBestMatches[maxID] = outerMap.first;
            // cout << maxID << ", " << outerMap.first << endl;
    }

}
