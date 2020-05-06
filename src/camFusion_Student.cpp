
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


void show3DObjects(std::vector<BoundingBox> &boundingBoxes, cv::Size worldSize, cv::Size imageSize, bool bWait)
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

    if(bWait)
    {
        cv::waitKey(0); // wait for key to be pressed
    }
}


// associate a given bounding box with the keypoints it contains
void clusterKptMatchesWithROI(BoundingBox &boundingBox, std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, std::vector<cv::DMatch> &kptMatches)
{
    // ...
}


// Compute time-to-collision (TTC) based on keypoint correspondences in successive images
void computeTTCCamera(std::vector<cv::KeyPoint> &kptsPrev, std::vector<cv::KeyPoint> &kptsCurr, 
                      std::vector<cv::DMatch> kptMatches, double frameRate, double &TTC, cv::Mat *visImg)
{
    // ...
}

float getMedian(priority_queue<float> q)
{
    // if (q.size() == 0)
    //     // throw exception?
    int mid = int(q.size() / 2);
    int i = 0;
    while (i < mid)
    {
        q.pop();
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
    priority_queue <float> prevMinXQueue;
    priority_queue <float> currMinXQueue;
    int queue_size = 5;
    for (auto it1 = lidarPointsPrev.begin(); it1 != lidarPointsPrev.end(); ++it1)
    {
        float xw = (*it1).x;
        prevMinXQueue.push(xw);
        if (prevMinXQueue.size() > queue_size)
            prevMinXQueue.pop();
    }

    for (auto it2 = lidarPointsCurr.begin(); it2 != lidarPointsCurr.end(); ++it2)
    {
        float xw = (*it2).x;
        currMinXQueue.push(xw);
        if (currMinXQueue.size() > queue_size)
            currMinXQueue.pop();
    }

    float prevMinXValueRobust = getMedian(prevMinXQueue);
    float currMinXValueRobust = getMedian(currMinXQueue);

    double dT = 1 / frameRate;
    TTC = currMinXValueRobust * dT / (prevMinXValueRobust - currMinXValueRobust);
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
        cv::KeyPoint prevKpt = prevFrame.keypoints[match.queryIdx];
        cv::KeyPoint currKpt = currFrame.keypoints[match.trainIdx];
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
            cout << maxID << ", " << outerMap.first << endl;
    }

}
