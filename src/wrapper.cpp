#include <iostream>
#include <vector>

using namespace std;

void project(string detectorType, string descriptorType);

int main(int argc, const char *argv[])
{
    vector<string> detectorVec = {"SHITOMASI", "HARRIS", "FAST", "BRISK", "ORB"}; // SIFT pending
    // vector<string> detectorVec = {"SIFT"};
    // vector<string> descVec = {"BRISK", "FREAK"};
    vector<string> descVec = {"BRISK", "ORB", "FREAK"};
    // vector<string> descVec = {"ORB"};


    for (string detectorType: detectorVec)
    {
        for (string descType: descVec)
        {
            // cout << detectorType << "," << descType << endl;
            project(detectorType, descType);
        }
    }
    project("AKAZE", "AKAZE");
    // project("SIFT", "SIFT");


    return 0;
}