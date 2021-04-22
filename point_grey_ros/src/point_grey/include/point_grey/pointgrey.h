#ifndef POINTGREY_H
#define POINTGREY_H

#include "Spinnaker.h"
#include "SpinGenApi/SpinnakerGenApi.h"

using namespace Spinnaker;
using namespace Spinnaker::GenApi;
using namespace Spinnaker::GenICam;


class PointGrey
{
private:
    SystemPtr system;
    CameraList camList;
    uint8_t cameraCount;
    CameraPtr pCam;
    ImagePtr pResultImage;
    int width, height;

public:
    PointGrey();
    bool Initialize();
    void DeInitialize();
    int GrabImage(char* img_data);
//    bool RunGrab();
    int GetWidth();
    int GetHeight();

};

#endif // POINTGREY_H
