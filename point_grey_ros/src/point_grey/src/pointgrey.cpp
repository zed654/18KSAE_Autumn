#include "pointgrey.h"

PointGrey::PointGrey()
{
    system = System::GetInstance();
    pCam = NULL;
    pResultImage = NULL;
    width = height =0;
}
void PointGrey::DeInitialize(){
    pCam->EndAcquisition();
    pCam->DeInit();
    pCam = NULL;
    camList.Clear();
    system->ReleaseInstance();
}

bool PointGrey::Initialize(){
    camList = system->GetCameras();
    cameraCount = camList.GetSize();
    if(cameraCount==0){
        camList.Clear();
        system->ReleaseInstance();
        printf("Failed to Loading Camera\n");
        return false;
    }
    pCam = camList.GetByIndex(cameraCount-1);
    try{
        pCam->Init();
    }
    catch(Spinnaker::Exception &e){
        std::cout<<"Error: "<<e.what()<<std::endl;
        return false;
    }

    INodeMap& nodeMap = pCam->GetNodeMap();
//    CIntegerPtr pHeight = nodeMap.GetNode("Height");
//    CIntegerPtr pWidth = nodeMap.GetNode("Width");
//    pWidth->SetValue((uint64_t)1920);
//    pHeight->SetValue(1080);

    try{
        CEnumerationPtr ptrAcquisitionMode = nodeMap.GetNode("AcquisitionMode");
        if (!IsAvailable(ptrAcquisitionMode) || !IsWritable(ptrAcquisitionMode))
        {
            std::cout << "Unable to set acquisition mode to continuous (enum retrieval). Aborting..." << std::endl << std::endl;
            return -1;
        }
        CEnumEntryPtr ptrAcquisitionModeContinuous = ptrAcquisitionMode->GetEntryByName("Continuous");
        if (!IsAvailable(ptrAcquisitionModeContinuous) || !IsReadable(ptrAcquisitionModeContinuous))
        {
            std::cout << "Unable to set acquisition mode to continuous (entry retrieval). Aborting..." << std::endl << std::endl;
            return -1;
        }

        int64_t acquisitionModeContinuous = ptrAcquisitionModeContinuous->GetValue();

        ptrAcquisitionMode->SetIntValue(acquisitionModeContinuous);

        std::cout << "Acquisition mode set to continuous..." << std::endl;
        pCam->BeginAcquisition();
    }
    catch (Spinnaker::Exception &e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        return false;
    }
    pResultImage = pCam->GetNextImage();
    width = pResultImage->GetWidth();
    height = pResultImage->GetHeight();
    return true;
}

int PointGrey::GrabImage(char* img_data){
    int result = 0;
    try{
        pResultImage = pCam->GetNextImage();
        width = pResultImage->GetWidth();
        height = pResultImage->GetHeight();
        memcpy(img_data, (char*)pResultImage->GetData(), sizeof(char)*width*height);
    }
    catch (Spinnaker::Exception &e)
    {
        std::cout << "Error: " << e.what() << std::endl;
        result = -1;
    }
    return result;
}

int PointGrey::GetWidth(){
    return width;
}

int PointGrey::GetHeight(){
    return height;
}
