#include "ros/ros.h"
#include "opencv2/opencv.hpp"
#include "sensor_msgs/Image.h"
#include "cv_bridge/cv_bridge.h"
#include "pointgrey.h"
#include "memory"
#include "iostream"
#include <pthread.h>

//#define VIDEO_SAVE
#define IMAGE_SHOW

typedef struct ARG
{
    int argc;
    char** argv;
}ARG;

ARG *arg;

cv::Mat cv_img(30, 30, CV_8UC3);
int width, height;

void *ROS_thread(void *param)
{
    ARG* arg_ = (ARG *)param;

    ros::init(arg_->argc, arg_->argv, "cam_node");
    ros::NodeHandle nh("~");
    ros::Publisher pub = nh.advertise<sensor_msgs::Image>("image",1);
    //ros::Rate loop_rate(30);
    cv_bridge::CvImage bridge;

    std::shared_ptr<PointGrey> cam(new PointGrey());
    bool init = cam->Initialize();
    if(!init){
        ROS_ERROR("Camera Initialize Error");
        //return -1;
        exit(1);
    }
    int cam_wid = cam->GetWidth();
    int cam_hei = cam->GetHeight();
    cv::Mat img = cv::Mat::zeros(cam_hei, cam_wid, CV_8UC1);
    cv::Mat cvt;
    bridge.header.frame_id = "camera";
    bridge.encoding = "rgb8";

    while(ros::ok()){
        try{
        cam->GrabImage((char*)img.data);
        cv::cvtColor(img, cvt, CV_BayerRG2BGR);
        cv::resize(cvt, bridge.image, cv::Size(width, height));

        cv::cvtColor(bridge.image, cv_img, CV_BGR2RGB);

        //if(cv::waitKey(1) == 'q') break;

        pub.publish(bridge.toImageMsg());
        }
        catch(const cv::Exception e){
            std::cout<<e.what()<<std::endl;
        }
        //loop_rate.sleep();
    }
    cam->DeInitialize();
}



int main(int argc, char** argv){
    arg = (ARG *)malloc(sizeof(ARG));
    arg->argc = argc;
    arg->argv = argv;

    int rate;
    if(argc>=4){
        width = atoi(argv[1]);
        height = atoi(argv[2]);
        rate = atoi(argv[3]);
    }
    else{
        width = 640;//1280;
        height = 480;//960;//720;
        rate = 30;
    }

    pthread_t p_thread[2];

    pthread_create(&p_thread[0], NULL, ROS_thread, (void *)arg);

    cv::VideoWriter outputVideo;
    double fps = 60;//rate;
    cv::Size size = cv::Size((int)width, (int)height);


#ifdef VIDEO_SAVE
    outputVideo.open("/media/chp/새 볼륨/K-city_logging/k_city_logging.avi", cv::VideoWriter::fourcc('X', 'V', 'I', 'D'), fps, size, true);	//(640,480) 'X', 'V', 'I', 'D'
    if (!outputVideo.isOpened())
    {
        std::cout << "µ¿¿µ»óÀ» ÀúÀåÇÏ±â À§ÇÑ ÃÊ±âÈ­ ÀÛŸ÷ Áß ¿¡·¯ ¹ß»ý" << std::endl;
        return -1;
    }
#endif

//ros::ok()
    while(1){
#ifdef IMAGE_SHOW
        cv::imshow("What about my age?", cv_img);
#endif

#ifdef VIDEO_SAVE
        outputVideo << cv_img;
#endif

        if(cv::waitKey(1) == 'q') break;
    }

    pthread_join(p_thread[0], NULL);

    return 0;
}
