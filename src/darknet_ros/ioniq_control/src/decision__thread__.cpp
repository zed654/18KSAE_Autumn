#include <stdio.h>
#include <iostream>
#include "env_setting.h"
#include "TSR_receive.h"
//#include "coord_map.h"
#include <sys/time.h>
#include <unistd.h>
#include <queue>
#include <ros/ros.h>

#define __PINRT__
/*
result_num
tsr[result_num].data_class
tsr[result_num].data_probability
*/


/*
  < TSR Sginal decision >
  Getting Traffic Sign Type (Red, Yellow, Grre)

  < Longitunal Control >

 1. Getting Coordination of Stop Point  (56	36.72792210		127.44325960)
 2. Getting Coordination of vehicle

            : 36.72795090		127.44334110		269.04137000
    convert :
            gnss_x =(globalPVT.lat - base_latitude_tmp)* 10000000 * 0.888 / 100. + 30;//100; // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가
            gnss_y =(globalPVT.lon - base_longitude_tmp) * 10000000 * 1.1 / 100. + 10;//30;  // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가

 3. Calculate a distance between Stop Point to vehicle -> Distance : [d]
 4. if [d] is shorten than 50m,
*/




/*
double base_latitude_tmp = base_latitude; //36.62489480;
double base_longitude_tmp = base_longitude;//127.45667630;   //base_longitude;
// k-city offset ;; x = 300, y = 120
// ochang offset : x = 30, y = 10
gnss_x =(globalPVT.lat - base_latitude_tmp)* 10000000 * 0.888 / 100. + 30;//100; // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가
gnss_y =(globalPVT.lon - base_longitude_tmp) * 10000000 * 1.1 / 100. + 10;//30;  // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가
*/

// Stop Point X, Y
double SP_global_x = 36.72815850;//		127.44320640//coord_map->WP_stop_line_latitude;//36.72792210;
double SP_global_y = 127.44320640; //coord_map->WP_stop_line_longitude;//127.44325960;

double SP_x;    // =(SP_global_x - base_latitude)* 10000000 * 0.888 / 100. + 30;      //100; // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가
double SP_y;    // =(SP_global_y - base_longitude) * 10000000 * 1.1 / 100. + 10;      //30;  // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가

//dr_x;
//dr_y;


void* Decition_thread(void* param)
{
    // waiting for ros::init
    sleep(2);

    SP_x = 0;
    SP_y = 0;

    // Decision Behavior Flag of vehicle if signal is Yellow
    bool yellow_stop_flag = 0;

    // when search stop signal at first, this flag set to 1
    //bool Init_stop_search_flag = 0;

    while(ros::ok())    // if do not use ros::ok, system delay is happen ( ex. if use swhile(1) )
    {
        SP_x =(SP_global_x - base_latitude)* 10000000 * 0.888 / 100. + 30;      //100; // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가
        SP_y =(SP_global_y - base_longitude) * 10000000 * 1.1 / 100. + 10;      //30;  // Visualize를 위해 (0, 0)으로부터 좌표이동도 추가

        // Calc current_SP_distance
        current_SP_distance = sqrt((SP_x - dr_x)*(SP_x - dr_x) + (SP_y - dr_y)*(SP_y - dr_y));


        //
        if((current_SP_distance > 5) && ( coord_current_address >= 56 ))
            SP_passing_flag = 1;
        else
            SP_passing_flag = 0;

#ifdef __PINRT__
        if(vehicle_yaw_rate_error_correct == 1)
        {
            std::cout << "current_SP_distance : " << current_SP_distance << "\t\tcurrent_address : " << coord_current_address << "\t\tSP_passing_flag : " << SP_passing_flag <<
                     "\t\tinit_stop_search_flag : " << init_stop_search_flag << "\t\tcurrent_vehicle_speed : " << current_vehicle_speed << "\t\ttarget_speed : " << target_vehicle_speed << std::endl;
            //std::endl; //<< "\t\tinit_SP_distance : " << init_SP_distance << "\t\tinit_target_speed : " << init_target_speed << std::endl;
//            std::cout << "current_vehicle_speed : " << current_vehicle_speed << "\t\ttarget_speed : " << target_vehicle_speed << std::endl;
        }
#endif

        if(SP_passing_flag == 0)
        {
            if(current_SP_distance < 50)
            {
                if(tsr[result_num].data_search_flag == 1)
                {
                    switch(tsr[result_num].data_class)  // 0, 1, 2, 3
                    {
                    case 1:

                        if((current_SP_distance < 25) && (yellow_stop_flag == 0))
                        {
                            // Wrong Case
                            // Wrong Case
                            // Wrong Case
                            // Wrong Case
                            // When Yellow Signal is detected at nearby 25m, Car should go. But, It
                            std::cout << "Wrong Case RED. Have to go!!!" << std::endl;
                            // 1. Follow Target Speed of Coord Map
                        }
                        else if((current_SP_distance > 25) && (yellow_stop_flag == 0))
                        {
                            std::cout << "TSR : RED, first search" << std::endl;
                            yellow_stop_flag = 1;
                            init_stop_search_flag = 1;

                            init_SP_distance = current_SP_distance;
                            init_target_speed = current_vehicle_speed;

                            // 1. Follow Target Speed of stopping to SP
                            // 2. Calc Target Speed of stopping at SP           CalcCalcCalcCalc
                        }
                        else if((current_SP_distance > 25) && (yellow_stop_flag == 1))
                        {
                            std::cout << "TSR : RED" << std::endl;
                        }
                        break;

                    case 2:
                        if((current_SP_distance < 25) && (yellow_stop_flag == 0))
                        {
                            std::cout << "TSR : Yellow, Go" << std::endl;
                            // 1. Follow Set Target Speed of Coord Map
                        }
                        else if((current_SP_distance < 25) && (yellow_stop_flag == 1))
                        {
                            std::cout << "TSR : Yellow, Stop" << std::endl;
                            // 1. Follow Target Speed of stopping at SP
                            // 2. Do not calc Target Speed Fun
                        }
                        else if((current_SP_distance >= 25) && (yellow_stop_flag == 0)) // First Search
                        {
                            std:: cout << "TSR : Yellow, Stop, First search" << std::endl;
                            yellow_stop_flag = 1;
                            init_stop_search_flag = 1;

                            init_SP_distance = current_SP_distance;
                            init_target_speed = current_vehicle_speed;
                            // 1. Follow Target Speed of stopping at SP
                            // 2. Calc Target Speed of stopping at SP           CalcCalcCalcCalc
                        }
                        else if((current_SP_distance >= 25) && (yellow_stop_flag == 1))
                        {
                            std::cout << "TSR : Yellow, Stop, alread searched" << std::endl;
                            // 1. Follow Target Speed of stopping at SP
                            // 2. Do not calc Target Speed Fun
                        }
                        break;

                    case 3:
                        if(init_stop_search_flag == 1)
                        {
                            std::cout << "TSR : GREEN, Restart!" << std::endl;
                            //vehicle_restart_flag = 1;
                            init_stop_search_flag = 0;
                        }
                        else
                            std::cout << "TSR : GREEN" << std::endl;
                        // Follow Set Target Speed
                        break;

                    case 4:
                        if(init_stop_search_flag == 1)
                        {
                            std::cout << "TSR : L_GREEN, Restart!" << std::endl;
                            //vehicle_restart_flag = 1;
                            init_stop_search_flag = 0;
                        }
                        else
                            std::cout << "TSR : L_GREEN" << std::endl;
                        // Follow Set Target Speed
                        break;
                    }
                }
            }
        }
        else if(SP_passing_flag == 1) // SP_passing_flag == 1
        {
            std::cout << "SP_passing_flag == 1" << std::endl;
            //break;
        }
            // escape Loop
            // == break;

        if(tsr[result_num].data_search_flag == 1)
        {
            // use moving avg (voting)

        }
        //std::cout << "Classifier result :[" << result_num << "] :" << tsr[result_num].data_class << "\t\t" << tsr[result_num].data_probability << "\t\t" << tsr[result_num].data_xmin << "\t\t" << tsr[result_num].data_xmax << "\t\t" << tsr[result_num].data_ymin <<  "\t\t" << tsr[result_num].data_ymax << "\t\t" << tsr[result_num].roi_size << std::endl;
        //std::cout << tsr[result_num].data_search_flag << std::endl;


    }
}



