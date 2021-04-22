

< 2018 KSAE Autumn Full Paper >

	------------------------------------------------------------
		< How to Run >
	
	- The default recognition system is based on the saved video file, named 'ReadRoad.avi'. To connect camera, refer the section of 'Point_grey"


	- Recognition Algorithm Command (Yolo V3)
		1. Video File based.
			a. roslaunch darknet_ros yolo_v3_tiny_TSR_video.launch
			b. roslaunch darknet_ros yolo_v3_TSR_video.launch

		2. Point Grey Camera Based
			a. roslaunch darknet_ros yolo_v3_tiny_TSR_Point_grey.launch

	- Video File Execution Command
		roslaunch video_stream_opencv video_file.launch
			video_file.launch -> The selected file is written in '<arg name="video_stream_provider" ...>'. The name of default video file is RealRoad.avi
			(video_stream_opencv-master/data/RealRoad.avi)
			

	- Point Grey Execution Command
		rosrun point_grey point_grey_node
		(To use point_grey camera, 
			1. You should move the folder of 'point_grey_ros' to the 'src' folder.
			2. You should install the spinnaker. And should modify bios setting. You can find it on website of the PointGrey.


	- Vehicle Control and Decision Command
		rosrun ioniq_control ioniq_control
	------------------------------------------------------------
	
		< The Method of inserting input Img >
		
	- Use the pkg
		1. point_grey_ros
		2. usb_cam
		3. video_stream_opencv-master
		
	------------------------------------------------------------
	


 < etc ... >
 
	src : TSR + Vehicle Low Level Control written as ROS

	Latitude Control : Stanley Controller
	Longitudinal Control : PD Controller


	File :
		1. how_to_download_weights.txt
		2. how_to_download_videos.txt
