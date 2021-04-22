// #define MAKE_CHP			// 내가 추가한 코드들
// #define DETECTION_CHP		// detection 뺀 main문 돌리기
// #define TRAIN_CHP		// 내가 추가한 TRAIN쪽 코드
// #define ORIGIN_YOLO
#define DETECTION_CHP

#ifdef _DEBUG
//#define OPENCV
#include <stdlib.h> 
#include <crtdbg.h>  
#endif

#include <time.h>
#include <stdlib.h>
#include <stdio.h>

#include "parser.h"
#include "utils.h"
#include "cuda.h"
#include "blas.h"
#include "connected_layer.h"
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

extern void predict_classifier(char *datacfg, char *cfgfile, char *weightfile, char *filename, int top);
extern void test_detector(char *datacfg, char *cfgfile, char *weightfile, char *filename, float thresh, int ext_output);
extern void run_voxel(int argc, char **argv);
extern void run_yolo(int argc, char **argv);
extern void run_detector(int argc, char **argv);
extern void run_coco(int argc, char **argv);
extern void run_writing(int argc, char **argv);
extern void run_captcha(int argc, char **argv);
extern void run_nightmare(int argc, char **argv);
extern void run_dice(int argc, char **argv);
extern void run_compare(int argc, char **argv);
extern void run_classifier(int argc, char **argv);
extern void run_char_rnn(int argc, char **argv);
extern void run_vid_rnn(int argc, char **argv);
extern void run_tag(int argc, char **argv);
extern void run_cifar(int argc, char **argv);
extern void run_go(int argc, char **argv);
extern void run_art(int argc, char **argv);
extern void run_super(int argc, char **argv);


void average(int argc, char *argv[])
{
	char *cfgfile = argv[2];
	char *outfile = argv[3];
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	network sum = parse_network_cfg(cfgfile);

	char *weightfile = argv[4];
	load_weights(&sum, weightfile);

	int i, j;
	int n = argc - 5;
	for (i = 0; i < n; ++i) {
		weightfile = argv[i + 5];
		load_weights(&net, weightfile);
		for (j = 0; j < net.n; ++j) {
			layer l = net.layers[j];
			layer out = sum.layers[j];
			if (l.type == CONVOLUTIONAL) {
				int num = l.n*l.c*l.size*l.size;
				axpy_cpu(l.n, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(num, 1, l.weights, 1, out.weights, 1);
				if (l.batch_normalize) {
					axpy_cpu(l.n, 1, l.scales, 1, out.scales, 1);
					axpy_cpu(l.n, 1, l.rolling_mean, 1, out.rolling_mean, 1);
					axpy_cpu(l.n, 1, l.rolling_variance, 1, out.rolling_variance, 1);
				}
			}
			if (l.type == CONNECTED) {
				axpy_cpu(l.outputs, 1, l.biases, 1, out.biases, 1);
				axpy_cpu(l.outputs*l.inputs, 1, l.weights, 1, out.weights, 1);
			}
		}
	}
	n = n + 1;
	for (j = 0; j < net.n; ++j) {
		layer l = sum.layers[j];
		if (l.type == CONVOLUTIONAL) {
			int num = l.n*l.c*l.size*l.size;
			scal_cpu(l.n, 1. / n, l.biases, 1);
			scal_cpu(num, 1. / n, l.weights, 1);
			if (l.batch_normalize) {
				scal_cpu(l.n, 1. / n, l.scales, 1);
				scal_cpu(l.n, 1. / n, l.rolling_mean, 1);
				scal_cpu(l.n, 1. / n, l.rolling_variance, 1);
			}
		}
		if (l.type == CONNECTED) {
			scal_cpu(l.outputs, 1. / n, l.biases, 1);
			scal_cpu(l.outputs*l.inputs, 1. / n, l.weights, 1);
		}
	}
	save_weights(sum, outfile);
}

void speed(char *cfgfile, int tics)
{
	if (tics == 0) tics = 1000;
	network net = parse_network_cfg(cfgfile);
	set_batch_network(&net, 1);
	int i;
	time_t start = time(0);
	image im = make_image(net.w, net.h, net.c);
	for (i = 0; i < tics; ++i) {
		network_predict(net, im.data);
	}
	double t = difftime(time(0), start);
	printf("\n%d evals, %f Seconds\n", tics, t);
	printf("Speed: %f sec/eval\n", t / tics);
	printf("Speed: %f Hz\n", tics / t);
}

void operations(char *cfgfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	int i;
	long ops = 0;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			ops += 2l * l.n * l.size*l.size*l.c * l.out_h*l.out_w;
		}
		else if (l.type == CONNECTED) {
			ops += 2l * l.inputs * l.outputs;
		}
	}
	printf("Floating Point Operations: %ld\n", ops);
	printf("Floating Point Operations: %.2f Bn\n", (float)ops / 1000000000.);
}

void oneoff(char *cfgfile, char *weightfile, char *outfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	int oldn = net.layers[net.n - 2].n;
	int c = net.layers[net.n - 2].c;
	net.layers[net.n - 2].n = 9372;
	net.layers[net.n - 2].biases += 5;
	net.layers[net.n - 2].weights += 5 * c;
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	net.layers[net.n - 2].biases -= 5;
	net.layers[net.n - 2].weights -= 5 * c;
	net.layers[net.n - 2].n = oldn;
	printf("%d\n", oldn);
	layer l = net.layers[net.n - 2];
	copy_cpu(l.n / 3, l.biases, 1, l.biases + l.n / 3, 1);
	copy_cpu(l.n / 3, l.biases, 1, l.biases + 2 * l.n / 3, 1);
	copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + l.n / 3 * l.c, 1);
	copy_cpu(l.n / 3 * l.c, l.weights, 1, l.weights + 2 * l.n / 3 * l.c, 1);
	*net.seen = 0;
	save_weights(net, outfile);
}

void partial(char *cfgfile, char *weightfile, char *outfile, int max)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights_upto(&net, weightfile, max);
	}
	*net.seen = 0;
	save_weights_upto(net, outfile, max);
}

#include "convolutional_layer.h"
void rescale_net(char *cfgfile, char *weightfile, char *outfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			rescale_weights(l, 2, -.5);
			break;
		}
	}
	save_weights(net, outfile);
}

void rgbgr_net(char *cfgfile, char *weightfile, char *outfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL) {
			rgbgr_weights(l);
			break;
		}
	}
	save_weights(net, outfile);
}

void reset_normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL && l.batch_normalize) {
			denormalize_convolutional_layer(l);
		}
		if (l.type == CONNECTED && l.batch_normalize) {
			denormalize_connected_layer(l);
		}
		if (l.type == GRU && l.batch_normalize) {
			denormalize_connected_layer(*l.input_z_layer);
			denormalize_connected_layer(*l.input_r_layer);
			denormalize_connected_layer(*l.input_h_layer);
			denormalize_connected_layer(*l.state_z_layer);
			denormalize_connected_layer(*l.state_r_layer);
			denormalize_connected_layer(*l.state_h_layer);
		}
	}
	save_weights(net, outfile);
}

layer normalize_layer(layer l, int n)
{
	int j;
	l.batch_normalize = 1;
	l.scales = calloc(n, sizeof(float));
	for (j = 0; j < n; ++j) {
		l.scales[j] = 1;
	}
	l.rolling_mean = calloc(n, sizeof(float));
	l.rolling_variance = calloc(n, sizeof(float));
	return l;
}

void normalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL && !l.batch_normalize) {
			net.layers[i] = normalize_layer(l, l.n);
		}
		if (l.type == CONNECTED && !l.batch_normalize) {
			net.layers[i] = normalize_layer(l, l.outputs);
		}
		if (l.type == GRU && l.batch_normalize) {
			*l.input_z_layer = normalize_layer(*l.input_z_layer, l.input_z_layer->outputs);
			*l.input_r_layer = normalize_layer(*l.input_r_layer, l.input_r_layer->outputs);
			*l.input_h_layer = normalize_layer(*l.input_h_layer, l.input_h_layer->outputs);
			*l.state_z_layer = normalize_layer(*l.state_z_layer, l.state_z_layer->outputs);
			*l.state_r_layer = normalize_layer(*l.state_r_layer, l.state_r_layer->outputs);
			*l.state_h_layer = normalize_layer(*l.state_h_layer, l.state_h_layer->outputs);
			net.layers[i].batch_normalize = 1;
		}
	}
	save_weights(net, outfile);
}

void statistics_net(char *cfgfile, char *weightfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONNECTED && l.batch_normalize) {
			printf("Connected Layer %d\n", i);
			statistics_connected_layer(l);
		}
		if (l.type == GRU && l.batch_normalize) {
			printf("GRU Layer %d\n", i);
			printf("Input Z\n");
			statistics_connected_layer(*l.input_z_layer);
			printf("Input R\n");
			statistics_connected_layer(*l.input_r_layer);
			printf("Input H\n");
			statistics_connected_layer(*l.input_h_layer);
			printf("State Z\n");
			statistics_connected_layer(*l.state_z_layer);
			printf("State R\n");
			statistics_connected_layer(*l.state_r_layer);
			printf("State H\n");
			statistics_connected_layer(*l.state_h_layer);
		}
		printf("\n");
	}
}

void denormalize_net(char *cfgfile, char *weightfile, char *outfile)
{
	gpu_index = -1;
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	int i;
	for (i = 0; i < net.n; ++i) {
		layer l = net.layers[i];
		if (l.type == CONVOLUTIONAL && l.batch_normalize) {
			denormalize_convolutional_layer(l);
			net.layers[i].batch_normalize = 0;
		}
		if (l.type == CONNECTED && l.batch_normalize) {
			denormalize_connected_layer(l);
			net.layers[i].batch_normalize = 0;
		}
		if (l.type == GRU && l.batch_normalize) {
			denormalize_connected_layer(*l.input_z_layer);
			denormalize_connected_layer(*l.input_r_layer);
			denormalize_connected_layer(*l.input_h_layer);
			denormalize_connected_layer(*l.state_z_layer);
			denormalize_connected_layer(*l.state_r_layer);
			denormalize_connected_layer(*l.state_h_layer);
			l.input_z_layer->batch_normalize = 0;
			l.input_r_layer->batch_normalize = 0;
			l.input_h_layer->batch_normalize = 0;
			l.state_z_layer->batch_normalize = 0;
			l.state_r_layer->batch_normalize = 0;
			l.state_h_layer->batch_normalize = 0;
			net.layers[i].batch_normalize = 0;
		}
	}
	save_weights(net, outfile);
}

void visualize(char *cfgfile, char *weightfile)
{
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	visualize_network(net);
#ifdef OPENCV
	cvWaitKey(0);
#endif
}

#ifdef ORIGIN_YOLO

int main(int argc, char **argv)
{
#ifdef _DEBUG
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	//test_resize("data/bad.jpg");
	//test_box();
	//test_convolutional_layer();
	if (argc < 2) {
		fprintf(stderr, "usage: %s <function>\n", argv[0]);
		return 0;
	}
	gpu_index = find_int_arg(argc, argv, "-i", 0);
	if (find_arg(argc, argv, "-nogpu")) {
		gpu_index = -1;
	}

#ifndef GPU
	gpu_index = -1;
#else
	if (gpu_index >= 0) {
		cuda_set_device(gpu_index);
	}
#endif

	if (0 == strcmp(argv[1], "average")) {
		average(argc, argv);
	}
	else if (0 == strcmp(argv[1], "yolo")) {
		run_yolo(argc, argv);
	}
	else if (0 == strcmp(argv[1], "voxel")) {
		run_voxel(argc, argv);
	}
	else if (0 == strcmp(argv[1], "super")) {
		run_super(argc, argv);
	}
	else if (0 == strcmp(argv[1], "detector")) {
		run_detector(argc, argv);
	}
	else if (0 == strcmp(argv[1], "detect")) {
		float thresh = find_float_arg(argc, argv, "-thresh", .24);
		int ext_output = find_arg(argc, argv, "-ext_output");
		char *filename = (argc > 4) ? argv[4] : 0;
		test_detector("cfg/coco.data", argv[2], argv[3], filename, thresh, ext_output);
	}
	else if (0 == strcmp(argv[1], "cifar")) {
		run_cifar(argc, argv);
	}
	else if (0 == strcmp(argv[1], "go")) {
		run_go(argc, argv);
	}
	else if (0 == strcmp(argv[1], "rnn")) {
		run_char_rnn(argc, argv);
	}
	else if (0 == strcmp(argv[1], "vid")) {
		run_vid_rnn(argc, argv);
	}
	else if (0 == strcmp(argv[1], "coco")) {
		run_coco(argc, argv);
	}
	else if (0 == strcmp(argv[1], "classify")) {
		predict_classifier("cfg/imagenet1k.data", argv[2], argv[3], argv[4], 5);
	}
	else if (0 == strcmp(argv[1], "classifier")) {
		run_classifier(argc, argv);
	}
	else if (0 == strcmp(argv[1], "art")) {
		run_art(argc, argv);
	}
	else if (0 == strcmp(argv[1], "tag")) {
		run_tag(argc, argv);
	}
	else if (0 == strcmp(argv[1], "compare")) {
		run_compare(argc, argv);
	}
	else if (0 == strcmp(argv[1], "dice")) {
		run_dice(argc, argv);
	}
	else if (0 == strcmp(argv[1], "writing")) {
		run_writing(argc, argv);
	}
	else if (0 == strcmp(argv[1], "3d")) {
		composite_3d(argv[2], argv[3], argv[4], (argc > 5) ? atof(argv[5]) : 0);
	}
	else if (0 == strcmp(argv[1], "test")) {
		test_resize(argv[2]);
	}
	else if (0 == strcmp(argv[1], "captcha")) {
		run_captcha(argc, argv);
	}
	else if (0 == strcmp(argv[1], "nightmare")) {
		run_nightmare(argc, argv);
	}
	else if (0 == strcmp(argv[1], "rgbgr")) {
		rgbgr_net(argv[2], argv[3], argv[4]);
	}
	else if (0 == strcmp(argv[1], "reset")) {
		reset_normalize_net(argv[2], argv[3], argv[4]);
	}
	else if (0 == strcmp(argv[1], "denormalize")) {
		denormalize_net(argv[2], argv[3], argv[4]);
	}
	else if (0 == strcmp(argv[1], "statistics")) {
		statistics_net(argv[2], argv[3]);
	}
	else if (0 == strcmp(argv[1], "normalize")) {
		normalize_net(argv[2], argv[3], argv[4]);
	}
	else if (0 == strcmp(argv[1], "rescale")) {
		rescale_net(argv[2], argv[3], argv[4]);
	}
	else if (0 == strcmp(argv[1], "ops")) {
		operations(argv[2]);
	}
	else if (0 == strcmp(argv[1], "speed")) {
		speed(argv[2], (argc > 3 && argv[3]) ? atoi(argv[3]) : 0);
	}
	else if (0 == strcmp(argv[1], "oneoff")) {
		oneoff(argv[2], argv[3], argv[4]);
	}
	else if (0 == strcmp(argv[1], "partial")) {
		partial(argv[2], argv[3], argv[4], atoi(argv[5]));
	}
	else if (0 == strcmp(argv[1], "average")) {
		average(argc, argv);
	}
	else if (0 == strcmp(argv[1], "visualize")) {
		visualize(argv[2], (argc > 3) ? argv[3] : 0);
	}
	else if (0 == strcmp(argv[1], "imtest")) {
		test_resize(argv[2]);
	}
	else {
		fprintf(stderr, "Not an option: %s\n", argv[1]);
	}
	int avxcxzd = 3;
	return 0;
}

#endif // ORIGIN_YOLO


/*
#include "FPS.h"

#define FRAMES 3
static network net;
static int cpp_video_capture = 0;
static CvCapture *cap;
static float *avg;
static float *predictions[FRAMES];
static image images[FRAMES];
static box *boxes;
static float **probs;
IplImage* in_img;
IplImage* det_img;
IplImage* show_img;
static image in_s;
static image det_s;
static int demo_index = 0;
//static float demo_thresh = 0;
//static float fps = 0;
static IplImage* ipl_images[FRAMES];
//static image **demo_alphabet;

void img_to_IplImage(image img_, IplImage* Ipl_img_);
void IplImage_to_img(image img_, IplImage* Ipl_img_);
//image ipl_to_image2(IplImage* src);

int main(int argc, char **argv)
{
// cam으로 열기
#ifndef GPU
gpu_index = -1;
#else
if (gpu_index >= 0) {
cuda_set_device(gpu_index);
}
#endif

char* cfg = "x64/cfg/yolov3.cfg";
char* weights = "x64/weights/yolov3.weights";
float thresh = 0.25;
float hier_thresh = 0.5;
int cam_index = 0;
int classes = 80;
char **names = "x64/data/coco.names";//"x64/data/names.list";	//다시보기. 안쓸듯.
char *filename = 0;	//비디오파일열때
int frame_skip = 0;
char* prefix = 0;
char *out_filename = 0;
int http_stream_port = -1;
int dont_show = 0;	// 1이면 Dont show인듯.
//int num_of_cluster = 5;

net = parse_network_cfg_custom(cfg, 1);	// set batch=1
if (weights) {
load_weights(&net, weights);
}

fuse_conv_batchnorm(net);
srand(2222222);

printf("Webcam index: %d\n", cam_index);
cpp_video_capture = 1;
//cap = get_capture_webcam(cam_index);
cap = cvCaptureFromCAM(0);
//if (!cap) error("Couldn't connect to webcam.\n");

layer l = net.layers[net.n - 1];

int j;

avg = (float *)calloc(l.outputs, sizeof(float));
for (j = 0; j < FRAMES; ++j) predictions[j] = (float *)calloc(l.outputs, sizeof(float));
for (j = 0; j < FRAMES; ++j) images[j] = make_image(1, 1, 3);

boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));



IplImage* src;
// seojong_1.avi
// ochang_85to15m_20seq.avi
// ochang_exposure_75to15m_5seq.avi
//cap = cvCaptureFromAVI("x64/data/videos/seojong_1.avi");
IplImage* new_img;
while (1)
{
FPS_start();

//src = cvQueryFrame(cap);	// src는 여기서 자동으로 releaseimage하나..?
cvGrabFrame(cap);
src = cvRetrieveFrame(cap, 0);


new_img = cvCreateImage(cvSize(net.w, net.h), IPL_DEPTH_8U, 3);
in_img = cvCreateImage(cvSize(src->width, src->height), IPL_DEPTH_8U, 3);
cvResize(src, in_img, CV_INTER_LINEAR);
cvResize(src, new_img, CV_INTER_LINEAR);

image out = make_image(new_img->width, new_img->height, new_img->nChannels);
IplImage_to_img(out, new_img);
in_s = out;
cvReleaseImage(&new_img);

///fetch_in_thread(0);
///det_img = in_img;
///det_s = in_s;

///fetch_in_thread(0);
///detect_in_thread(0);
float nms = .45;	// 0.4F

///////////////////////////////////////////////////////
//////////// 여기 아래에서 Net 불러오는 것 ////////////
///////////////////////////////////////////////////////
//layer l = net.layers[net.n - 1];	// 위에서 선언함
float *X = det_s.data;
float *prediction = network_predict(net, X);

memcpy(predictions[demo_index], prediction, l.outputs * sizeof(float));
mean_arrays(predictions, FRAMES, l.outputs, avg);
l.output = avg;

free_image(det_s);

det_img = in_img;
det_s = in_s;

int letter = 0;
int nboxes = 0;

///////////////////////////////////////////////////////
////////// 여기서 이미지 가져와서 Detection ///////////
///////////////////////////////////////////////////////
detection *dets = get_network_boxes(&net, det_s.w, det_s.h, thresh, 0.05, 0, 1, &nboxes, letter); // hier_thresh에 원래 thresh였음
if (nms) do_nms_obj(dets, nboxes, l.classes, nms);
//if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
///box *bb;
///bb->x = 0;
///bb->y = 0;
///bb->w = 0;
///bb->h = 0;
///if (nms) do_nms(bb, probs, nboxes, classes, nms);
//printf("\033[2J");
//printf("\033[1;1H");
//printf("\nFPS:%.1f\n", fps);
//printf("Objects:\n\n");

//ipl_images[demo_index] = det_img;
//det_img = ipl_images[(demo_index + FRAMES / 2 + 1) % FRAMES];
//demo_index = (demo_index + 1) % FRAMES;

for (int i = 0; i < nboxes; i++)
{
for (int j = 0; j < classes; j++)
{
if (dets[i].prob[j] > thresh)
{
box b = dets[i].bbox;
int left = (b.x - b.w / 2.)*det_img->width;
int right = (b.x + b.w / 2.)*det_img->width;
int top = (b.y - b.h / 2.)*det_img->height;
int bot = (b.y + b.h / 2.)*det_img->height;

if (left < 0) left = 0;
if (right > det_img->width - 1) right = det_img->width - 1;
if (top < 0) top = 0;
if (bot > det_img->height - 1) bot = det_img->height - 1;

CvPoint pt1, pt2;
pt1.x = left;
pt1.y = top;
pt2.x = right;
pt2.y = bot;
CvScalar color;
color.val[0] = 0;
color.val[1] = 0;
color.val[2] = 255;
int width_ = 1;// det_img->height * .006;

cvRectangle(det_img, pt1, pt2, color, width_, 8, 0);
}
}
}

free_detections(dets, nboxes);

cvShowImage("Demo", det_img);
if (cvWaitKey(1) == 'q') break;
printf("FPS : %f\n", FPS_finish());

//cvReleaseImage(&new_img);
//cvReleaseImage(&in_img);
cvReleaseImage(&det_img);
//cvReleaseImage(&src);
}
return 0;
}


void img_to_IplImage(image img_, IplImage* Ipl_img_)
{
int k, i, j, count = 0;
for (k = img_.c - 1; k > -1; --k) {///for (k = 0; k < img_.c; ++k) {
for (i = 0; i < img_.h; ++i) {
for (j = 0; j < img_.w; ++j) {
Ipl_img_->imageData[i*Ipl_img_->widthStep + j*net.c + k] = img_.data[count++] * 255.;
//img_.data[count++] = Ipl_img_->imageData[i*Ipl_img_->widthStep + j*net.c + k] / 255.; // 원래는 data[i*step + j*c + k]/255.; 임
}
}
}
}

void IplImage_to_img(image img_, IplImage* Ipl_img_)
{


int h = Ipl_img_->height;
int w = Ipl_img_->width;
int c = Ipl_img_->nChannels;
int step = Ipl_img_->widthStep;

int i, j, k, count = 0;;

for (k = 0; k < c; ++k) {
for (i = 0; i < h; ++i) {
for (j = 0; j < w; ++j) {
img_.data[count++] = (unsigned char)Ipl_img_->imageData[i*step + j*c + k] / 255.; // 원래는 data[i*step + j*c + k]/255.; 임
}
}
}


rgbgr_image(img_);
}

*/


















////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////

//#define DETECTION_CHP

#ifdef DETECTION_CHP
#include "FPS.h"
//#include "utils_p.cuh"

//void mean_arrays(float **a, int n, int els, float *avg);

#define FRAMES 3
static network net;
static int cpp_video_capture = 0;
static CvCapture *cap;
static float *avg;
static float *predictions[FRAMES];
static image images[FRAMES];
static box *boxes;
static float **probs;
IplImage* in_img;
IplImage* det_img;
IplImage* show_img;
static image in_s;
static image det_s;
static int demo_index = 0;
//static float demo_thresh = 0;
//static float fps = 0;
static IplImage* ipl_images[FRAMES];
//static image **demo_alphabet;

void img_to_IplImage(image img_, IplImage* Ipl_img_);
void IplImage_to_img(image img_, IplImage* Ipl_img_);
void RED_text_to_img(IplImage *src_);
void YELLOW_text_to_img(IplImage *src_);
void GREEN_text_to_img(IplImage *src_);
void GREEN_L_ARROW_text_to_img(IplImage *src_);
int sel_biggest_one(float *array_, float count_);
//image ipl_to_image2(IplImage* src);

int main(int argc, char **argv)
{
	// cam으로 열기
#ifndef GPU
	gpu_index = -1;
#else
	if (gpu_index >= 0) {
		cuda_set_device(gpu_index);
	}
#endif

	//char* cfg = "x64/cfg/yolov3-tiny-obj-test.cfg";	// yolo v3 tiny
	char* cfg = "x64/cfg/yolo-v3-obj-test.cfg";	// yolo v3
	//char* weights = "x64/weights/yolo-v3-tiny-obj-train_151424.weights"; //yolo-v3-obj-train_113344.weights(yolov3 best) //yolov3.weights //yolo-v3-obj-train_230048_ROI.weights //yolo-v3-obj-train_130368_ROI.weights
	char* weights = "x64/weights/yolo-v3-obj-train_113344.weights"; //yolo-v3-obj-train_113344.weights(yolov3 best) //yolov3.weights //yolo-v3-obj-train_230048_ROI.weights //yolo-v3-obj-train_130368_ROI.weights
	float thresh = 0.25;//0.25;
	float hier_thresh = 0.5;//0.5;
	int cam_index = 0;
	int classes = 10;// 80;
	char **names = "x64/data/coco.names";//"x64/data/names.list";	//다시보기. 안쓸듯.
	char *filename = 0;	//비디오파일열때
	int frame_skip = 0;
	char* prefix = 0;
	char *out_filename = "x64/yolo_v3_113344weights_final.avi";// 0;
	int http_stream_port = -1;
	int dont_show = 0;	// 1이면 Dont show인듯.
						//int num_of_cluster = 5;

	net = parse_network_cfg_custom(cfg, 1);	// set batch=1
	if (weights) {
		load_weights(&net, weights);		// 여기서 (약 177ms)
	}

	fuse_conv_batchnorm(net);
	srand(2222222);

	printf("Webcam index: %d\n", cam_index);
	cpp_video_capture = 1;
	//cap = get_capture_webcam(cam_index);
	cap = cvCaptureFromCAM(0);				// 여기서 (약 528ms)
											//if (!cap) error("Couldn't connect to webcam.\n");

	layer l = net.layers[net.n - 1];

	int j;

	avg = (float *)calloc(l.outputs, sizeof(float));
	for (j = 0; j < FRAMES; ++j) predictions[j] = (float *)calloc(l.outputs, sizeof(float));
	for (j = 0; j < FRAMES; ++j) images[j] = make_image(1, 1, 3);

	boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float *)calloc(l.classes, sizeof(float *));



	IplImage* src;
	// seojong_1.avi
	// ochang_85to15m_20seq.avi
	// ochang_exposure_75to15m_5seq.avi
	//cap = cvCaptureFromAVI("..//..//..//..//..//..//..//Sekonix_DB//ocam_WB_ochang_sunlight_1.avi"); // 오창 신호등
	//cap = cvCaptureFromAVI("..//..//..//..//..//..//..//Sekonix_DB//ocam_WB_ochang_1.avi"); // 오창 신호등
	//cap = cvCaptureFromAVI("..//..//..//..//..//..//..//Sekonix_DB//20180709_cheongju-rainy.avi"); // 오창 신호등
	//cap = cvCaptureFromAVI("..//..//..//..//..//..//..//Sekonix_DB//ochang//ochang.avi"); // 오창 신호등
	//cap = cvCaptureFromAVI("..//..//..//..//..//..//..//Sekonix_DB//cheongju//cheongju.avi"); // 오창 신호등																							 
	cap = cvCaptureFromAVI("x64/data/videos/seojong_1.avi");	// 세종시 신호등
	
	// 비디오 저장 목적으로 생성
	CvVideoWriter *writer = NULL;

	IplImage* new_img;
	while (1)
	{
		FPS_start();

		if (!(src = cvQueryFrame(cap)))	break;
									// src는 여기서 자동으로 releaseimage하나..?
									//cvGrabFrame(cap);
									//src = cvRetrieveFrame(cap, 0);
		
		new_img = cvCreateImage(cvSize(net.w, net.h), IPL_DEPTH_8U, 3);
		cvResize(src, new_img, CV_INTER_LINEAR);

		image out = make_image(new_img->width, new_img->height, new_img->nChannels);
		IplImage_to_img(out, new_img);
		cvReleaseImage(&new_img);

		float nms = .45;	// 0.4F

							///////////////////////////////////////////////////////
							//////////// 여기 아래에서 Net 불러오는 것 ////////////
							///////////////////////////////////////////////////////	

		layer l = net.layers[net.n - 1];	// 위에서 선언함
		float *prediction = network_predict(net, det_s.data);	// 여기서 대부분 시간 빼먹음 (약 150ms)

		memcpy(predictions[demo_index], prediction, l.outputs * sizeof(float));
		mean_arrays(predictions, FRAMES, l.outputs, avg);		// 여기서 (1)의 1/10 먹음 (약 10ms)
		l.output = avg;

		free_image(det_s);

		det_s = out;// in_s;

		int letter = 0;
		int nboxes = 0;

		///////////////////////////////////////////////////////
		////////// 여기서 이미지 가져와서 Detection ///////////
		///////////////////////////////////////////////////////
		detection *dets = get_network_boxes(&net, out.w, out.h, thresh, hier_thresh, 0, 1, &nboxes, letter); // hier_thresh에 원래 thresh였음
																											 //cudaThreadSynchronize();
		//if (nms) do_nms_obj(dets, nboxes, l.classes, nms);
		//if (nms) do_nms_sort(dets, nboxes, l.classes, nms);
		///box *bb;
		///bb->x = 0;
		///bb->y = 0;
		///bb->w = 0;
		///bb->h = 0;
		///if (nms) do_nms(bb, probs, nboxes, classes, nms);

		/*
			1)
			classes = 4


			2)
			dets 재정의	-	dets[0 ~ nboxes-1].
							dets[0 ~ nboxes-1].classes = 4
							dets[0 ~ nboxes-1].prob[0~10]을 .prob[0~3]으로
								// 0,4 비교 -> 0으로	0 : RED
								// 1, 5비교 -> 1로		1 : YELLOW
								// 2, 6비교 -> 2로		2 : GREEN
								// 7 유지	-> 3로		3 : GREEN / L_ARROW
								// 3, 8, 9 삭제
								// dets[].prob[0] = dets[].prob[0]>dets[].prob[4] ? dets[].prob[0]:dets[].prob[4];
								// dets[].prob[1] = dets[].prob[1]>dets[].prob[5] ? dets[].prob[1]:dets[].prob[5];
								// dets[].prob[2] = dets[].prob[2]>dets[].prob[6] ? dets[].prob[2]:dets[].prob[6];
								// dets[].prob[3] = dets[].prob[7];
								// dets[].prob[4 ~ 9] = 0;


			3)
				dets[].prob[0~3] 비교해서 가장 큰 애 찾기 (x라 가정)


			4)
				dets[].prob[x]의 색상을 cv::putText() 로 글 넣기
					RED_text_to_img(src);
					YELLOW_text_to_img(src);
					GREEN_text_to_img(src);
			
			5)
				putText(image_hough, whatw, Point(0, 100), 1, 1, (0, 0, 255));//FONT_HERSHEY_SCRIPT_SIMPLEX
		*/

		// dets[i].classes	- 클래스 총 개수
		// dets[i].prob[x]	- x번째 class일 확률 (0부터 시작)
		// nboxes			- 찾은 박스 개수

		float *roi_size = 0;	// 가장 큰 ROI를 갖는 object를 찾기 위한 변수
		roi_size = (float*)malloc(nboxes * sizeof(float));

		//printf("nboxes = %d\n", nboxes);
		for (int i = 0; i < nboxes; i++)
		{
			// 클래스 합치기 ( 2 과정 )
			dets[i].classes = 4;
			dets[i].prob[0] = dets[i].prob[0]>dets[i].prob[4] ? dets[i].prob[0]:dets[i].prob[4];
			dets[i].prob[1] = dets[i].prob[1]>dets[i].prob[5] ? dets[i].prob[1]:dets[i].prob[5];
			dets[i].prob[2] = dets[i].prob[2]>dets[i].prob[6] ? dets[i].prob[2]:dets[i].prob[6];
			dets[i].prob[3] = dets[i].prob[7];
			dets[i].prob[4] = 0;
			dets[i].prob[5] = 0;
			dets[i].prob[6] = 0;
			dets[i].prob[7] = 0;
			dets[i].prob[8] = 0;
			dets[i].prob[9] = 0;

			/*
					1. dets[].prob[0~3] 비교해서 가장 큰 애 찾기(x라 가정)
					2. dets[].prob[x]의 색상을 cv::putText() 로 글 넣기
						RED_text_to_img(src);
						YELLOW_text_to_img(src);
						GREEN_text_to_img(src);
			*/

			
			// 가장 큰 ROI를 갖는 object를 찾는다.
			roi_size[i] = dets[i].bbox.h * dets[i].bbox.w;
			//printf("dets[%d].bbox.h * dets[%d].bbox.w = %f\n", i, i, roi_size[i]);
			//for(int j = 0; j < classes; j++)
			//	printf("prob[%d] = %f\n", j, dets[i].prob[j]);
		}
		
		//printf("%d\n", biggest_roi_size_num);
		
		// ROI 그리기
		if (nboxes > 0)
		{
			int biggest_roi_size_num = sel_biggest_one(roi_size, nboxes);
			for (int j = 0; j < classes; j++)
			{
				// 가장 큰 ROI를 고르기
				int biggest_roi_size_num = sel_biggest_one(roi_size, nboxes);

				if (dets[biggest_roi_size_num].prob[j] > thresh)
				{
					// 아래는 텍스트 넣기
					int class_ = sel_biggest_one(dets[biggest_roi_size_num].prob, 4);
					printf("prob = %f\n", dets[biggest_roi_size_num].prob[class_]);
					//printf("%d\n", class_);
					switch (class_)
					{
					case 0:
						RED_text_to_img(src);
						break;
					case 1:
						YELLOW_text_to_img(src);
						break;
					case 2:
						GREEN_text_to_img(src);
						break;
					case 3:
						GREEN_L_ARROW_text_to_img(src);
						break;
					}

					// 아래는 박스 그리기
					box b = dets[biggest_roi_size_num].bbox;
					int left = (b.x - b.w / 2.)*src->width;
					int right = (b.x + b.w / 2.)*src->width;
					int top = (b.y - b.h / 2.)*src->height;
					int bot = (b.y + b.h / 2.)*src->height;

					if (left < 0) left = 0;
					if (right > src->width - 1) right = src->width - 1;
					if (top < 0) top = 0;
					if (bot > src->height - 1) bot = src->height - 1;

					CvPoint pt1, pt2;
					pt1.x = left;
					pt1.y = top;
					pt2.x = right;
					pt2.y = bot;
					CvScalar color;
					color.val[0] = 0;
					color.val[1] = 0;
					color.val[2] = 255;
					int width_ = 1;// src->height * .006;

					cvRectangle(src, pt1, pt2, color, width_, 8, 0);
				}
			}
		}

		//for (int i = 0; i < nboxes; i++)
		//{
		//	for (int j = 0; j < classes; j++)
		//	{
		//		if (dets[i].prob[j] > thresh)
		//		{
		//			int class_ = sel_biggest_one(dets[i].prob, 4);	// 얘 잘못만듬
		//			//printf("%d\n", class_);
		//			switch (class_)
		//			{
		//			case 0:
		//				RED_text_to_img(src);
		//				break;
		//			case 1:
		//				YELLOW_text_to_img(src);
		//				break;
		//			case 2:
		//				GREEN_text_to_img(src);
		//				break;
		//			case 3:
		//				GREEN_L_ARROW_text_to_img(src);
		//				break;

		//			}

		//			box b = dets[i].bbox;
		//			int left = (b.x - b.w / 2.)*src->width;
		//			int right = (b.x + b.w / 2.)*src->width;
		//			int top = (b.y - b.h / 2.)*src->height;
		//			int bot = (b.y + b.h / 2.)*src->height;

		//			if (left < 0) left = 0;
		//			if (right > src->width - 1) right = src->width - 1;
		//			if (top < 0) top = 0;
		//			if (bot > src->height - 1) bot = src->height - 1;

		//			CvPoint pt1, pt2;
		//			pt1.x = left;
		//			pt1.y = top;
		//			pt2.x = right;
		//			pt2.y = bot;
		//			CvScalar color;
		//			color.val[0] = 0;
		//			color.val[1] = 0;
		//			color.val[2] = 255;
		//			int width_ = 1;// src->height * .006;

		//			cvRectangle(src, pt1, pt2, color, width_, 8, 0);
		//		}
		//	}
		//}

		free_detections(dets, nboxes);

		//cvSize src_size; (src->width, src->height);

		// 비디오 저장
		if (out_filename != 0)
		{
			printf("%c\n", out_filename);
			CvSize src_size;
			src_size.width = src->width;
			src_size.height = src->height;

			int fps_write = (int)cvGetCaptureProperty(cap, CV_CAP_PROP_FPS);

			// 처음 1번만 실행시킨다는 뜻
			if(writer == NULL)
				writer = cvCreateVideoWriter(out_filename, CV_FOURCC('D', 'I', 'V', 'X'), fps_write, src_size, 1);

			cvWriteFrame(writer, src);
			//cvWaitKey(1);

			
		}

		cvShowImage("Demo", src);
		if (cvWaitKey(1) == 'q') break;				// 여기서 약 15ms
		printf("FPS : %f\n", FPS_finish());

		//cvReleaseImage(&new_img);
		//cvReleaseImage(&in_img);
		//cvReleaseImage(&det_img);
		//cvReleaseImage(&src);
		
		free(roi_size);

	}

	//저장 비디오 프레임의 해제
	cvReleaseVideoWriter(&writer);
	//cvReleaseImage(&src);
	free(boxes);
	return 0;
}





#endif // DETECTION_CHP

//버블정렬 함수
int sel_biggest_one(float *array_, float count_)
{
	//int tmp = array_[0];
	int tmp = 0;
	for (int i = 0; i < count_; i++)
	{
		tmp = (array_[tmp] > array_[i]) ? tmp:i;
	}
	return tmp;
}

void RED_text_to_img(IplImage *src_)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_VECTOR0, 3, 3, 0, 3, 0);
	cvPutText(src_, "RED", cvPoint(src_->width - 400, 100), &font, CV_RGB(253, 61, 38));
}
void YELLOW_text_to_img(IplImage *src_)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_VECTOR0, 3, 3, 0, 3, 0);
	cvPutText(src_, "YELLOW", cvPoint(src_->width - 400, 100), &font, CV_RGB(249, 220, 70));
}
void GREEN_text_to_img(IplImage *src_)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_VECTOR0, 3, 3, 0, 3, 0);
	cvPutText(src_, "GREEN", cvPoint(src_->width - 400, 100), &font, CV_RGB(28, 227, 118));
}
void GREEN_L_ARROW_text_to_img(IplImage *src_)
{
	CvFont font;
	cvInitFont(&font, CV_FONT_VECTOR0, 3, 3, 0, 3, 0);
	cvPutText(src_, "GREEN_LEFT", cvPoint(src_->width - 800, 100), &font, CV_RGB(28, 227, 118));
}

void img_to_IplImage(image img_, IplImage* Ipl_img_)
{
	int k, i, j, count = 0;
	for (k = img_.c - 1; k > -1; --k) {///for (k = 0; k < img_.c; ++k) {
		for (i = 0; i < img_.h; ++i) {
			for (j = 0; j < img_.w; ++j) {
				Ipl_img_->imageData[i*Ipl_img_->widthStep + j*img_.c + k] = img_.data[count++] * 255.;
				//img_.data[count++] = Ipl_img_->imageData[i*Ipl_img_->widthStep + j*net.c + k] / 255.; // 원래는 data[i*step + j*c + k]/255.; 임
			}
		}
	}
}

void IplImage_to_img(image img_, IplImage* Ipl_img_)
{


	int h = Ipl_img_->height;
	int w = Ipl_img_->width;
	int c = Ipl_img_->nChannels;
	int step = Ipl_img_->widthStep;

	int i, j, k, count = 0;;

	for (k = 0; k < c; ++k) {
		for (i = 0; i < h; ++i) {
			for (j = 0; j < w; ++j) {
				img_.data[count++] = (unsigned char)Ipl_img_->imageData[i*step + j*c + k] / 255.; // 원래는 data[i*step + j*c + k]/255.; 임
			}
		}
	}


	rgbgr_image(img_);
}

