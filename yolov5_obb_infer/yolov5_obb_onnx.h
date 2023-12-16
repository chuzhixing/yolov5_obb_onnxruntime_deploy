#ifndef YOLOV5_OBB_H_
#define YOLOV5_OBB_H_

#define _USE_MATH_DEFINES

#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include "yolov8_utils.h"
#include<onnxruntime_cxx_api.h>

struct OBB {
    float cx;
    float cy;
    float longside;
    float shortside;
    float theta_pred;
    float max_class_score;
    int class_idx;
};

struct RotatedBox {
    cv::Point2f center;
    float w;
    float h;
    float theta;
};

struct OutputSegObb {
    int id;             //结果类别id
    float confidence;   //结果置信度
    cv::RotatedRect box; //矩形框
};

class Yolov5ObbOnnx {
public:
	Yolov5ObbOnnx() :_OrtMemoryInfo(Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtDeviceAllocator, OrtMemType::OrtMemTypeCPUOutput)) {};
	~Yolov5ObbOnnx() {};// delete _OrtMemoryInfo;

public:
	/** \brief Read onnx-model
	* \param[in] modelPath:onnx-model path
	* \param[in] isCuda:if true,use Ort-GPU,else run it on cpu.
	* \param[in] cudaID:if isCuda==true,run Ort-GPU on cudaID.
	* \param[in] warmUp:if isCuda==true,warm up GPU-model.
	*/
	bool ReadModel(const std::string& modelPath, bool isCuda = false, int cudaID = 0, bool warmUp = true);

	/** \brief  detect.
	* \param[in] srcImg:a 3-channels image.
	* \param[out] output:detection results of input image.
	*/
	bool OnnxDetect(cv::Mat& srcImg, std::vector<OutputSegObb>& output);
	/** \brief  detect,batch size= _batchSize
	* \param[in] srcImg:A batch of images.
	* \param[out] output:detection results of input images.
	*/
	bool OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSegObb>>& output);

	void set_net_w_h(int net_w, int net_h);

private:
	template <typename T>
	T VectorProduct(const std::vector<T>& v) {
		return std::accumulate(v.begin(), v.end(), 1, std::multiplies<T>());
	};
	int Preprocessing(const std::vector<cv::Mat>& SrcImgs, std::vector<cv::Mat>& OutSrcImgs, std::vector<cv::Vec4d>& params);
	
	void rbox2poly(const std::vector<OBB>& obboxes, cv::Mat& polys);
	bool scale_polys(cv::Mat& polys, cv::Vec4d& params);
	bool polys_2_OutputSegObbs(cv::Mat polys, std::vector<OutputSegObb>& out_segs);

	int _netWidth = 1024;   //ONNX-net-input-width
	int _netHeight = 1024;  //ONNX-net-input-height

	int _batchSize = 1;  //if multi-batch,set this
	bool _isDynamicShape = false;//onnx support dynamic shape

	//ONNXRUNTIME	
	Ort::Env _OrtEnv = Ort::Env(OrtLoggingLevel::ORT_LOGGING_LEVEL_ERROR, "Yolov5-Seg");
	Ort::SessionOptions _OrtSessionOptions = Ort::SessionOptions();
	Ort::Session* _OrtSession = nullptr;
	Ort::MemoryInfo _OrtMemoryInfo;
#if ORT_API_VERSION < ORT_OLD_VISON
	char* _inputName, * _output_name0;
#else
	std::shared_ptr<char> _inputName, _output_name0;
#endif

	std::vector<char*> _inputNodeNames; //输入节点名
	std::vector<char*> _outputNodeNames;//输出节点名

	size_t _inputNodesNum = 0;        //输入节点数
	size_t _outputNodesNum = 0;       //输出节点数

	ONNXTensorElementDataType _inputNodeDataType; //数据类型
	ONNXTensorElementDataType _outputNodeDataType;
	std::vector<int64_t> _inputTensorShape; //输入张量shape

	std::vector<int64_t> _outputTensorShape;

public:	
	std::vector<std::string> _className = { "h0", "h1"};

	float _classThreshold = 0.25;
	float _nmsThreshold = 0.20;
	float _maskThreshold = 0.5;

};


#endif // YOLOV5_OBB_H_