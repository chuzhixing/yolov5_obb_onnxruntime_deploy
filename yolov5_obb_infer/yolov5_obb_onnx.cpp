#include "yolov5_obb_onnx.h"

using namespace std;
using namespace cv;
using namespace cv::dnn;
using namespace Ort;


void Yolov5ObbOnnx::set_net_w_h(int net_w, int net_h) {
	this->_netWidth = net_w;
	this->_netHeight = net_h;
}

bool Yolov5ObbOnnx::ReadModel(const std::string& modelPath, bool isCuda, int cudaID, bool warmUp) {
	if (_batchSize < 1) _batchSize = 1;
	try {
		if (!CheckModelPath(modelPath))
			return false;
		std::vector<std::string> available_providers = GetAvailableProviders();
		auto cuda_available = std::find(available_providers.begin(), available_providers.end(), "CUDAExecutionProvider");


		if (isCuda && (cuda_available == available_providers.end())) {
			std::cout << "Your ORT build without GPU. Change to CPU." << std::endl;
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		} else if (isCuda && (cuda_available != available_providers.end())) {
			std::cout << "************* Infer model on GPU! *************" << std::endl;
#if ORT_API_VERSION < ORT_OLD_VISON
			OrtCUDAProviderOptions cudaOption;
			cudaOption.device_id = cudaID;
			_OrtSessionOptions.AppendExecutionProvider_CUDA(cudaOption);
#else
			OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(_OrtSessionOptions, cudaID);
#endif
		} else {
			std::cout << "************* Infer model on CPU! *************" << std::endl;
		}
		//

		_OrtSessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);

#ifdef _WIN32
		std::wstring model_path(modelPath.begin(), modelPath.end());
		_OrtSession = new Ort::Session(_OrtEnv, model_path.c_str(), _OrtSessionOptions);
#else
		_OrtSession = new Ort::Session(_OrtEnv, modelPath.c_str(), _OrtSessionOptions);
#endif

		Ort::AllocatorWithDefaultOptions allocator;
		//init input
		_inputNodesNum = _OrtSession->GetInputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_inputName = _OrtSession->GetInputName(0, allocator);
		_inputNodeNames.push_back(_inputName);
#else
		_inputName = std::move(_OrtSession->GetInputNameAllocated(0, allocator));
		_inputNodeNames.push_back(_inputName.get());
#endif
		//cout << _inputNodeNames[0] << endl;
		Ort::TypeInfo inputTypeInfo = _OrtSession->GetInputTypeInfo(0);
		auto input_tensor_info = inputTypeInfo.GetTensorTypeAndShapeInfo();
		_inputNodeDataType = input_tensor_info.GetElementType();
		_inputTensorShape = input_tensor_info.GetShape();

		if (_inputTensorShape[0] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[0] = _batchSize;

		}
		if (_inputTensorShape[2] == -1 || _inputTensorShape[3] == -1) {
			_isDynamicShape = true;
			_inputTensorShape[2] = _netHeight;
			_inputTensorShape[3] = _netWidth;
		}
		//init output
		_outputNodesNum = _OrtSession->GetOutputCount();
#if ORT_API_VERSION < ORT_OLD_VISON
		_output_name0 = _OrtSession->GetOutputName(0, allocator);
		_outputNodeNames.push_back(_output_name0);
#else
		_output_name0 = std::move(_OrtSession->GetOutputNameAllocated(0, allocator));
		_outputNodeNames.push_back(_output_name0.get());
#endif
		Ort::TypeInfo type_info_output0(nullptr);
		type_info_output0 = _OrtSession->GetOutputTypeInfo(0);  //output0

		auto tensor_info_output0 = type_info_output0.GetTensorTypeAndShapeInfo();
		_outputNodeDataType = tensor_info_output0.GetElementType();
		_outputTensorShape = tensor_info_output0.GetShape();


		//warm up
		if (isCuda && warmUp) {
			//draw run
			cout << "Start warming up" << endl;
			size_t input_tensor_length = VectorProduct(_inputTensorShape);
			float* temp = new float[input_tensor_length];
			std::vector<Ort::Value> input_tensors;
			std::vector<Ort::Value> output_tensors;
			input_tensors.push_back(Ort::Value::CreateTensor<float>(
				_OrtMemoryInfo, temp, input_tensor_length, _inputTensorShape.data(),
				_inputTensorShape.size()));
			for (int i = 0; i < 3; ++i) {
				output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
					_inputNodeNames.data(),
					input_tensors.data(),
					_inputNodeNames.size(),
					_outputNodeNames.data(),
					_outputNodeNames.size());
			}

			delete[]temp;
		}
	} catch (const std::exception&) {
		return false;
	}
	return true;
}


int Yolov5ObbOnnx::Preprocessing(const std::vector<cv::Mat>& srcImgs, std::vector<cv::Mat>& outSrcImgs, std::vector<cv::Vec4d>& params) {
	outSrcImgs.clear();
	Size input_size = Size(_netWidth, _netHeight);
	for (int i = 0; i < srcImgs.size(); ++i) {
		Mat temp_img = srcImgs[i];
		Vec4d temp_param = { 1,1,0,0 };
		if (temp_img.size() != input_size) {
			Mat borderImg;
			LetterBox(temp_img, borderImg, temp_param, input_size, false, false, true, 32);
			//cout << borderImg.size() << endl;
			outSrcImgs.push_back(borderImg);
			params.push_back(temp_param);
		} else {
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}

	int lack_num = _batchSize - srcImgs.size();
	if (lack_num > 0) {
		for (int i = 0; i < lack_num; ++i) {
			Mat temp_img = Mat::zeros(input_size, CV_8UC3);
			Vec4d temp_param = { 1,1,0,0 };
			outSrcImgs.push_back(temp_img);
			params.push_back(temp_param);
		}
	}
	return 0;

}

void Yolov5ObbOnnx::rbox2poly(const std::vector<OBB>& obboxes, cv::Mat& polys) {	
	for (size_t i = 0; i < obboxes.size(); ++i) {
		float cx = obboxes[i].cx;
		float cy = obboxes[i].cy;
		float w = obboxes[i].longside;
		float h = obboxes[i].shortside;
		float theta = obboxes[i].theta_pred;
		float conf = obboxes[i].max_class_score;
		int cls_id = obboxes[i].class_idx;

		double Cos = std::cos(theta);
		double Sin = std::sin(theta);

		cv::Point2d vector1(w / 2 * Cos, -w / 2 * Sin);
		cv::Point2d vector2(-h / 2 * Sin, -h / 2 * Cos);
		cv::Point2d point1 = cv::Point2d(cx, cy) + vector1 + vector2;
		cv::Point2d point2 = cv::Point2d(cx, cy) + vector1 - vector2;
		cv::Point2d point3 = cv::Point2d(cx, cy) - vector1 - vector2;
		cv::Point2d point4 = cv::Point2d(cx, cy) - vector1 + vector2;
		
		polys.ptr<float>(i)[0] = point1.x;
		polys.ptr<float>(i)[1] = point1.y;

		polys.ptr<float>(i)[2] = point2.x;
		polys.ptr<float>(i)[3] = point2.y;

		polys.ptr<float>(i)[4] = point3.x;
		polys.ptr<float>(i)[5] = point3.y;

		polys.ptr<float>(i)[6] = point4.x;
		polys.ptr<float>(i)[7] = point4.y;

		polys.ptr<float>(i)[8] = conf;
		polys.ptr<float>(i)[9] = cls_id;
	}
}

/**
 * \param pred_poly
 * \param scaled_poly
 * \param params
	params[0]: 模型输入宽度 / 原始输入图像宽度
	params[1]: 模型输入高度 / 原始输入图像高度
	params[2]: 以模型输入图像左上角为原点，原始图片resize后的左上角点的x坐标（即原始图片resize后，x方向的左侧padding宽度）
	params[3]: 以模型输入图像左上角为原点，原始图片resize后的左上角点的y坐标（即原始图片resize后，y方向的上侧padding的高度）
 * \return 
 */
bool Yolov5ObbOnnx::scale_polys(cv::Mat& polys, cv::Vec4d& params) {

	float r_w_in_to_raw = params[0];
	float r_h_in_to_raw = params[1];
	float left_padding = params[2];
	float top_padding = params[3];

	for (int i = 0; i < polys.rows; i++) {	
		// -padding原因: 把图像坐标系，模型输入图像的左上角做为坐标原点，切换成padding之前，resize之后的图像左上角做为坐标原点。
		// 即（left_padding, top_padding）点在模型模型输入图像坐标系中坐标是：（left_padding, top_padding）
		// 在padding之前，resize之后的图像坐标系的坐标是(0, 0)
		// /r原因：宽度与高度方向分别放缩到原始输入图片的大小
		polys.ptr<float>(i)[0] = (polys.ptr<float>(i)[0] - left_padding) / r_w_in_to_raw;
		polys.ptr<float>(i)[2] = (polys.ptr<float>(i)[2] - left_padding) / r_w_in_to_raw;
		polys.ptr<float>(i)[4] = (polys.ptr<float>(i)[4] - left_padding) / r_w_in_to_raw;
		polys.ptr<float>(i)[6] = (polys.ptr<float>(i)[6] - left_padding) / r_w_in_to_raw;

		polys.ptr<float>(i)[1] = (polys.ptr<float>(i)[1] - top_padding) / r_h_in_to_raw;
		polys.ptr<float>(i)[3] = (polys.ptr<float>(i)[3] - top_padding) / r_h_in_to_raw;
		polys.ptr<float>(i)[5] = (polys.ptr<float>(i)[5] - top_padding) / r_h_in_to_raw;
		polys.ptr<float>(i)[7] = (polys.ptr<float>(i)[7] - top_padding) / r_h_in_to_raw;
	}

	return true;
}

bool Yolov5ObbOnnx::polys_2_OutputSegObbs(cv::Mat polys, std::vector<OutputSegObb>& out_segs) {
	for (int i = 0; i < polys.rows; i++) {
		OutputSegObb out_seg;
		out_seg.id = polys.ptr<float>(i)[9];
		out_seg.confidence = polys.ptr<float>(i)[8];

		std::vector<cv::Point2f> polygon_list;
		for (int j = 0; j < 3; ++j) {
			cv::Point2f point(polys.ptr<float>(i)[j * 2], polys.ptr<float>(i)[j * 2 + 1]);
			polygon_list.push_back(point);
		}

		out_seg.box = cv::RotatedRect(polygon_list[0], polygon_list[1], polygon_list[2]);
		out_segs.emplace_back(out_seg);
	}

	return true;
}


bool Yolov5ObbOnnx::OnnxBatchDetect(std::vector<cv::Mat>& srcImgs, std::vector<std::vector<OutputSegObb>>& output) {
	auto t0 = std::chrono::high_resolution_clock::now();   //结束时间    
	std::vector<cv::Vec4d> params;
	std::vector<cv::Mat> input_images;
	cv::Size input_size(_netWidth, _netHeight);
	//preprocessing
	Preprocessing(srcImgs, input_images, params);
	cv::Mat blob = cv::dnn::blobFromImages(input_images, 1 / 255.0, input_size, Scalar(0, 0, 0), true, false);
	auto t1 = std::chrono::high_resolution_clock::now();   //结束时间

	int64_t input_tensor_length = VectorProduct(_inputTensorShape);
	std::vector<Ort::Value> input_tensors;
	std::vector<Ort::Value> output_tensors;
	input_tensors.push_back(Ort::Value::CreateTensor<float>(_OrtMemoryInfo, (float*)blob.data, input_tensor_length, _inputTensorShape.data(), _inputTensorShape.size()));

	output_tensors = _OrtSession->Run(Ort::RunOptions{ nullptr },
		_inputNodeNames.data(),
		input_tensors.data(),
		_inputNodeNames.size(),
		_outputNodeNames.data(),
		_outputNodeNames.size()
	);
	auto t2 = std::chrono::high_resolution_clock::now();   //结束时间

	if (output_tensors.size() < 1) {
		return false;
	}

	//post-process
	for (int img_i = 0; img_i < srcImgs.size(); ++img_i) {
		float* all_data = output_tensors[img_i].GetTensorMutableData<float>();
		_outputTensorShape = output_tensors[img_i].GetTensorTypeAndShapeInfo().GetShape();

		int nc = _outputTensorShape[2] - 5 - 180;
		cv::Mat out(_outputTensorShape[1], _outputTensorShape[2], CV_32F, all_data);

		std::vector<cv::RotatedRect> bboxes;
		std::vector<float> scores;
		std::vector<OBB> generate_boxes;

		for (int i = 0; i < out.rows; ++i) {
			float cx = out.at<float>(i, 0);
			float cy = out.at<float>(i, 1);
			float longside = out.at<float>(i, 2);
			float shortside = out.at<float>(i, 3);
			float obj_score = out.at<float>(i, 4);

			if (obj_score < this->_classThreshold)
				continue;

			cv::Mat class_scores = out.row(i).colRange(5, 5 + nc);
			class_scores *= obj_score;
			double minV, maxV;
			cv::Point minI, maxI;
			cv::minMaxLoc(class_scores, &minV, &maxV, &minI, &maxI);

			int class_idx = maxI.x;
			float max_class_score = maxV;
			if (max_class_score < this->_classThreshold)
				continue;
			scores.push_back(max_class_score);

			cv::Mat theta_scores = out.row(i).colRange(5 + nc, out.row(i).cols);
			cv::minMaxLoc(theta_scores, &minV, &maxV, &minI, &maxI);
			float theta_idx = maxI.x;
			float theta_pred = (theta_idx - 90) / 180 * M_PI;

			bboxes.push_back(cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(longside, shortside), theta_pred));

			OBB obb;
			obb.cx = cx;
			obb.cy = cy;
			obb.longside = longside;
			obb.shortside = shortside;
			obb.theta_pred = theta_pred;
			obb.max_class_score = max_class_score;
			obb.class_idx = class_idx;
			generate_boxes.push_back(obb);
		}

		std::vector<int> indices;
		cv::dnn::NMSBoxes(bboxes, scores, this->_classThreshold, this->_nmsThreshold, indices);

		std::vector<OBB> det;
		for (int idx : indices) {
			det.push_back(generate_boxes[idx]);
		}

		cv::Mat pred_poly(det.size(), 10, CV_32FC1);
		rbox2poly(det, pred_poly);
		
		cv::Mat scaled_poly(det.size(), 10, CV_32FC1);;
		scale_polys(pred_poly, params[img_i]);
		
		// 一个输入图片的推理结果封装
		std::vector<OutputSegObb> out_segs;
		polys_2_OutputSegObbs(pred_poly, out_segs);

		// 所有输入图片的推理结果封装
		output.emplace_back(out_segs);
	}
	auto t3 = std::chrono::high_resolution_clock::now();   //结束时间

	auto delta_1_0 = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0);
	auto delta_2_1 = std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1);
	auto delta_3_2 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t2);
	auto delta_3_0 = std::chrono::duration_cast<std::chrono::microseconds>(t3 - t0);
	std::cout << "pre："  << delta_1_0.count() / 1000 << "ms;" << " ";
	std::cout << "infer：" << delta_2_1.count() / 1000 << "ms;" << " ";
	std::cout << "post：" << delta_3_2.count() / 1000 << "ms;" << " ";
	std::cout << "all：" << delta_3_0.count() / 1000 << "ms;" << std::endl;

	return true;
}

bool Yolov5ObbOnnx::OnnxDetect(cv::Mat& srcImg, std::vector<OutputSegObb>& output) {
	
	std::vector<cv::Mat> input_data = { srcImg };
	std::vector<std::vector<OutputSegObb>> tenp_output;
	if (OnnxBatchDetect(input_data, tenp_output)) {
		output = tenp_output[0];
		return true;
	} else return false;

}
