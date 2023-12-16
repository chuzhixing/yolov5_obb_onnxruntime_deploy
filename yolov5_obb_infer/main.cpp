#include <fstream>
#include "yolov5_obb_onnx.h"
#include<time.h>


// _main
int main() {
    bool use_cuda = false;
    std::string model_path = "assets/your_yolov5obb.onnx";
    std::string image_path = "assets/your.jpg";
    std::string out_path = "res/xx.jpg";

    Yolov5ObbOnnx yolov5_obb;
    yolov5_obb._className = { "h0", "h1" };
    yolov5_obb._classThreshold = 0.25f;   //
    yolov5_obb._nmsThreshold = 0.45;      // 这个超参跟mAP的IOU阈值不是一会事，作用也不一样;
    yolov5_obb.set_net_w_h(800, 800);

    if (yolov5_obb.ReadModel(model_path, true)) {
        std::cout << "read net ok!" << std::endl;
    } else {
        return 0;
    }
    
    // read img
    cv::Mat img = cv::imread(image_path);

    // infer
    std::vector<OutputSegObb> output_segs;
    yolov5_obb.OnnxDetect(img, output_segs);

    // draw result
    cv::Mat res_mat = img.clone();
    for (int i = 0; i < output_segs.size(); i++) {
        cv::Point2f vertices[4];
        output_segs[i].box.points(vertices);
        for (int j = 0; j < 4; j++)
            cv::line(res_mat, vertices[j], vertices[(j + 1) % 4], cv::Scalar(0, 255, 0), 1);
    }
    cv::imwrite("res/gg.png", res_mat);

    std::cout << "end ..." << std::endl;
    system("pause");
    return 0;
}
