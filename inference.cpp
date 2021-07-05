#include <iostream>
#include <cmath>
#include <vector>
#include <opencv2/opencv.hpp>
#include "inference_engine.hpp"


using namespace InferenceEngine;

class BoxResult
{
    int x1;
    int y1;
    int x2;
    int y2;
    std::string label;
    float conf;
    int group;

    public:

    int getx1()
    {
        return x1;
    }

    int getx2()
    {
        return x2;
    }

    int gety1()
    {
        return y1;
    }

    int gety2()
    {
        return y2;
    }

    std::string getLabel()
    {
        return label;
    }

    int getConf()
    {
        return conf;
    }

    int getGroup()
    {
        return group;
    }

    void setGroup(int ngroup)
    {
        group = ngroup;
    }

    BoxResult(float nconf, std::string nlabel, int nx1, int ny1, int nx2, int ny2)
    {
        x1 = nx1;
        y1 = ny1;
        x2 = nx2;
        y2 = ny2;
        label = nlabel;
        conf = nconf;
        group = 0;
    }
    
    void print()
    {
        std::cout << "Confidence : " << conf << " ; ";
        std::cout << "label : " << label << " ; ";
        std::cout << "x1 = " << x1 << " ; ";
        std::cout << "y1 = " << y1 << " ; ";
        std::cout << "x2 = " << x2 << " ; ";
        std::cout << "y2 = " << y2 << std::endl;
    }
};

std::vector<BoxResult> outputReader(const float* output, double scoreThresh, int size, int imageWidth, int imageHeight)
{
    int dimensions = 85;
    std::string categories[80] = {"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"};
    int rows = size/dimensions;
    int confidenceIndex = 4;
    int labelStartIndex = 5;
    float modelWidth = 640.0f;
    float modelHeight = 640.0f;
    float xGain = modelWidth / imageWidth;
    float yGain = modelHeight / imageHeight;

    std::vector<BoxResult> resultsNms;
    int compt = 0;
    for (int i=0; i<rows; i++)
    {
        int index = i*dimensions;
        std::string label;
        float confidence;
        int x1;
        int x2;
        int y1;
        int y2;
        if (output[index + confidenceIndex] >= scoreThresh)
        {
            for (int k = labelStartIndex; k<dimensions; k++)
            {
                if (output[index+k]*output[index+confidenceIndex] >= scoreThresh)
                {
                    label = categories[k-labelStartIndex];
                    confidence = output[index+k]*output[index+confidenceIndex];
                    x1 = (output[index] - output[index + 2] / 2) / yGain; //top left x //I know x and y are modified but I'm sure this configuration is the right one
                    y1 = (output[index + 1] - output[index + 3] / 2) / xGain; //top left y
                    x2 = (output[index] + output[index + 2] / 2) / yGain; //bottom right x
                    y2 = (output[index + 1] + output[index + 3] / 2) / xGain; //bottom right y
                    resultsNms.push_back(BoxResult(confidence, label, x1, y1, x2, y2));
                }
            }
        }
    }
    return resultsNms;
}

std::vector<BoxResult> merge_vector(std::vector<BoxResult> linkedList1, std::vector<BoxResult> linkedList2)
{
    std::vector<BoxResult> res ;
    for (std::vector<BoxResult>::iterator obj=linkedList1.begin(); obj != linkedList1.end(); ++obj)
    {
        while (linkedList2.empty() == false && (*obj).getConf()<(*linkedList2.begin()).getConf())
        {
            res.push_back(*linkedList2.begin());
            linkedList2.erase(linkedList2.begin());
        }
        res.push_back(*obj);
    }
    if (linkedList2.empty() == false)
    {
        for (std::vector<BoxResult>::iterator obj=linkedList2.begin(); obj != linkedList2.end(); ++obj)
        {
            res.push_back(*obj);
        }
    }
    return res;
}

std::vector<BoxResult> sortConf (std::vector<BoxResult>::iterator a, int size)
{
    if (size == 1)
    {
        std::vector<BoxResult> res;
        res.push_back(*a);
        return res;
    }
    int interSize = size/2;
    return merge_vector(sortConf(a, interSize), sortConf(a+interSize, size-interSize));
}


int main()
{
    std::string model("/home/jb/Documents/metralabs_internship/people_detection/yolov5s.xml");
    cv::VideoCapture camera(0);


    Core core;
    CNNNetwork network;
    ExecutableNetwork executableNetwork;

    network = core.ReadNetwork(model);

    /** Take information about inputs **/
    InputInfo::Ptr input_info = network.getInputsInfo().begin()->second;
    std::string input_name = network.getInputsInfo().begin()->first;

    input_info->setLayout(Layout::NCHW);
    input_info->setPrecision(Precision::U8);
    input_info->getPreProcess().setResizeAlgorithm(InferenceEngine::RESIZE_BILINEAR);
    input_info->getPreProcess().setColorFormat(InferenceEngine::ColorFormat::RGB);

    /** Take information about outputs **/
    OutputsDataMap output_info = network.getOutputsInfo();
    

    /** Iterate over all output info**/
    for (auto &item : output_info) {
        auto output_data = item.second;
        output_data->setPrecision(InferenceEngine::Precision::FP32);
        //output_data->setLayout(InferenceEngine::Layout::NC);
    }


    executableNetwork = core.LoadNetwork(network, "MYRIAD");

    
    while(true)
    {
    InferRequest infer_request = executableNetwork.CreateInferRequest();


    cv::Mat image;
    camera >> image;
    cv::Mat imageResized;
    //cv::Mat intermediateImage;

    int width = image.size[0];
    int height = image.size[1];
    Blob::Ptr input2 = infer_request.GetBlob(input_name);
    auto input_data2 = input2->buffer().as<PrecisionTrait<Precision::U8>::value_type *>();
    cv::resize(image, imageResized, cv::Size(input_info->getTensorDesc().getDims()[3], input_info->getTensorDesc().getDims()[2]));
    cv::cvtColor(imageResized, imageResized, cv::COLOR_BGR2RGB);
    //Ptr<BackgroundSubtractor> pBackSub;
    //pBackSub = cv::createBackgroundSubtractorMOG2();
    //pBackSub->apply(imageResized, imageResized);

    imageResized = imageResized/255;

    size_t channels_number = input2->getTensorDesc().getDims()[1];
    size_t image_size = input2->getTensorDesc().getDims()[3] * input2->getTensorDesc().getDims()[2];

    for (size_t pid=0; pid<image_size; ++pid)
    {
        for (size_t ch=0; ch< channels_number; ++ch)
        {
            input_data2[ch*image_size+pid] = imageResized.at<cv::Vec3b>(pid)[ch];
        }
    }

    infer_request.Infer();

    const float* out;

    for (auto &item : output_info) 
    {
        auto output_name = item.first;
        auto output = infer_request.GetBlob(output_name);
        {
            auto const memLocker = output->cbuffer(); // use const memory locker
            // output_buffer is valid as long as the lifetime of memLocker
            const float *output_buffer = memLocker.as<const float *>();
            out = output_buffer;
            /** output_buffer[] - accessing output blob data **/
        }
    }

    std::vector<BoxResult> boxes;
    boxes = outputReader(out, 0.0005, 2142000, width, height);

    if (boxes.empty() == false)
    {
    boxes = sortConf(boxes.begin() , boxes.size());
    std::vector<BoxResult>::iterator obj=boxes.begin();
    for (std::vector<BoxResult>::iterator obj=boxes.begin(); obj!=boxes.end(); ++obj)
    {
        if ((*obj).getLabel()!= "traffic light" && (*obj).getLabel()!= "kite" && (*obj).getLabel()!= "dining table" )
        {
            (*obj).print();
            cv::Point point1 = cv::Point(int(std::min(std::max((*obj).getx1(),1),638)), int(std::min(std::max((*obj).gety1(),1),638)));
            cv::Point point2 = cv::Point(int(std::min(std::max((*obj).getx2(),1),638)), int(std::min(std::max((*obj).gety2(),1),638)));
            cv::rectangle(image, point1, point2, cv::Scalar(0,0,0), 2);
        

            cv::Point point3 = cv::Point(int(std::min(std::max((*obj).getx1()-20,1),638)), int(std::min(std::max((*obj).gety1(),1),638)));
            cv::putText(image, (*obj).getLabel(), point3, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,0,0), 2);
        }
    }
    }

    //for (std::vector<BoxResult>::iterator obj=boxes.begin(); obj!=boxes.end(); ++obj)
    //{
    //if ((*obj).getLabel()!= "traffic light" && (*obj).getLabel()!= "kite" && (*obj).getLabel()!= "dining table" )
    //{
        //(*obj).print();
        //cv::Point point1 = cv::Point(int(std::min(std::max((*obj).getx1(),1),638)), int(std::min(std::max((*obj).gety1(),1),638)));
        //cv::Point point2 = cv::Point(int(std::min(std::max((*obj).getx2(),1),638)), int(std::min(std::max((*obj).gety2(),1),638)));
        //cv::rectangle(image, point1, point2, cv::Scalar(0,0,0), 2);
        

        //cv::Point point3 = cv::Point(int(std::min(std::max((*obj).getx1()-20,1),638)), int(std::min(std::max((*obj).gety1(),1),638)));
        //cv::putText(image, (*obj).getLabel(), point3, cv::FONT_HERSHEY_COMPLEX, 1, cv::Scalar(0,0,0), 2);
    //}
    //}


    cv::imshow("Display", image);
    cv::waitKey(2);

    }
    return 0;
}
