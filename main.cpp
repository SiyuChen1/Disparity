#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/stereo.hpp>


int main() {
//    std::string left_img_path = "/home/siyuchen/lib/opencv_all/opencv_contrib/modules/stereo/testdata/imL2l.bmp";
//    std::string right_img_path = "/home/siyuchen/lib/opencv_all/opencv_contrib/modules/stereo/testdata/imL2.bmp";

    std::string left_img_path = "/home/siyuchen/lib/opencv_all/opencv_contrib/modules/stereo/testdata/imgKittyl.bmp";
    std::string right_img_path = "/home/siyuchen/lib/opencv_all/opencv_contrib/modules/stereo/testdata/imgKitty.bmp";

    cv::Mat left_img = cv::imread(left_img_path, cv::IMREAD_GRAYSCALE);
    cv::Mat right_img = cv::imread(right_img_path, cv::IMREAD_GRAYSCALE);

    if(left_img.empty())
    {
        std::cout << "Could not read the image: " << left_img_path << std::endl;
        return 1;
    }

    if(right_img.empty())
    {
        std::cout << "Could not read the image: " << left_img_path << std::endl;
        return 1;
    }

    cv::Mat horizontal_concat;
    cv::hconcat(left_img, right_img, horizontal_concat);

    // Start and end points of the line
    // Starting point in the top-left corner
    cv::Point start(0, (int)(horizontal_concat.rows / 3));
    // Ending point in the bottom-right corner
    cv::Point end(horizontal_concat.cols, (int)(horizontal_concat.rows / 3));

    // Draw a white line with thickness of 2 pixels
    cv::line(horizontal_concat, start, end, cv::Scalar(255, 0, 255), 2);

    cv::imshow("Concatenated Image", horizontal_concat); // or verticalConcat

    int numDisparities = 32;  // Must be divisible by 16
    int blockSize = 7;       // Odd number

    cv::Mat disparity_bbm = cv::Mat(left_img.size(), left_img.type());
    cv::Mat disparity_bm;

    cv::Ptr<cv::StereoBM> stereoBM = cv::StereoBM::create(numDisparities, blockSize);
    stereoBM->compute(left_img, right_img, disparity_bm);
    cv::normalize(disparity_bm, disparity_bm, 0, 255, cv::NORM_MINMAX, CV_8U);

    auto stereoBinaryBM = cv::stereo::StereoBinaryBM::create(numDisparities, blockSize);
    stereoBinaryBM->compute(left_img, right_img, disparity_bbm);

    cv::normalize(disparity_bbm, disparity_bbm, 0, 255, cv::NORM_MINMAX, CV_8U);

    cv::Mat depth_concat;
    cv::hconcat(disparity_bm, disparity_bbm, depth_concat);

    cv::imshow("Disparity Map", depth_concat);

    cv::waitKey(0);
}
