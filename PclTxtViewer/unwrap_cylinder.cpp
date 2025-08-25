#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <algorithm>
#include <sstream>
#include <fstream>
#include <limits> // Cần cho std::numeric_limits

#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using PointT = pcl::PointXYZ;

// Một cấu trúc để trả về nhiều kết quả từ hàm trải phẳng
struct UnwrapResult {
    pcl::PointCloud<PointT>::Ptr cloud; // Đám mây điểm đã trải phẳng
    double avg_radius;                  // Bán kính trung bình đã tính
    double min_z;                       // Chiều cao tối thiểu
    double max_z;                       // Chiều cao tối đa
};

// ----------------------------
// Hàm đọc file TXT (x y z mỗi dòng)
// ----------------------------
#include <filesystem>
#include <string>
#include <algorithm> // Cần cho std::replace

// ----------------------------
// Hàm đọc file TXT (phiên bản nâng cấp, có khả năng gỡ lỗi)
// ----------------------------
pcl::PointCloud<PointT>::Ptr loadTXTFile(const std::string& filename) {
    pcl::PointCloud<PointT>::Ptr cloud(new pcl::PointCloud<PointT>());
    std::filesystem::path file_path(filename);
    std::ifstream infile(file_path);

    if (!infile.is_open()) {
        throw std::runtime_error("Cannot open TXT file. Attempted to open: " + std::filesystem::absolute(file_path).string());
    }

    std::string line;
    long long line_number = 0;
    long long error_lines_reported = 0;

    while (std::getline(infile, line)) {
        line_number++;
        
        // Bỏ qua các dòng trống hoặc comment (bắt đầu bằng #, //, ;)
        if (line.empty() || line[0] == '#' || (line.length() > 1 && line.substr(0, 2) == "//") || line[0] == ';') {
            continue;
        }

        // (MỚI) Thay thế tất cả dấu phẩy bằng khoảng trắng để chuẩn hóa
        std::replace(line.begin(), line.end(), ',', ' ');

        std::stringstream ss(line);
        double x, y, z;
        
        // Đọc 3 số double
        if (ss >> x && ss >> y && ss >> z) {
            cloud->points.emplace_back(static_cast<float>(x),
                                       static_cast<float>(y),
                                       static_cast<float>(z));
        } else {
            // (MỚI) In ra các dòng đầu tiên bị lỗi để gỡ rối
            if (error_lines_reported < 5) { // Chỉ in ra 5 lỗi đầu tiên để tránh spam
                std::cerr << "Warning: Could not parse line " << line_number << ": \"" << line << "\"" << std::endl;
                error_lines_reported++;
            }
        }
    }
    
    if (error_lines_reported > 0) {
        std::cerr << "..." << std::endl;
        std::cerr << "There were parsing errors. Please check the file format." << std::endl;
    }

    cloud->width = static_cast<uint32_t>(cloud->points.size());
    cloud->height = 1;
    cloud->is_dense = true;
    return cloud;
}
/**
 * @brief Trải phẳng một đám mây điểm hình trụ thành một mặt phẳng 2D.
 * GIẢ ĐỊNH: Hình trụ đã được căn chỉnh với trục Z.
 */
UnwrapResult unwrapCylinderSimple(const pcl::PointCloud<PointT>::ConstPtr& cloud_in)
{
    if (cloud_in->points.empty()) {
        throw std::runtime_error("Input point cloud is empty!");
    }

    UnwrapResult result;
    double total_radius = 0.0;
    result.min_z = std::numeric_limits<double>::max();
    result.max_z = std::numeric_limits<double>::lowest();

    // Lặp qua 1 lần để tính bán kính trung bình + min/max Z
    for (const auto& point : cloud_in->points) {
        total_radius += std::sqrt(point.x * point.x + point.y * point.y);
        if (point.z < result.min_z) result.min_z = point.z;
        if (point.z > result.max_z) result.max_z = point.z;
    }
    result.avg_radius = total_radius / cloud_in->points.size();
    
    std::cout << "Calculated average radius = " << result.avg_radius << std::endl;
    std::cout << "Z-height is in range [" << result.min_z << ", " << result.max_z << "]" << std::endl;

    result.cloud.reset(new pcl::PointCloud<PointT>());
    result.cloud->points.reserve(cloud_in->points.size());

    // Trải phẳng: (r,theta,z) -> (R*theta, z)
    for (const auto& p_in : cloud_in->points) {
        PointT p_out;
        double theta = std::atan2(p_in.y, p_in.x);
        p_out.x = static_cast<float>(result.avg_radius * theta);
        p_out.y = p_in.z;
        p_out.z = 0.0f;
        result.cloud->points.push_back(p_out);
    }

    result.cloud->width = static_cast<uint32_t>(result.cloud->points.size());
    result.cloud->height = 1;
    return result;
}


int main()
{
    const std::string FILE_PATH = R"(C:\Users\MAY02\Documents\E3C\NP-05_SCN0001.txt)";
    pcl::PointCloud<PointT>::Ptr original_cloud(new pcl::PointCloud<PointT>());

    try {
        // (FIX) Tự định nghĩa hàm ends_with để tương thích với C++17 trở về trước
        auto ends_with = [](const std::string& str, const std::string& suffix) {
            if (str.length() < suffix.length()) {
                return false;
            }
            return str.compare(str.length() - suffix.length(), suffix.length(), suffix) == 0;
        };

        if (ends_with(FILE_PATH, ".pcd")) {
            std::cout << "Detected .pcd file. Loading..." << std::endl;
            if (pcl::io::loadPCDFile<PointT>(FILE_PATH, *original_cloud) == -1) {
                throw std::runtime_error("Could not read PCD file: " + FILE_PATH);
            }
        } else if (ends_with(FILE_PATH, ".txt")) {
            std::cout << "Detected .txt file. Loading..." << std::endl;
            original_cloud = loadTXTFile(FILE_PATH);
        } else {
            throw std::runtime_error("Unsupported file format (only .pcd or .txt are supported).");
        }
    } catch (const std::exception& e) {
        std::cerr << "File reading error: " << e.what() << std::endl;
        return -1;
    }

    std::cout << "Loaded " << original_cloud->points.size() << " points." << std::endl;

    try {
        UnwrapResult result = unwrapCylinderSimple(original_cloud);

        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Simple Cylinder Unwrapping"));
        viewer->setBackgroundColor(0.1, 0.1, 0.1);

        int v1(0), v2(0);
        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);

        // Viewport 1: gốc
        viewer->addText("Original Cylinder", 10, 10, "v1_text", v1);
        pcl::visualization::PointCloudColorHandlerCustom<PointT> original_color(original_cloud, 150, 150, 150);
        viewer->addPointCloud<PointT>(original_cloud, original_color, "original_cloud", v1);
        viewer->addCoordinateSystem(1.0, "original_cs", v1);

        // Viewport 2: unwrap
        viewer->addText("Unwrapped Surface", 10, 10, "v2_text", v2);
        pcl::visualization::PointCloudColorHandlerCustom<PointT> unwrapped_color(result.cloud, 0, 255, 255);
        viewer->addPointCloud<PointT>(result.cloud, "unwrapped_cloud", v2);
        viewer->addCoordinateSystem(1.0, "unwrapped_cs", v2);

        // Kích thước unwrap
        double unwrapped_width = 2 * M_PI * result.avg_radius;
        double unwrapped_height = result.max_z - result.min_z;
        double x_min = -unwrapped_width / 2.0;
        double x_max = unwrapped_width / 2.0;

        viewer->addCube(x_min, x_max, result.min_z, result.max_z, -0.01, 0.01, 1.0, 1.0, 0.0, "bbox", v2);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION,
                                            pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME,
                                            "bbox");

        std::stringstream width_label, height_label;
        width_label.precision(3);
        height_label.precision(3);
        width_label << std::fixed << "Circumference (Width) = " << unwrapped_width;
        height_label << std::fixed << "Height = " << unwrapped_height;
        viewer->addText(width_label.str(), 10, 50, 16, 1.0, 1.0, 0.0, "width_label", v2);
        viewer->addText(height_label.str(), 10, 30, 16, 1.0, 1.0, 0.0, "height_label", v2);

        viewer->initCameraParameters();

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
        }
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}