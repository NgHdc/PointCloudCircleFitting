#include <iostream>
#include <vector>
#include <cmath>
#include <string>
#include <stdexcept>
#include <filesystem>
#include <algorithm> // Για std::minmax_element
#include <sstream>   // Για std::stringstream

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

/**
 * @brief Trải phẳng một đám mây điểm hình trụ thành một mặt phẳng 2D.
 * GIẢ ĐỊNH: Hình trụ đã được căn chỉnh với trục Z.
 * @param cloud_in Đám mây điểm hình trụ đầu vào.
 * @return Một cấu trúc chứa đám mây đã trải phẳng và các thông số hình học.
 */
UnwrapResult unwrapCylinderSimple(const pcl::PointCloud<PointT>::ConstPtr& cloud_in)
{
    if (cloud_in->points.empty()) {
        throw std::runtime_error("Đám mây điểm đầu vào rỗng!");
    }

    UnwrapResult result;
    double total_radius = 0.0;
    result.min_z = std::numeric_limits<double>::max();
    result.max_z = std::numeric_limits<double>::lowest();

    // Lặp qua một lần để tính bán kính trung bình và tìm min/max Z
    for (const auto& point : cloud_in->points) {
        total_radius += std::sqrt(point.x * point.x + point.y * point.y);
        if (point.z < result.min_z) result.min_z = point.z;
        if (point.z > result.max_z) result.max_z = point.z;
    }
    result.avg_radius = total_radius / cloud_in->points.size();
    
    std::cout << "Bán kính trung bình được tính toán là: " << result.avg_radius << std::endl;
    std::cout << "Chiều cao (trục Z) trong khoảng: [" << result.min_z << ", " << result.max_z << "]" << std::endl;

    result.cloud.reset(new pcl::PointCloud<PointT>());
    result.cloud->points.reserve(cloud_in->points.size());

    // Lặp qua lần thứ hai để thực hiện phép biến đổi
    for (const auto& p_in : cloud_in->points) {
        PointT p_out;
        double theta = std::atan2(p_in.y, p_in.x);
        
        p_out.x = static_cast<float>(result.avg_radius * theta);
        p_out.y = p_in.z;
        p_out.z = 0.0f; // Đặt Z = 0 để tạo ra mặt phẳng hoàn hảo

        result.cloud->points.push_back(p_out);
    }

    result.cloud->width = static_cast<uint32_t>(result.cloud->points.size());
    result.cloud->height = 1;
    return result;
}


int main()
{
    const std::string PCD_FILE_PATH = R"(C:\Users\MAY02\Documents\E3C\point_cloud_1.pcd)";

    pcl::PointCloud<PointT>::Ptr original_cloud(new pcl::PointCloud<PointT>());
    if (pcl::io::loadPCDFile<PointT>(PCD_FILE_PATH, *original_cloud) == -1) {
        PCL_ERROR("Lỗi: Không thể đọc file %s\n", PCD_FILE_PATH.c_str()); return -1;
    }
    std::cout << "Tải thành công " << original_cloud->points.size() << " điểm." << std::endl;

    try {
        // Gọi hàm trải phẳng để lấy kết quả
        UnwrapResult result = unwrapCylinderSimple(original_cloud);
        
        pcl::visualization::PCLVisualizer::Ptr viewer(new pcl::visualization::PCLVisualizer("Simple Cylinder Unwrapping"));
        viewer->setBackgroundColor(0.1, 0.1, 0.1);

        int v1(0), v2(0);
        viewer->createViewPort(0.0, 0.0, 0.5, 1.0, v1);
        viewer->createViewPort(0.5, 0.0, 1.0, 1.0, v2);

        // --- Khung nhìn bên trái (v1): Hình trụ gốc ---
        viewer->addText("Original Cylinder", 10, 10, "v1_text", v1);
        pcl::visualization::PointCloudColorHandlerCustom<PointT> original_color(original_cloud, 150, 150, 150); // Màu xám
        viewer->addPointCloud<PointT>(original_cloud, original_color, "original_cloud", v1);

        // --- Khung nhìn bên phải (v2): Bề mặt đã trải phẳng và các thông số ---
        viewer->addText("Unwrapped Surface", 10, 10, "v2_text", v2);
        pcl::visualization::PointCloudColorHandlerCustom<PointT> unwrapped_color(result.cloud, 0, 255, 255); // Màu xanh cyan
        viewer->addPointCloud<PointT>(result.cloud, "unwrapped_cloud", v2);
        
        // Thêm hệ trục tọa độ
        viewer->addCoordinateSystem(result.avg_radius, "ref_frame", v2);

        // --- Thêm thước đo và nhãn ---
        double unwrapped_width = 2 * M_PI * result.avg_radius;
        double unwrapped_height = result.max_z - result.min_z;
        double x_min = -unwrapped_width / 2.0;
        double x_max = unwrapped_width / 2.0;

        // Thêm hộp giới hạn (bounding box) hoạt động như một cây thước
        viewer->addCube(x_min, x_max, result.min_z, result.max_z, -0.01, 0.01, 1.0, 1.0, 0.0, "bbox", v2);
        viewer->setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_REPRESENTATION, pcl::visualization::PCL_VISUALIZER_REPRESENTATION_WIREFRAME, "bbox");

        // Thêm nhãn văn bản giải thích kích thước
        std::stringstream width_label, height_label;
        width_label.precision(2);
        height_label.precision(2);
        width_label << std::fixed << "Circumference (Width) = " << unwrapped_width << " units";
        height_label << std::fixed << "Height = " << unwrapped_height << " units";
        
        viewer->addText(width_label.str(), 10, 50, 16, 1.0, 1.0, 0.0, "width_label", v2);
        viewer->addText(height_label.str(), 10, 30, 16, 1.0, 1.0, 0.0, "height_label", v2);

        // Đặt camera nhìn thẳng từ trên xuống (top-down view)
        viewer->setCameraPosition(0, 0, result.max_z + unwrapped_width, // Vị trí camera
                                  0, 0, 0,                           // Điểm nhìn vào (tâm)
                                  0, 1, 0,                           // Vector "up"
                                  v2);
        
        viewer->initCameraParameters();

        while (!viewer->wasStopped()) {
            viewer->spinOnce(100);
        }

    } catch (const std::exception& e) {
        std::cerr << "Đã xảy ra lỗi: " << e.what() << std::endl;
        return -1;
    }

    return 0;
}