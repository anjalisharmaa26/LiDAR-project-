#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/crop_box.h>
#include <Eigen/Dense>

class LidarFilterNode : public rclcpp::Node
{
public:
    LidarFilterNode() : Node("lidar_filter_node")
    {
        RCLCPP_INFO(this->get_logger(), "LidarFilterNode initialized.");

        // Subscribe to /velodyne_points
        sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/velodyne_points", 10,
            std::bind(&LidarFilterNode::pointCloudCallback, this, std::placeholders::_1));

        // Publish to /filtered_points
        pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/filtered_points", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZI> cloud;
        pcl::fromROSMsg(*msg, cloud);

        // 1️⃣ Downsample
        pcl::VoxelGrid<pcl::PointXYZI> voxel;
        voxel.setInputCloud(cloud.makeShared());
        voxel.setLeafSize(0.2f, 0.2f, 0.2f);

        pcl::PointCloud<pcl::PointXYZI> cloud_downsampled;
        voxel.filter(cloud_downsampled);

        // 2️⃣ Crop box
        pcl::CropBox<pcl::PointXYZI> crop;
        crop.setInputCloud(cloud_downsampled.makeShared());
        crop.setMin(Eigen::Vector4f(-10, -5, -2, 1));  // Adjust as needed
        crop.setMax(Eigen::Vector4f(30, 5, 2, 1));

        pcl::PointCloud<pcl::PointXYZI> cloud_cropped;
        crop.filter(cloud_cropped);

        // 3️⃣ Convert back to ROS2 message
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(cloud_cropped, output);
        output.header = msg->header;

        // 4️⃣ Publish
        pub_->publish(output);
        RCLCPP_INFO(this->get_logger(), "Published filtered cloud: %zu points", cloud_cropped.size());
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pub_;
};

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<LidarFilterNode>());
    rclcpp::shutdown();
    return 0;
} 