#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>

class SegmentationNode : public rclcpp::Node
{
public:
    SegmentationNode() : Node("segmentation_node")
    {
        RCLCPP_INFO(this->get_logger(), "SegmentationNode with Euclidean Clustering initialized");

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/filtered_points", 10,
            std::bind(&SegmentationNode::pointCloudCallback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/segmented_points", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        // Convert ROS2 message to PCL
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty cloud");
            return;
        }

        // --- KDTree + Euclidean Cluster Extraction ---
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.5); // distance threshold (meters)
        ec.setMinClusterSize(50);    // discard small clusters
        ec.setMaxClusterSize(25000); // discard very large clusters
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        // Combine all clusters into one cloud (can later assign colors per cluster)
        pcl::PointCloud<pcl::PointXYZ>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (auto &indices : cluster_indices)
        {
            for (auto &idx : indices.indices)
                clustered_cloud->points.push_back(cloud->points[idx]);
        }
        clustered_cloud->width = clustered_cloud->points.size();
        clustered_cloud->height = 1;
        clustered_cloud->is_dense = true;

        // Convert back to ROS2 PointCloud2
        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clustered_cloud, output);
        output.header = msg->header;

        publisher_->publish(output);
        RCLCPP_INFO(this->get_logger(), "Published %zu clusters with %zu total points",
                    cluster_indices.size(), clustered_cloud->points.size());
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<SegmentationNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
