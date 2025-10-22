#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <cstdlib> // rand
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

class ClusteringNode : public rclcpp::Node
{
public:
    ClusteringNode() : Node("clustering_node")
    {
        RCLCPP_INFO(this->get_logger(), "ClusteringNode initialized");

        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/segmented_points", 10,
            std::bind(&ClusteringNode::pointCloudCallback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>("/clusters", 10);
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>("/cluster_markers", 10);
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty cloud");
            return;
        }

        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.5);
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(cloud);
        ec.extract(cluster_indices);

        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());
        visualization_msgs::msg::MarkerArray marker_array;
        int cluster_id = 0;

        for (auto &indices : cluster_indices)
        {
            uint8_t r = rand() % 256;
            uint8_t g = rand() % 256;
            uint8_t b = rand() % 256;
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>());

            for (auto &idx : indices.indices)
            {
                pcl::PointXYZRGB pt;
                pt.x = cloud->points[idx].x;
                pt.y = cloud->points[idx].y;
                pt.z = cloud->points[idx].z;
                pt.r = r;
                pt.g = g;
                pt.b = b;
                clustered_cloud->points.push_back(pt);
                cluster->points.push_back(cloud->points[idx]);
            }

            pcl::PointXYZ min_pt, max_pt;
            pcl::getMinMax3D(*cluster, min_pt, max_pt);

            visualization_msgs::msg::Marker box;
            box.header.frame_id = msg->header.frame_id;
            box.header.stamp = this->get_clock()->now();
            box.ns = "clusters";
            box.id = cluster_id;
            box.type = visualization_msgs::msg::Marker::CUBE;
            box.action = visualization_msgs::msg::Marker::ADD;
            box.pose.position.x = (min_pt.x + max_pt.x) / 2.0;
            box.pose.position.y = (min_pt.y + max_pt.y) / 2.0;
            box.pose.position.z = (min_pt.z + max_pt.z) / 2.0;
            box.pose.orientation.w = 1.0;
            box.scale.x = max_pt.x - min_pt.x;
            box.scale.y = max_pt.y - min_pt.y;
            box.scale.z = max_pt.z - min_pt.z;
            box.color.r = static_cast<float>(r) / 255.0;
            box.color.g = static_cast<float>(g) / 255.0;
            box.color.b = static_cast<float>(b) / 255.0;
            box.color.a = 0.8;
            box.lifetime = rclcpp::Duration::from_seconds(0.1);
            marker_array.markers.push_back(box);
            cluster_id++;
        }

        clustered_cloud->width = clustered_cloud->points.size();
        clustered_cloud->height = 1;
        clustered_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clustered_cloud, output);
        output.header = msg->header;
        publisher_->publish(output);
        marker_pub_->publish(marker_array);

        RCLCPP_INFO(this->get_logger(), "Published %d clusters with %zu points", cluster_id, clustered_cloud->points.size());
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<ClusteringNode>());
    rclcpp::shutdown();
    return 0;
}
