#include <memory>
#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/point_cloud2.hpp"
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/common/common.h>
#include <pcl/io/pcd_io.h> // Optional PCD reading
#include <cstdlib>   // for rand()

// For RViz markers
#include "visualization_msgs/msg/marker_array.hpp"
#include "visualization_msgs/msg/marker.hpp"

class ClusteringNode : public rclcpp::Node
{
public:
    ClusteringNode() : Node("clustering_node")
    {
        RCLCPP_INFO(this->get_logger(), "ClusteringNode initialized");

        // Subscribe to segmented points from segmentation_node
        subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/segmented_points", 10,
            std::bind(&ClusteringNode::pointCloudCallback, this, std::placeholders::_1));

        // Publish clustered points
        publisher_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
            "/clusters", 10);

        // Publish bounding boxes as MarkerArray for RViz
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/cluster_markers", 10);

        // Optional: Load PCD file for offline testing
        // loadPCD("/home/anjali26/ros2_ws/src/lidar_processing/pcd_samples/sample.pcd");
    }

private:
    void pointCloudCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        pcl::fromROSMsg(*msg, *cloud);

        if (cloud->empty())
        {
            RCLCPP_WARN(this->get_logger(), "Received empty segmented cloud");
            return;
        }

        // Optional: split cloud into road and objects (example using z-threshold)
        pcl::PointCloud<pcl::PointXYZ>::Ptr objects_cloud(new pcl::PointCloud<pcl::PointXYZ>());
        for (auto &p : cloud->points)
        {
            if (p.z >= -1.5) // adjust threshold for objects
                objects_cloud->points.push_back(p);
        }

        // KDTree for clustering objects
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());
        tree->setInputCloud(objects_cloud);

        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.5);
        ec.setMinClusterSize(50);
        ec.setMaxClusterSize(25000);
        ec.setSearchMethod(tree);
        ec.setInputCloud(objects_cloud);
        ec.extract(cluster_indices);

        // Create colored cloud for clusters
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr clustered_cloud(new pcl::PointCloud<pcl::PointXYZRGB>());

        // Prepare MarkerArray for bounding boxes
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
                pt.x = objects_cloud->points[idx].x;
                pt.y = objects_cloud->points[idx].y;
                pt.z = objects_cloud->points[idx].z;
                pt.r = r; pt.g = g; pt.b = b;
                clustered_cloud->points.push_back(pt);
                cluster->points.push_back(objects_cloud->points[idx]);
            }

            // Bounding box marker
            pcl::PointXYZ minPoint, maxPoint;
            pcl::getMinMax3D(*cluster, minPoint, maxPoint);

            visualization_msgs::msg::Marker box;
            box.header.frame_id = msg->header.frame_id;
            box.header.stamp = this->get_clock()->now();
            box.ns = "clusters";
            box.id = cluster_id;
            box.type = visualization_msgs::msg::Marker::CUBE;
            box.action = visualization_msgs::msg::Marker::ADD;

            box.pose.position.x = (minPoint.x + maxPoint.x) / 2.0;
            box.pose.position.y = (minPoint.y + maxPoint.y) / 2.0;
            box.pose.position.z = (minPoint.z + maxPoint.z) / 2.0;

            box.scale.x = maxPoint.x - minPoint.x;
            box.scale.y = maxPoint.y - minPoint.y;
            box.scale.z = maxPoint.z - minPoint.z;

            box.color.r = static_cast<float>(r) / 255.0;
            box.color.g = static_cast<float>(g) / 255.0;
            box.color.b = static_cast<float>(b) / 255.0;
            box.color.a = 0.8;

            marker_array.markers.push_back(box);
            cluster_id++;
        }

        // Publish clustered cloud
        clustered_cloud->width = clustered_cloud->points.size();
        clustered_cloud->height = 1;
        clustered_cloud->is_dense = true;

        sensor_msgs::msg::PointCloud2 output;
        pcl::toROSMsg(*clustered_cloud, output);
        output.header = msg->header;
        publisher_->publish(output);

        // Publish bounding boxes
        marker_pub_->publish(marker_array);

        RCLCPP_INFO(this->get_logger(), "Published %d clusters with %zu points", cluster_id, clustered_cloud->points.size());
    }

    // Optional: Load PCD file
    void loadPCD(const std::string &path)
    {
        pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>());
        if (pcl::io::loadPCDFile<pcl::PointXYZ>(path, *cloud) == -1)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load PCD file: %s", path.c_str());
            return;
        }
        RCLCPP_INFO(this->get_logger(), "Loaded PCD file with %zu points", cloud->points.size());
    }

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr publisher_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;
};

int main(int argc, char *argv[])
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<ClusteringNode>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
