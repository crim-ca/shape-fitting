#include <ctime>
#include <chrono>
#include <algorithm>
#include <vector>
#include <boost/filesystem.hpp>
#include <boost/filesystem.hpp>
#include <boost/range/iterator_range.hpp>
#include "helpers.h"
#include "cylinder_fitting_hough.h"

typedef pcl::PointCloud<pcl::PointXYZ> PointCloud;

pcl::NormalEstimation<PointT, pcl::Normal> ne;
pcl::search::KdTree<PointT>::Ptr tree;

int main (int argc, char** argv)
{
	// Arguments
	std::string pointcloud_clusters_path=std::string(argv[1]);

	// HOUGH PARAMETERS
	unsigned int angle_bins=atoi(argv[2]);
	std::cout << "angle_bins: " << angle_bins<< std::endl;

	unsigned int radius_bins=atoi(argv[3]);
	std::cout << "radius_bins: " << radius_bins<< std::endl;

	unsigned int position_bins=atoi(argv[4]);
	std::cout << "position_bins: " << position_bins<< std::endl;

	// OUR HOUGH IMPLEMENTATION PARAMETERS
	unsigned int orientation_accumulators_num=atoi(argv[5]);
	std::cout << "orientation_accumulators_num: " << orientation_accumulators_num<< std::endl;

	unsigned int gaussian_sphere_points_num=atoi(argv[6]);
	std::cout << "gaussian_sphere_points_num: " << gaussian_sphere_points_num<< std::endl;

	float accumulator_peak_threshold=atof(argv[7]);
	std::cout << "accumulator_peak_threshold: " << accumulator_peak_threshold<< std::endl;

	float min_radius=atof(argv[8]);
	std::cout << "min_radius: " << min_radius << std::endl;

	float max_radius=atof(argv[9]);
	std::cout << "max_radius: " << max_radius << std::endl;

	std::vector<double> radius_values = splitNumbers(argv[10]);
	std::cout << "radius values:" << std::endl;
	for (auto value : radius_values)
		std::cout << "  " << value << std::endl;
	std::sort(radius_values.begin(), radius_values.end());

	int mode=atof(argv[11]);
	std::cout << "mode: " << mode << std::endl;

	bool soft_voting=atoi(argv[12]);
	std::cout << "soft_voting: " << soft_voting << std::endl;

	bool visualize=atoi(argv[13]);
	std::cout << "visualize: " << visualize << std::endl;

	double min_confidence = 0.8;
	if (argc > 14)
		min_confidence = atof(argv[14]);
	if (min_confidence < 0.0)
		min_confidence = 0.0;
	std::cout << "min confidence: " << visualize << std::endl;

	unsigned int max_attempts = 10;
	if (argc > 15)
		max_attempts = atoi(argv[15]);
	if (max_attempts < 1)
		max_attempts = 1;
	std::cout << "max attempts: " << max_attempts << std::endl;

	std::vector<std::string> image_files,annotation_files,point_cloud_files;

	boost::shared_ptr<VisualizeFittingData> visualizer;

	if(visualize)
		visualizer=boost::shared_ptr<VisualizeFittingData>(new VisualizeFittingData());

	// Gaussian Sphere Uniform
	std::vector<double> weights;
	std::vector<Eigen::Matrix<double, 3 ,1> > means;
	std::vector<Eigen::Matrix<double, 3 ,1> > std_devs;
	weights.push_back(1.0);
	Eigen::Matrix<double, 3 ,1> mean_eigen(0,0,0);
	means.push_back(mean_eigen);
	Eigen::Matrix<double, 3 ,1> std_dev_eigen(0.5,0.5,0.5);
	std_devs.push_back(std_dev_eigen);

	unsigned int current_dir=0;
	pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
	pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>());

	std::fstream cluster_fitting_score;

	// For each directory
	for (boost::filesystem::directory_iterator itr(pointcloud_clusters_path); itr!=boost::filesystem::directory_iterator(); ++itr)
	{
		std::cout << "directory number:" << ++current_dir << std::endl;

		std::string pointcloud_clusters_sub_path = pointcloud_clusters_path+itr->path().filename().string()+"/";
		PointClouds point_clouds(pointcloud_clusters_sub_path);
		std::string results_path_current=pointcloud_clusters_sub_path+"results/";

		createDirectory(results_path_current);
		cluster_fitting_score.open(results_path_current+"cluster_fitting_score.txt", std::fstream::in | std::fstream::out | std::fstream::trunc);
		for(unsigned int i=0;i<point_clouds.file_names.size();++i)
		{
			try
			{
				pcl::PointCloud<pcl::PointXYZ>::Ptr point_cloud(new pcl::PointCloud<pcl::PointXYZ>);
				pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>());

				// Load 3D point cloud information in PCL format
				point_cloud=point_clouds.loadPointCloud(pointcloud_clusters_sub_path+point_clouds.file_names[i]);

				// Normal estimation parameter (not organized)
				ne.setKSearch (6);
				ne.setSearchMethod(tree);
				ne.setInputCloud(point_cloud);
				ne.compute(*cloud_normals);

				FittingData best_fit;
				best_fit.confidence = 0.0;

				for (unsigned int i_attempt=0; i_attempt < max_attempts; ++i_attempt) {
					// Random initialization parameters from gaussian normal distributions can make estimated results
					// very good or very bad. Retry as necessary to obtain highest possible confidence.
					GaussianMixtureModel gmm(weights,means,std_devs);
					GaussianSphere gaussian_sphere(gmm,gaussian_sphere_points_num,orientation_accumulators_num);
					boost::shared_ptr<CylinderFittingHough> cylinder_fitting(new CylinderFittingHough(
						gaussian_sphere,
						angle_bins,
						radius_bins,
						position_bins,
						min_radius,
						max_radius,
						radius_values,
						accumulator_peak_threshold,
						mode,
						false,
						soft_voting
					));

					FittingData fitting_data=cylinder_fitting->fit(point_cloud,cloud_normals);
					if (fitting_data.confidence > best_fit.confidence)
						best_fit = fitting_data;
					if (fitting_data.confidence >= min_confidence)
						break;
					else
						std::cout << "Confidence too low [" << fitting_data.confidence << "], "
								  << "retrying... (" << i_attempt << "/" << max_attempts << ")" << std::endl;
				}

				cluster_fitting_score << best_fit << std::endl;

				/* VISUALIZE */
				if(visualize)
					best_fit.visualize(point_cloud,visualizer);
				/* END VISUALIZE */

			}
			catch (std::exception& e)
			{
				std::cout << e.what() << std::endl;
				continue;
			}
		}
		cluster_fitting_score.close();
	}
	std::cout << "end" << std::endl;
	return 0;
}
