
#include <iostream>
#include <vector>
#include <pcl/gpu/containers/initialization.h>
#include <boost/filesystem.hpp>
#include <pcl/io/grabber.h>
#include <pcl/io/pcd_grabber.h>
#include <pcl/gpu/kinfu/kinfu.h>
#include <pcl/common/angles.h>
#include <pcl/io/ply_io.h>
#include <pcl/gpu/kinfu/marching_cubes.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <boost/thread.hpp>
#include <pcl/io/pcd_io.h>

using namespace std;
using namespace pcl;
using namespace pcl::gpu;
using namespace Eigen;

KinfuTracker kinfu_;
pcl::PointCloud<pcl::PointXYZRGBA>::Ptr tsdf_cloud_ptr_;
boost::mutex data_ready_mutex_;
boost::condition_variable data_ready_cond_;
PtrStepSz<const unsigned short> depth_;
std::vector<unsigned short> source_depth_data_;
KinfuTracker::DepthMap depth_device_;
boost::shared_ptr<pcl::PolygonMesh> mesh_ptr_;
DeviceArray<PointXYZ> triangles_buffer_device_;
boost::shared_ptr<pcl::visualization::PCLVisualizer> viewer_;

boost::shared_ptr<pcl::PolygonMesh> convertToMesh(const DeviceArray<PointXYZ>& triangles)
{ 
      if (triangles.empty())
          return boost::shared_ptr<pcl::PolygonMesh>();

      pcl::PointCloud<pcl::PointXYZ> cloud;
      cloud.width  = (int)triangles.size();
      cloud.height = 1;
      triangles.download(cloud.points);

      boost::shared_ptr<pcl::PolygonMesh> mesh_ptr( new pcl::PolygonMesh() ); 
      pcl::toPCLPointCloud2(cloud, mesh_ptr->cloud);
      
      mesh_ptr->polygons.resize (triangles.size() / 3);
      for (size_t i = 0; i < mesh_ptr->polygons.size (); ++i)
      {
            pcl::Vertices v;
            v.vertices.push_back(i*3+0);
            v.vertices.push_back(i*3+2);
            v.vertices.push_back(i*3+1);              
            mesh_ptr->polygons[i] = v;
      }    
      return mesh_ptr;
}

void execute(const PtrStepSz<const unsigned short>& depth, bool has_data)
{        
    bool has_image = false;
    if (has_data)
    {
        depth_device_.upload (depth.data, depth.step, depth.rows, depth.cols);
    
        //run kinfu algorithm
        has_image = kinfu_ (depth_device_);
    }
}

void source_cb1_device(const boost::shared_ptr<openni_wrapper::DepthImage>& depth_wrapper)  
{        
    {
        boost::mutex::scoped_try_lock lock(data_ready_mutex_);
        if (!lock)
            return;
      
        depth_.cols = depth_wrapper->getWidth();
        depth_.rows = depth_wrapper->getHeight();
        depth_.step = depth_.cols * depth_.elemSize();

        source_depth_data_.resize(depth_.cols * depth_.rows);
        depth_wrapper->fillDepthImageRaw(depth_.cols, depth_.rows, &source_depth_data_[0]);
        depth_.data = &source_depth_data_[0];     
    }
    data_ready_cond_.notify_one();
}

vector<string> getPcdFilesInDir(const string& directory)
{
    namespace fs = boost::filesystem;
    fs::path dir(directory);
 
    std::cout << "path: " << directory << std::endl;
    if (directory.empty() || !fs::exists(dir) || !fs::is_directory(dir))
        PCL_THROW_EXCEPTION (pcl::IOException, "No valid PCD directory given!\n");
    
    vector<string> result;
    fs::directory_iterator pos(dir);
    fs::directory_iterator end;           

    for(; pos != end ; ++pos)
    {
        if (fs::is_regular_file(pos->status()) )
        {
            if (fs::extension(*pos) == ".pcd")
            {
                result.push_back (pos->path ().string ());
                cout << "added: " << result.back() << endl;
            }
        }
    }
    
    return result;  
}

void test ()
{
    viewer_ = boost::shared_ptr<pcl::visualization::PCLVisualizer>(new pcl::visualization::PCLVisualizer("3D Viewer"));
    viewer_->setBackgroundColor(0, 0, 0);
    viewer_->spin();
}

void parsePCDFile (string filename)
{
    pcl::PCDReader reader;
    int pcd_version;
    pcl::PCLPointCloud2 cloud2;
    Eigen::Vector4f origin;
    Eigen::Quaternionf orientation;
    bool valid = (reader.read (filename, cloud2, origin, orientation, pcd_version) == 0);

    // pcd_grabber::publish
    pcl::PointCloud<PointXYZRGBA>::Ptr cloud (new pcl::PointCloud<PointXYZRGBA> ());
    pcl::fromPCLPointCloud2 (cloud2, *cloud);
    cloud->sensor_origin_ = origin;
    cloud->sensor_orientation_ = orientation;

    // If dataset is not organized, return
    if (!cloud->isOrganized ())
        return;

    boost::shared_ptr<xn::DepthMetaData> depth_meta_data (new xn::DepthMetaData);
    depth_meta_data->AllocateData (cloud->width, cloud->height);
    XnDepthPixel* depth_map = depth_meta_data->WritableData ();
    uint32_t k = 0;
    for (uint32_t i = 0; i < cloud->height; ++i)
    {
        for (uint32_t j = 0; j < cloud->width; ++j)
        {
            depth_map[k] = static_cast<XnDepthPixel> ((*cloud)[k].z * 1000);
            ++k;
        }
    }

    boost::shared_ptr<openni_wrapper::DepthImage> depth_image (new openni_wrapper::DepthImage (depth_meta_data, 0.075f, 525, 0, 0));
    //depth_image_signal_->operator()(depth_image);
    source_cb1_device(depth_image);
}

int main (int argc, char* argv[])
{
    int device = 0;
    pcl::gpu::setDevice (device);
    pcl::gpu::printShortCudaDeviceInfo (device);

    boost::shared_ptr<pcl::PCDGrabber<pcl::PointXYZRGBA>> capture;
  
    bool triggered_capture = true;

    float fps_pcd = triggered_capture ? 0 : 15.0f;
    vector<string> pcd_files = getPcdFilesInDir("c:/code/pcl/data/office");

    // Sort the read files by name
    sort (pcd_files.begin (), pcd_files.end ());
    capture.reset (new pcl::PCDGrabber<pcl::PointXYZRGBA> (pcd_files, fps_pcd, false));
    
    std::vector<float> depth_intrinsics;

    // KinfuApp
    //Init Kinfu Tracker
    Eigen::Vector3f volume_size = Vector3f::Constant (3.f/*meters*/);    
    kinfu_.volume().setSize (volume_size);

    Eigen::Matrix3f R = Eigen::Matrix3f::Identity ();   // * AngleAxisf( pcl::deg2rad(-30.f), Vector3f::UnitX());
    Eigen::Vector3f t = volume_size * 0.5f - Vector3f (0, 0, volume_size (2) / 2 * 1.2f);

    Eigen::Affine3f pose = Eigen::Translation3f (t) * Eigen::AngleAxisf (R);

    kinfu_.setInitalCameraPose (pose);
    kinfu_.volume().setTsdfTruncDist (0.030f/*meters*/);    
    kinfu_.setIcpCorespFilteringParams (0.1f/*meters*/, sin ( pcl::deg2rad(20.f) ));
    //kinfu_.setDepthTruncationForICP(5.f/*meters*/);
    kinfu_.setCameraMovementThreshold(0.001f);

    //Init KinfuApp
    tsdf_cloud_ptr_ = pcl::PointCloud<pcl::PointXYZRGBA>::Ptr (new pcl::PointCloud<pcl::PointXYZRGBA>);

    boost::thread thread = boost::thread(test);
    MarchingCubes::Ptr marching_cubes_ = MarchingCubes::Ptr( new MarchingCubes() );

#if 1
    for (std::vector<string>::size_type i = 0; i < pcd_files.size(); ++i)
    {
        cout << pcd_files.size()-i << endl;
        parsePCDFile(pcd_files[i]);
        execute (depth_, true);
    }
#endif

#if 0
    // startMainLoop
    using namespace openni_wrapper;
    typedef boost::shared_ptr<DepthImage> DepthImagePtr;
    typedef boost::shared_ptr<Image> ImagePtr;

    boost::function<void (const DepthImagePtr&)> func2 = boost::bind (&source_cb1_device, _1);

    boost::signals2::connection c = capture->registerCallback (func2);
    
    boost::unique_lock<boost::mutex> lock(data_ready_mutex_);

    if (!triggered_capture)
        capture->start (); // Start stream

    int count = capture->size();
    
    //pcl::visualization::PCLVisualizer viewer("3D Viewer");
	//viewer.setBackgroundColor (0, 0, 0);
    bool added = false;
    while(count)
    {
        if (triggered_capture)
            capture->start ();

        bool has_data = data_ready_cond_.timed_wait (lock, boost::posix_time::millisec(100));
        if (has_data)
        {
            cout << count << endl;
            --count;
    
            /*DeviceArray<PointXYZRGBA> triangles_device = marching_cubes_->run(kinfu_.volume(), triangles_buffer_device_);
            mesh_ptr_ = convertToMesh(triangles_device);
            if (!added && mesh_ptr_ != NULL)
            {
                added = true;
                viewer_->addPolygonMesh(*mesh_ptr_);
                viewer_->resetCamera();
            }
            else if (mesh_ptr_ != NULL)
            {
                viewer_->updatePolygonMesh(*mesh_ptr_);
            }*/
        }
        execute (depth_, has_data);
    }
#endif
    
    DeviceArray<PointXYZ> triangles_device = marching_cubes_->run(kinfu_.volume(), triangles_buffer_device_);
    mesh_ptr_ = convertToMesh(triangles_device);
    
    //c.disconnect();

    // visualization
	viewer_->addPolygonMesh(*mesh_ptr_);
    viewer_->resetCamera();
	/*while (!viewer.wasStopped ())
	{
		boost::this_thread::sleep (boost::posix_time::microseconds (100000));
	}*/
    thread.join();

    return 0;
}