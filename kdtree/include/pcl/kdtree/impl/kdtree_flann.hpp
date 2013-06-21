/*
 * Software License Agreement (BSD License)
 *
 *  Point Cloud Library (PCL) - www.pointclouds.org
 *  Copyright (c) 2009-2011, Willow Garage, Inc.
 *  Copyright (c) 2012-, Open Perception, Inc.
 *
 *  All rights reserved.
 *
 *  Redistribution and use in source and binary forms, with or without
 *  modification, are permitted provided that the following conditions
 *  are met:
 *
 *   * Redistributions of source code must retain the above copyright
 *     notice, this list of conditions and the following disclaimer.
 *   * Redistributions in binary form must reproduce the above
 *     copyright notice, this list of conditions and the following
 *     disclaimer in the documentation and/or other materials provided
 *     with the distribution.
 *   * Neither the name of the copyright holder(s) nor the names of its
 *     contributors may be used to endorse or promote products derived
 *     from this software without specific prior written permission.
 *
 *  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 *  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 *  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 *  FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 *  COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 *  INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 *  BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 *  LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 *  CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 *  LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 *  ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *  POSSIBILITY OF SUCH DAMAGE.
 *
 */

#ifndef PCL_KDTREE_KDTREE_IMPL_NANO_FLANN_H_
#define PCL_KDTREE_KDTREE_IMPL_NANO_FLANN_H_

#include <cstdio>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/kdtree/impl/nanoflann.hpp>
#include <pcl/console/print.h>

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::KdTreeFLANN<PointT>::KdTreeFLANN (bool sorted)
: pcl::KdTree<PointT> (sorted)
, flann_index_ ()
, index_mapping_ (), identity_mapping_ (false)
, dim_ (0), total_nr_points_ (0)
, param_k_ (new nanoflann::SearchParams (-1 , epsilon_))
, param_radius_ (new nanoflann::SearchParams (-1, epsilon_, sorted))
{
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT>
pcl::KdTreeFLANN<PointT>::KdTreeFLANN (const KdTreeFLANN<PointT> &k)
: pcl::KdTree<PointT> (false)
, flann_index_ ()
, index_mapping_ (), identity_mapping_ (false)
, dim_ (0), total_nr_points_ (0)
, param_k_ (new nanoflann::SearchParams (-1 , epsilon_))
, param_radius_ (new nanoflann::SearchParams (-1, epsilon_, false))
{
    *this = k;
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::KdTreeFLANN<PointT>::setEpsilon (float eps)
{
    epsilon_ = eps;
    param_k_.reset (new nanoflann::SearchParams (-1 , epsilon_));
    param_radius_.reset (new nanoflann::SearchParams (-1 , epsilon_, sorted_));
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::KdTreeFLANN<PointT>::setSortedResults (bool sorted)
{
    sorted_ = sorted;
    
    param_k_.reset (new nanoflann::SearchParams (-1, epsilon_));
    param_radius_.reset (new nanoflann::SearchParams (-1, epsilon_, sorted_));
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> void
pcl::KdTreeFLANN<PointT>::setInputCloud (const PointCloudConstPtr &cloud, const IndicesConstPtr &indices)
{
    epsilon_ = 0.0f;   // default error bound value
    dim_ = point_representation_->getNumberOfDimensions (); // Number of dimensions - default is 3 = xyz
    
    input_   = cloud;
    indices_ = indices;
    
    // Allocate enough data
    if (!input_)
    {
        PCL_ERROR ("[pcl::KdTreeFLANN::setInputCloud] Invalid input!\n");
        return;
    }
    
    pcl::PointIndices valid_indices;
    index_mapping_.clear();
    if (indices != NULL)
    {
        index_mapping_.reserve(indices_->size());
        identity_mapping_ = indices_->size() == input_->points.size();
        
        for (std::vector<int>::const_iterator iIt = indices_->begin (); iIt != indices_->end (); ++iIt)
        {
            // Check if the point is invalid
            if (!point_representation_->isValid (input_->points[*iIt])) {
                identity_mapping_ = false;
                continue;
            }
            
            // map from 0 - N -> indices [0] - indices [N]
            index_mapping_.push_back (*iIt);  // If the returned index should be for the indices vector
            valid_indices.indices.push_back(*iIt);
            if (*iIt != iIt - indices_->begin()) {
                identity_mapping_ = false;
            }
        }
    }
    else
    {
        index_mapping_.reserve(input_->points.size());
        
        identity_mapping_ = true;
        
        for (int cloud_index = 0; cloud_index < input_->points.size(); ++cloud_index)
        {
            // Check if the point is invalid
            if (!point_representation_->isValid (input_->points[cloud_index]))
            {
                identity_mapping_ = false;
                continue;
            }
            
            index_mapping_.push_back (cloud_index);
            valid_indices.indices.push_back(cloud_index);
        }
    }
    
    int stride = sizeof (PointT) / sizeof (float);
    
    if (identity_mapping_) {
        data_ = input_->getMatrixXfMap(dim_, stride, 0).transpose();
    } else {
        PointCloud cloud_;
        pcl::copyPointCloud(*input_, valid_indices, cloud_);
        
        data_ = cloud_.getMatrixXfMap(dim_, stride, 0).transpose();
    }
    
    total_nr_points_ = static_cast<int> (data_.rows());
    if (total_nr_points_ == 0)
    {
        PCL_ERROR ("[pcl::KdTreeFLANN::setInputCloud] Cannot create a KDTree with an empty input cloud!\n");
        return;
    }
    
    flann_index_.reset (new FLANNIndex(
                                       dim_,
                                       data_,
                                       15 // max 15 points/leaf
                                       ));
    flann_index_->index->buildIndex ();
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> int
pcl::KdTreeFLANN<PointT>::nearestKSearch (const PointT &point, int k,
                                              std::vector<int> &k_indices,
                                              std::vector<float> &k_distances) const
{
    assert (point_representation_->isValid (point) && "Invalid (NaN, Inf) point coordinates given to nearestKSearch!");
    
    if (k > total_nr_points_)
        k = total_nr_points_;
    
    k_indices.resize (k);
    k_distances.resize (k);
    
    std::vector<float> query (dim_);
    point_representation_->vectorize (static_cast<PointT> (point), query);
    
    nanoflann::KNNResultSet<float, int> resultSet(k);
    resultSet.init(&k_indices.front(), &k_distances.front());
    flann_index_->index->findNeighbors(resultSet, &query.front(), *param_k_);
    
    // Do mapping to original point cloud
    if (!identity_mapping_)
    {
        for (size_t i = 0; i < static_cast<size_t> (k); ++i)
        {
            int& neighbor_index = k_indices[i];
            neighbor_index = index_mapping_[neighbor_index];
        }
    }
    
    return (k);
}

///////////////////////////////////////////////////////////////////////////////////////////
template <typename PointT> int
pcl::KdTreeFLANN<PointT>::radiusSearch (const PointT &point, double radius, std::vector<int> &k_indices,
                                            std::vector<float> &k_sqr_dists, unsigned int max_nn) const
{
    assert (point_representation_->isValid (point) && "Invalid (NaN, Inf) point coordinates given to radiusSearch!");
    
    std::vector<float> query (dim_);
    point_representation_->vectorize (static_cast<PointT> (point), query);
    
    if (max_nn == 0 || max_nn > static_cast<unsigned int> (total_nr_points_))
        max_nn = total_nr_points_;
    
    std::vector<std::pair<int, float> > indices;
    
    nanoflann::RadiusResultSet<float, int> resultSet(radius*radius, indices);
    flann_index_->index->findNeighbors(resultSet, &query.front(), *param_radius_);
    
    if (param_radius_->sorted)
        std::sort(indices.begin(), indices.end(), nanoflann::IndexDist_Sorter() );
    
    int neighbors_in_radius = static_cast<int>(resultSet.size());
    
    if (neighbors_in_radius > max_nn) {
        neighbors_in_radius = max_nn;
    }
    
    k_indices.resize(neighbors_in_radius);
    k_sqr_dists.resize(neighbors_in_radius);
    
    for (int i = 0;i<neighbors_in_radius;i++) {
        k_indices[i] = indices[i].first;
        k_sqr_dists[i] = indices[i].second;
    }
    
    // Do mapping to original point cloud
    if (!identity_mapping_)
    {
        for (int i = 0; i < neighbors_in_radius; ++i)
        {
            int& neighbor_index = k_indices[i];
            neighbor_index = index_mapping_[neighbor_index];
        }
    }
    
    return (neighbors_in_radius);
}

#define PCL_INSTANTIATE_KdTreeFLANN(T) template class PCL_EXPORTS pcl::KdTreeFLANN<T>;

#endif  //#ifndef _PCL_KDTREE_KDTREE_IMPL_NANO_FLANN_H_
