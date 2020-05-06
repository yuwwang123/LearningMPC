
#include <ros/ros.h>
#include <ros/package.h>
#include <sensor_msgs/LaserScan.h>
#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/Pose.h>
#include <geometry_msgs/Point.h>
#include <ackermann_msgs/AckermannDriveStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <tf/transform_listener.h>

// standard
#include <cmath>
#include <vector>
#include <array>
#include <iostream>
#include <fstream>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <LearningMPC/spline.h>
#include <LearningMPC/CSVReader.h>
#include <Eigen/Dense>
#include <nav_msgs/OccupancyGrid.h>
#include <LearningMPC/occupancy_grid.h>

const int SEARCH_RANGE = 10;
const double HALF_WIDTH_MAX = 0.8;
using namespace std;

struct Point_ref{
    double x;
    double y;
    double theta;  // theta represents progress along centerline_points
    double left_half_width;
    double right_half_width;
};

class Track{
public:
    vector<Point_ref> centerline_points;
    tk::spline X_spline;
    tk::spline Y_spline;
    double length;
    double space;
    nav_msgs::OccupancyGrid map;
    vector<double> width_info;

    Track(string wp_file_name, nav_msgs::OccupancyGrid& map, bool sparse=false) : space(0.05), map(map) {
        space = 0.05;
        centerline_points.clear();
        vector<geometry_msgs::Point> waypoints;

        // Get the data from CSV File
        CSVReader reader(wp_file_name);
        std::vector<std::vector<std::string> > dataList = reader.getData();
        for(std::vector<std::string> vec : dataList){
            geometry_msgs::Point wp;
            wp.x = std::stof(vec.at(0));
            wp.y = std::stof(vec.at(1));
            waypoints.push_back(wp);
        }
        /*** process raw waypoints data, extract equally spaced points ***/
        int curr = 0; // current point
        int next =1;  // next point
        Point_ref p_start;
        p_start.x = waypoints.at(0).x;
        p_start.y = waypoints.at(0).y;
        p_start.theta = 0.0;
        centerline_points.push_back(p_start);
        float theta = 0.0;

        while(next < waypoints.size()){

            float dist_to_next_wp = (float)getEucliDist(waypoints.at(next).x, waypoints.at(next).y, waypoints.at(curr).x, waypoints.at(curr).y);
            float dist_to_start = (float)getEucliDist(waypoints.at(next).x, waypoints.at(next).y, waypoints.at(0).x, waypoints.at(0).y);
            if (dist_to_next_wp > space){
                theta += dist_to_next_wp;
                Point_ref p;
                p.x = waypoints.at(next).x; p.y = waypoints.at(next).y; p.theta = theta;
                p.left_half_width = p.right_half_width = HALF_WIDTH_MAX;
                centerline_points.push_back(p);
                curr = next;
            }
            next++;
            // terminate when finished a lap
            if (next > waypoints.size()/2 && dist_to_start<space){
                break;
            }
        }
        double dist_from_last_cp_to_start = getEucliDist(centerline_points.back().x, centerline_points.back().y, waypoints.at(0).x, waypoints.at(0).y);

        length = theta + dist_from_last_cp_to_start;
        Point_ref p_last;
        p_last.x = waypoints.at(0).x;
        p_last.y = waypoints.at(0).y;
        p_last.theta = length;

        centerline_points.push_back(p_last);   //close the loop

        vector<double> X;
        vector<double> Y;
        vector<double> thetas;
        for (const Point_ref & point : centerline_points) {
            X.push_back(point.x);
            Y.push_back(point.y);
            thetas.push_back(point.theta);
        }

        X_spline.set_points(thetas, X);
        Y_spline.set_points(thetas, Y);

        /** if sparse, reinitialize centerline points such that they are densely and equally spaced **/
        if (sparse) {
            double s=0;
            centerline_points.clear();
            while (s<length){
                Point_ref p;
                p.x = X_spline(s);
                p.y = Y_spline(s);
                p.theta = s;
                s += space;
                centerline_points.push_back(p);
            }
        }
        initialize_width();
    }

    void initialize_width() {
        using namespace Eigen;

        Vector2d p_right, p_left;
        for (Point_ref & point: centerline_points) {
            double dx_dtheta = x_eval_d(point.theta);
            double dy_dtheta = y_eval_d(point.theta);
            Vector2d cur_point(point.x, point.y);

            int t = 0;
            //search right until hit right track boundary
            while(true){
                float x = (cur_point + (float)t * map.info.resolution * Vector2d(dy_dtheta, -dx_dtheta).normalized())(0);
                float y = (cur_point + (float)t * map.info.resolution * Vector2d(dy_dtheta, -dx_dtheta).normalized())(1);
                if(occupancy_grid::is_xy_occupied(map, x, y)){
                    p_right(0) = x;
                    p_right(1) = y;
                    break;
                }
                t++;
            }
            t=0;
            //search left until hit right track boundary
            while(true){
                float x = (cur_point + (float)t * map.info.resolution * Vector2d(-dy_dtheta, dx_dtheta).normalized())(0);
                float y = (cur_point + (float)t * map.info.resolution * Vector2d(-dy_dtheta, dx_dtheta).normalized())(1);
                if(occupancy_grid::is_xy_occupied(map, x, y)){
                    p_left(0) = x;
                    p_left(1) = y;
                    break;
                }
                t++;
            }
            point.left_half_width = (p_left-cur_point).norm();
            point.right_half_width = (p_right-cur_point).norm();
        }
    }

    static double getEucliDist(const double & p1x, const double & p1y, const double & p2x, const double & p2y) {
        return sqrt(pow(p1x-p2x, 2)
                    +pow(p1y-p2y, 2));
    }

//    double findTheta(double x, double y, double theta_guess, bool global_search= false) {
    double findTheta(double x, double y) {
        /* return: projected theta along centerline_points, theta is between [0, length]
        * i.e. return the progress along centerline
        * */
        int min_ind = 0;
        double min_dist = DBL_MAX;

        for (size_t i = 0; i < centerline_points.size(); i++) {
            double dist = getEucliDist(x, y, centerline_points.at(i).x, centerline_points.at(i).y);
            if (dist < min_dist) {
                min_dist = dist;
                min_ind = i;
            }
        }

        return min_ind * space;
    }

    void wrapTheta(double& theta){
        while (theta > length) {
            theta -= length;
        }
        while (theta < 0) {
            theta += length;
        }
    }

    double x_eval(double theta){
        wrapTheta(theta);
        return X_spline(theta);
    }

    double y_eval(double theta){
        wrapTheta(theta);
        return Y_spline(theta);
    }

    double x_eval_d(double theta){
        wrapTheta(theta);
        return X_spline.eval_d(theta);
    }

    double y_eval_d(double theta){
        wrapTheta(theta);
        return Y_spline.eval_d(theta);
    }

    double x_eval_dd(double theta){
        wrapTheta(theta);
        return X_spline.eval_dd(theta);
    }

    double y_eval_dd(double theta){
        wrapTheta(theta);
        return Y_spline.eval_dd(theta);
    }

    // given a progress 'theta', compute the tangent line angle 'Phi'
    double getPhi(double theta) {
        wrapTheta(theta);

        double dx_dtheta = X_spline.eval_d(theta);
        double dy_dtheta = Y_spline.eval_d(theta);

        return atan2(dy_dtheta, dx_dtheta);
    }

    double getLeftHalfWidth(const double & theta) {
        // wrapTheta(theta);
        int ind = static_cast<int>(floor(theta / space));
        // clamp ind between 0 to centerline size to prevent segmentation fault
        ind = max(0, min(int(centerline_points.size()-1), ind));
        return centerline_points.at(ind).left_half_width;
    }

    double getRightHalfWidth(const double & theta) {
        // wrapTheta(theta);
        int ind = static_cast<int>(floor(theta/space));
        // clamp ind between 0 to centerline size to prevent segmentation fault
        ind = max(0, min(int(centerline_points.size()-1), ind));
        return centerline_points.at(ind).right_half_width;
    }

    void setHalfWidth(double theta, double left_val, double right_val){
        // wrapTheta(theta);
        int ind = static_cast<int>(floor(theta/space));
        ind = max(0, min(int(centerline_points.size()-1), ind));
        centerline_points.at(ind).left_half_width = left_val;
        centerline_points.at(ind).right_half_width = right_val;
    }

    double getcenterline_pointsCurvature(double theta){
        return (X_spline.eval_d(theta)*Y_spline.eval_dd(theta) - Y_spline.eval_d(theta)*X_spline.eval_dd(theta))/
               pow((pow(X_spline.eval_d(theta),2) + pow(Y_spline.eval_d(theta),2)), 1.5);
    }

    double getcenterline_pointsRadius(double theta){
        return 1.0/(getcenterline_pointsCurvature(theta));
    }
};