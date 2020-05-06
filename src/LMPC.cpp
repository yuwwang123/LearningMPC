
//ros library
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
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <visualization_msgs/MarkerArray.h>

//std library
#include <cmath>
#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <random>

//LMPC related and own header files
#include <LearningMPC/track.h>
#include <Eigen/Sparse>
#include "OsqpEigen/OsqpEigen.h"
#include <unsupported/Eigen/MatrixFunctions>
#include <LearningMPC/car_params.h>

const int nx = 6; //# of states
const int nu = 2; //# of control inputs

using namespace std;
using namespace Eigen;

struct Sample{
    Matrix<double,nx,1> x; // state
    Matrix<double,nu,1> u; // control inputs
    double s;              // corresponding progress 's' on the track
    int timestep;              // timestep, not absolute time
    int iter;
    int cost;
};

enum rviz_id{
    CENTERLINE,
    CENTERLINE_POINTS,
    CENTERLINE_SPLINE,
    PREDICTION,
    BORDERLINES,
    SAFE_SET,
    TERMINAL_CANDIDATE,
    DEBUG
};

class LMPC{
public:
    LMPC(ros::NodeHandle& nh); //constructor
    void run();

private:
    // publishers, subscribers
    ros::NodeHandle nh_;
    ros::Publisher track_viz_pub_;
    ros::Publisher LMPC_viz_pub_;
    ros::Publisher drive_pub_;
    ros::Publisher debugger_pub_;

    ros::Subscriber odom_sub_;
    ros::Subscriber rrt_sub_;
    ros::Subscriber map_sub_;

    /*Paramaters with values defined in "Lmpc_params.yaml"*/
    CarParams car; //struct from car_params.h
    string pose_topic;
    string drive_topic;
    string wp_file_name;
    string initial_safe_set_file_name;
    string lap_time_data_file_name;
    string x_y_velocity_data_file_name;
    double WAYPOINT_SPACE;
    double MAP_MARGIN;

    int N; //mpc horizon length, N timesteps ahead
    double Ts; //time duration of every timestep
    int K_NEAR; //# of nearest neighbors
    double SPEED_MAX;
    double STEER_MAX;
    double ACCELERATION_MAX;
    double DECELERATION_MAX;
    double VEL_THRESHOLD;
    int INITIAL_ITER;
    int SAFETY_SET_ITERS;

    // MPC params
    double q_s; // elements in Q matrix used in defining slack variable cost, will be high
    double r_accel; // elements in R matrix used in defining control cost
    double r_steer; // elements in R matrix
    Matrix<double, nu, nu> R; //R matrix in objective function, coefficient of u: control inputs

    Track* track_;
    //odometry
    tf::Transform tf_;
    tf::Vector3 car_pos_;
    double yaw_;
    double vel_;
    double yawdot_;
    double slip_angle_;
    double s_prev_;
    double s_curr_;

    // use dynamic model or not
    bool use_dyn_;

    //Sample Safe set
    vector<vector<Sample>> SS_; // each element is a set of Samples (trajectory) in one lap
    vector<Sample> curr_trajectory_;
    int iter_;
    int timestep_;
    Matrix<double,nx,1> terminal_state_pred_;

    // map info
    nav_msgs::OccupancyGrid map_;
    nav_msgs::OccupancyGrid map_updated_;

    VectorXd QPSolution_;
    bool first_run_ = true;
    vector<geometry_msgs::Point> border_lines_;


    void getParameters(ros::NodeHandle& nh);
    void init_occupancy_grid();
    void init_SS_from_data(string data_file);
    void visualize_centerline();
    int initialize_QPSolution(int iter);
    void odom_callback(const nav_msgs::Odometry::ConstPtr &odom_msg);
    void add_point();
    void select_trajectory();
    void simulate_dynamics(Matrix<double,nx,1>& state, Matrix<double,nu,1>& input, double dt, Matrix<double,nx,1>& new_state);

    void solve_MPC(const Matrix<double,nx,1>& terminal_candidate);

    void get_linearized_dynamics(Matrix<double,nx,nx>& Ad, Matrix<double,nx, nu>& Bd, Matrix<double,nx,1>& hd,
            Matrix<double,nx,1>& x_op, Matrix<double,nu,1>& u_op, bool use_dyn);

    Vector3d global_to_track(double x, double y, double yaw, double s);
    Vector3d track_to_global(double e_y, double e_yaw, double s);

    void applyControl();
    void visualize_mpc_solution(const vector<Sample>& convex_safe_set, const Matrix<double,nx,1>& terminal_candidate);

    Matrix<double,nx,1> select_terminal_candidate();
    void select_convex_safe_set(vector<Sample>& convex_safe_set, int iter_start, int iter_end, double s);
    int find_nearest_point(vector<Sample>& trajectory, double s);
    void update_cost_to_go(vector<Sample>& trajectory);
    Matrix<double,nx,1> get_nonlinear_dynamics(Matrix<double,nx,1>& x, Matrix<double,nu,1>& u,  double t);

    std::ofstream myfile;
    void record_lap_time();
    void record_xyvelo(double x, double y, float vel);

};

//constructor, run once only
LMPC::LMPC(ros::NodeHandle &nh): nh_(nh){

    getParameters(nh_);
    init_occupancy_grid();
    track_ = new Track(wp_file_name, map_, true);

    odom_sub_ = nh_.subscribe(pose_topic, 10, &LMPC::odom_callback, this);
    drive_pub_ = nh_.advertise<ackermann_msgs::AckermannDriveStamped>(drive_topic, 1);
   // rrt_sub_ = nh_.subscribe("path_found", 1, &LMPC::rrt_path_callback, this);
  //  map_sub_ = nh_.subscribe("map_updated", 1, &LMPC::map_callback, this);

    track_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("track_centerline", 1);

    LMPC_viz_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("LMPC", 1);
    debugger_pub_ = nh_.advertise<visualization_msgs::Marker>("Debugger", 1);

    //get odometry message and assign it to 'x' and 'y'
    nav_msgs::Odometry odom_msg;
    boost::shared_ptr<nav_msgs::Odometry const> odom_ptr;
    odom_ptr = ros::topic::waitForMessage<nav_msgs::Odometry>("odom", ros::Duration(5));
    if (odom_ptr == nullptr) {
        cout<< "fail to receive odom message!"<<endl;
    } else {
        odom_msg = *odom_ptr;
    }
    double x = odom_msg.pose.pose.position.x;
    double y = odom_msg.pose.pose.position.y;

    // current state of the car
    s_prev_ = track_->findTheta(x, y);
    car_pos_ = tf::Vector3(x, y, 0.0);
    yaw_ = tf::getYaw(odom_msg.pose.pose.orientation);
    vel_ = odom_msg.twist.twist.linear.x;
    yawdot_ = 0;
    slip_angle_ = 0;

    iter_ = INITIAL_ITER;
    use_dyn_ = false;
    init_SS_from_data(initial_safe_set_file_name);


}

// load parameters from Lmpc_params.yaml
void LMPC::getParameters(ros::NodeHandle &nh) {
    nh.getParam("pose_topic", pose_topic);
    nh.getParam("drive_topic", drive_topic);
    nh.getParam("wp_file_name", wp_file_name);
    nh.getParam("initial_safe_set_file_name", initial_safe_set_file_name);
    nh.getParam("lap_time_data_file_name", lap_time_data_file_name);
    nh.getParam("x_y_velocity_data_file_name", x_y_velocity_data_file_name);

    nh.getParam("N",N);
    nh.getParam("Ts",Ts);
    nh.getParam("K_NEAR", K_NEAR);
    nh.getParam("ACCELERATION_MAX", ACCELERATION_MAX);
    nh.getParam("DECELERATION_MAX", DECELERATION_MAX);
    nh.getParam("SPEED_MAX", SPEED_MAX);
    nh.getParam("STEER_MAX", STEER_MAX);
    nh.getParam("VEL_THRESHOLD", VEL_THRESHOLD);
    nh.getParam("INITIAL_ITER", INITIAL_ITER);
    nh.getParam("SAFETY_SET_ITERS", SAFETY_SET_ITERS);

    nh.getParam("WAYPOINT_SPACE", WAYPOINT_SPACE);
    nh.getParam("r_accel",r_accel);
    nh.getParam("r_steer",r_steer);
    nh.getParam("q_s",q_s);
    R.setZero();
    R.diagonal() << r_accel, r_steer;
    nh.getParam("MAP_MARGIN",MAP_MARGIN);

    nh.getParam("wheelbase", car.wheelbase);
    nh.getParam("friction_coeff", car.friction_coeff);
    nh.getParam("height_cg", car.h_cg);
    nh.getParam("l_cg2rear", car.l_r);
    nh.getParam("l_cg2front", car.l_f);
    nh.getParam("C_S_front", car.cs_f);
    nh.getParam("C_S_rear", car.cs_r);
    nh.getParam("moment_inertia", car.I_z);
    nh.getParam("mass", car.mass);
}

int compare_s(Sample& s1, Sample& s2){
    return (s1.s< s2.s);
}

// get the map data from ros
void LMPC::init_occupancy_grid(){
    boost::shared_ptr<nav_msgs::OccupancyGrid const> map_ptr;
    map_ptr = ros::topic::waitForMessage<nav_msgs::OccupancyGrid>("map", ros::Duration(5.0));

    if (map_ptr == nullptr) {
        ROS_INFO("No map received");
    } else{
        map_ = *map_ptr;
        map_updated_ = map_;
        ROS_INFO("Map received");
    }
    ROS_INFO("Initializing occupancy grid for map ...");
    occupancy_grid::inflate_map(map_, (float)MAP_MARGIN);
}

// Initialize all member variables for struct 'Sample'
void LMPC::init_SS_from_data(string data_file) {

    // Get the data from CSV File
    CSVReader reader(data_file);
    vector<vector<string>> dataList = reader.getData();
    SS_.clear();
    // Print the content of row by row on screen
    int timestep_prev = 0;
    int iter = 0;
    vector<Sample> traj; // the trajectory of one lap
    for (const vector<string> & vec : dataList) {
        Sample sample;
        sample.timestep = (int)std::stof(vec.at(0));

        // check if it's a new lap, if is, assign cost to go to this trajectory and put the traj into Safe Set
        if (sample.timestep < timestep_prev) {
            iter++;
            update_cost_to_go(traj); // assign the # of  to sample.cost
            SS_.push_back(traj);
            traj.clear();
        }

        sample.x(0) = std::stof(vec.at(1)); // x
        sample.x(1) = std::stof(vec.at(2)); // y
        sample.x(2) = std::stof(vec.at(3)); // yaw
        sample.x(3) = std::stof(vec.at(4)); // velocity
        sample.x(4) = 0; // yaw_dot
        sample.x(5) = 0; // slip_angle

        sample.u(0) = std::stof(vec.at(5));
        sample.u(1) = std::stof(vec.at(6));

        sample.s = std::stof(vec.at(7)); //progress
        sample.iter = iter;
        traj.push_back(sample);
        timestep_prev = sample.timestep;
    }
    update_cost_to_go(traj);
    SS_.push_back(traj);
}

void LMPC::odom_callback(const nav_msgs::Odometry::ConstPtr & odom_msg) {
    visualize_centerline(); // keep publishing centerline
    /** process pose info, assign values to state (x, y, yaw, velocity, yaw_dot, slip_angle, and s) **/
    double x = odom_msg->pose.pose.position.x;
    double y = odom_msg->pose.pose.position.y;
    s_curr_ = track_->findTheta(x, y);
    car_pos_ = tf::Vector3(x, y, 0.0);
    yaw_ = tf::getYaw(odom_msg->pose.pose.orientation);
    vel_ = sqrt(pow(odom_msg->twist.twist.linear.x,2) + pow(odom_msg->twist.twist.linear.y,2));
    yawdot_ = odom_msg->twist.twist.angular.z;
    slip_angle_ = atan2(odom_msg->twist.twist.linear.y, odom_msg->twist.twist.linear.x);

    /** STATE MACHINE: check if dynamic model should be used based on current speed **/
    // single track model is not applicable to low speed vehicle as from the dynamic model source
    // thus, low speed, kinematic model; high speed, single track model;
    if ((!use_dyn_) && (vel_ > VEL_THRESHOLD) && (iter_ > INITIAL_ITER)) {
        use_dyn_ = true;
    }
    if(use_dyn_ && (vel_< VEL_THRESHOLD * 0.7)) {
        use_dyn_ = false;
    }

    // adjust penalty on acceleration and steering according to the speed
    // when velocity gets high, more penalty on high acceleration and high steering angle
//    if (vel_ > 4.5) {
//        R(0,0) = 1.3 * r_accel;
//        R(1,1) = 1.8 * r_steer;
//    }
    record_xyvelo(x, y, vel_);
}

// record x, y, velocity for data analysis
void LMPC::record_xyvelo(double x, double y, float vel){
    myfile.open ("/home/baihong/baihong_ws/src/LearningMPC/LMPC_xyvelo_data.csv", ios::out | ios::app);
    myfile << iter_ << ", "
           << x << ", "
            << y << ", "
            << vel <<  "\n";
    myfile.close();

}

// record the definite ros time for further analysis on lap time
void LMPC::record_lap_time() {
    // record timestep, x, y, yaw, steering angle into 'LMPC_purepursuit_data'
    myfile.open ("/home/baihong/baihong_ws/src/LearningMPC/LMPC_lap_time_data.csv", ios::out | ios::app);
    myfile << iter_ << ", "
           << ros::Time::now() << "\n";
    myfile.close();
}

void LMPC::run() {
    if (first_run_){
        // initialize QPSolution_ from initial Sample Safe Set (using the 2nd iteration)
        initialize_QPSolution(1);
        record_lap_time();
    }

    /******** LMPC MAIN LOOP starts ********/

    /***check if it is new lap***/
    if (s_curr_ < s_prev_ - track_->length / 2) {
        iter_++;
        update_cost_to_go(curr_trajectory_);
        //sort(curr_trajectory_.begin(), curr_trajectory_.end(), compare_s);
        SS_.push_back(curr_trajectory_);
        curr_trajectory_.clear();
     //   initialize_QPSolution(iter_-1);
        timestep_ = 0;
        record_lap_time();
    }

    /*** select terminal state candidate and its convex safe set ***/
    Matrix<double,nx,1> terminal_candidate = select_terminal_candidate();
    /** solve MPC and record current state***/
    solve_MPC(terminal_candidate);
    applyControl();
    add_point();
    /*** store info and advance to next time step***/
    terminal_state_pred_ = QPSolution_.segment<nx>(N*nx);
    s_prev_ = s_curr_;
    timestep_++;
    first_run_ = false;
}


// visualize centerline as blue lines
void LMPC::visualize_centerline() {
    visualization_msgs::Marker spline_dots;
    spline_dots.header.stamp = ros::Time::now();
    spline_dots.header.frame_id = "map";
    spline_dots.id = rviz_id::CENTERLINE_SPLINE;
    spline_dots.ns = "centerline";
    spline_dots.type = visualization_msgs::Marker::LINE_STRIP;
    spline_dots.scale.x = spline_dots.scale.y = 0.02;
    spline_dots.scale.z = 0.02;
    spline_dots.action = visualization_msgs::Marker::ADD;
    spline_dots.pose.orientation.w = 1.0;
    spline_dots.color.b = 1.0;
    spline_dots.color.a = 1.0; //alpha, not contributing to the actual color
    // spline_dots.lifetime = ros::Duration();

    // from progress s, to get world coordinates x, y
    for (float t = 0.0; t < track_->length; t += 0.05f) {
        geometry_msgs::Point p;
        p.x = track_->x_eval(t);
        p.y = track_->y_eval(t);
        spline_dots.points.push_back(p);
    }

    visualization_msgs::MarkerArray markers;
    markers.markers.push_back(spline_dots);
    track_viz_pub_.publish(markers);
}

int LMPC::initialize_QPSolution(int iter) {
     QPSolution_ = VectorXd::Zero((N+1)*nx+ N*nu + (N+1) + (SAFETY_SET_ITERS*K_NEAR) + nx); // size = the # of decision variables
    // the extra 1 is itself?
    for (int i = 0; i < N+1; i++) {
        QPSolution_.segment<nx>(i*nx) = SS_[iter][i].x; //starting from i*nx, last for nx length, assign with state x value
        if (i < N) {
            QPSolution_.segment<nu>((N+1)*nx + i*nu) = SS_[iter][i].u; // assign values to each nu
        }
    }
}

// equ 10 in the paper, select center points to construct convex sets for terminal states
Matrix<double,nx,1> LMPC::select_terminal_candidate(){
    if (first_run_) { // when timestep == 0
        return SS_.back()[N].x; // the j-1th trajectory, timestep N, get all states x
    } else { // when timestep != 0
        return terminal_state_pred_; // take the terminal states from the last timestep, different from paper
    }
}

void LMPC::add_point() {
    Sample point;
    point.x << car_pos_.x(), car_pos_.y(), yaw_, vel_, yawdot_, slip_angle_;

    point.s = s_curr_;
    point.iter = iter_;
    point.timestep = timestep_;
    point.u = QPSolution_.segment<nu>((N+1)*nx);
    curr_trajectory_.push_back(point);
}

// construct a convex safe set from K_NEAR points from each iteration
// i.e. D_l^j(x) from paper
void LMPC::select_convex_safe_set (vector<Sample>& convex_safe_set, int iter_start, int iter_end, double s) {
    for (int it = iter_start; it <= iter_end; it++) {
        int nearest_ind = find_nearest_point(SS_[it], s);
        int lap_cost = SS_[it][0].cost; //cost of the whole lap

        int start_ind = nearest_ind - K_NEAR / 2; // start of a K_NEAR range
        int end_ind = start_ind + K_NEAR - 1;

        vector<Sample> curr_set;
        if (end_ind > SS_[it].size()-1 || start_ind < 0) { // nearest_ind is around finishing line
            if (end_ind > SS_[it].size()-1) { //nearest_ind has not crossed finishing line yet, but front portion of set crossed
                //do nothing
            } else if (start_ind < 0) { //nearest_ind crossed finishing line, back portion of the set behind finishing line
                start_ind += SS_[it].size();
                end_ind += SS_[it].size();
            }

            for (size_t ind = start_ind; ind <= end_ind; ind++) {
                if (ind < SS_[it].size()) {
                    curr_set.push_back(SS_[it][ind]);
                    // modify the cost-to-go for each point before finishing line
                    // to incentivize the car to cross finishing line towards a new lap
                    // i.e. the points behind finishing line have larger cost (+lap_cost), so the new point tends to be
                    // landed in further regions which is with lower cost
                    curr_set[curr_set.size()-1].cost += lap_cost;
                } else {
                    curr_set.push_back(SS_[it][ind - SS_[it].size()]);
                }
            }

            if (curr_set.size()!=K_NEAR) throw;  // for debug

        } else {// no overlapping with finishing line

            for (int ind=start_ind; ind<=end_ind; ind++){
                curr_set.push_back(SS_[it][ind]);
            }
        }
        convex_safe_set.insert(convex_safe_set.end(), curr_set.begin(), curr_set.end());
    }
}

int LMPC::find_nearest_point(vector<Sample>& trajectory, double s) {
    // binary search to find closest point to a given s in the 'trajectory'
    int low = 0;
    int high = (int)trajectory.size()-1;
    while (low <= high) {
        int mid = low + (high - low) / 2;
        if (s == trajectory[mid].s) {
            return mid;
        } else if (s < trajectory[mid].s) {
            high = mid-1;
        } else {
            low = mid+1;
        }
    }
    return abs(trajectory[low].s-s) < (abs(trajectory[high].s-s))? low : high;
}

// each sample's cost-to-go (J) equals its remaining timestep counts to the goal, i.e. time
void LMPC::update_cost_to_go(vector<Sample>& trajectory) {
    trajectory[trajectory.size() - 1].cost = 0; // terminal cost == 0

    for (int i = trajectory.size() - 2; i >= 0; i--) {
        trajectory[i].cost = trajectory[i+1].cost + 1;
    }
}

Vector3d LMPC::global_to_track(double x, double y, double yaw, double s){
    double x_proj = track_->x_eval(s);
    double y_proj = track_->y_eval(s);
    double e_y = sqrt((x-x_proj)*(x-x_proj) + (y-y_proj)*(y-y_proj));
    double dx_ds = track_->x_eval_d(s);
    double dy_ds = track_->y_eval_d(s);
    e_y = dx_ds*(y-y_proj) - dy_ds*(x-x_proj) >0 ? e_y : -e_y;
    double e_yaw = yaw - atan2(dy_ds, dx_ds);
    while(e_yaw > M_PI) e_yaw -= 2*M_PI;
    while(e_yaw < -M_PI) e_yaw += 2*M_PI;

    return Vector3d(e_y, e_yaw, s);
}

Vector3d LMPC::track_to_global(double e_y, double e_yaw, double s){
    double dx_ds = track_->x_eval_d(s);
    double dy_ds = track_->y_eval_d(s);
    Vector2d proj(track_->x_eval(s), track_->y_eval(s));
    Vector2d pos = proj + Vector2d(-dy_ds, dx_ds).normalized()*e_y;
    double yaw = e_yaw + atan2(dy_ds, dx_ds);
    return Vector3d(pos(0), pos(1), yaw);
}

void LMPC::get_linearized_dynamics(Matrix<double,nx,nx>& Ad, Matrix<double,nx, nu>& Bd, Matrix<double,nx,1>& hd,
        Matrix<double,nx,1>& x_op, Matrix<double,nu,1>& u_op, bool use_dyn) {

    double yaw = x_op(2);
    double v = x_op(3);
    double yaw_dot = x_op(4);
    double slip_angle = x_op(5);

    double accel = u_op(0);
    double steer = u_op(1);


    VectorXd states_dot(6), h(6);
    Matrix<double, nx, nx> A, M12;
    Matrix<double, nx, nu> B;

    // linearization, using first-order Taylor approximation
    // state_dot = A*state + B*control
    if (!use_dyn) {
        // Kinematic Model
        states_dot(0) = v * cos(yaw);
        states_dot(1) = v * sin(yaw);
        states_dot(2) = v * tan(steer)/car.wheelbase;
        states_dot(3) = accel;
        states_dot(4) = 0;
        states_dot(5) = 0;

        A <<    0.0, 0.0, -v*sin(yaw),  cos(yaw),                   0.0,  0.0,
                0.0, 0.0,  v*cos(yaw),  sin(yaw),                   0.0,  0.0,
                0.0, 0.0,         0.0,  tan(steer)/car.wheelbase,   0.0,  0.0,
                0.0, 0.0,         0.0,       0.0,                   0.0,  0.0,
                0.0, 0.0,         0.0,       0.0,                   0.0,  0.0,
                0.0, 0.0,         0.0,       0.0,                   0.0,  0.0;

        B <<    0.0, 0.0,
                0.0, 0.0,
                0.0, v / (cos(steer) * cos(steer) * car.wheelbase),
                1.0, 0.0,
                0.0, 0.0,
                0.0, 0.0;
    }
    else{
        // Single Track Dynamic Model

        double g = 9.81;
        double rear_val = g * car.l_r - accel * car.h_cg;
        double front_val = g * car.l_f + accel * car.h_cg;

        states_dot(0) = v * cos(yaw+slip_angle);
        states_dot(1) = v * sin(yaw+slip_angle);
        states_dot(2) = yaw_dot;
        states_dot(3) = accel;
        states_dot(4) = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase)) *
                      (car.l_f * car.cs_f * steer * (rear_val) +
                       slip_angle * (car.l_r * car.cs_r * (front_val) - car.l_f * car.cs_f * (rear_val)) -
                       (yaw_dot/v) * (pow(car.l_f, 2) * car.cs_f * (rear_val) + pow(car.l_r, 2) * car.cs_r * (front_val)));        // yaw_dot dynamics
        states_dot(5) = (car.friction_coeff / (v * (car.l_r + car.l_f))) *
                      (car.cs_f * steer * rear_val - slip_angle * (car.cs_r * front_val + car.cs_f * rear_val) +
                              (yaw_dot/v) * (car.cs_r * car.l_r * front_val - car.cs_f * car.l_f * rear_val)) - yaw_dot;        // slip_angle dynamics

        double dfyawdot_dv, dfyawdot_dyawdot, dfyawdot_dslip, dfslip_dv, dfslip_dyawdot, dfslip_dslip;
        double dfyawdot_da, dfyawdot_dsteer, dfslip_da, dfslip_dsteer;

        dfyawdot_dv = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                * (pow(car.l_f, 2) * car.cs_f * (rear_val) + pow(car.l_r, 2) * car.cs_r * (front_val))
                * yaw_dot / pow(v, 2);

        dfyawdot_dyawdot = -(car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                           * (pow(car.l_f, 2) * car.cs_f * (rear_val) + pow(car.l_r, 2) * car.cs_r * (front_val))/v;

        dfyawdot_dslip = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                            * (car.l_r * car.cs_r * (front_val) - car.l_f * car.cs_f * (rear_val));

        dfslip_dv = -(car.friction_coeff / (car.l_r + car.l_f)) *
                    (car.cs_f * steer * rear_val - slip_angle * (car.cs_r * front_val + car.cs_f * rear_val))/pow(v,2)
                -2*(car.friction_coeff / (car.l_r + car.l_f)) * (car.cs_r * car.l_r * front_val - car.cs_f * car.l_f * rear_val) * yaw_dot/pow(v,3);

        dfslip_dyawdot = (car.friction_coeff / (pow(v,2) * (car.l_r + car.l_f))) * (car.cs_r * car.l_r * front_val - car.cs_f * car.l_f * rear_val) - 1;

        dfslip_dslip = -(car.friction_coeff / (v * (car.l_r + car.l_f)))*(car.cs_r * front_val + car.cs_f * rear_val);

        dfyawdot_da = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase))
                *(-car.l_f*car.cs_f*car.h_cg*steer + car.l_r*car.cs_r*car.h_cg*slip_angle + car.l_f*car.cs_f*car.h_cg*slip_angle
                  - (yaw_dot/v)*(-pow(car.l_f,2)*car.cs_f*car.h_cg) + pow(car.l_r,2)*car.cs_r*car.h_cg);

        dfyawdot_dsteer = (car.friction_coeff * car.mass / (car.I_z * car.wheelbase)) *
                      (car.l_f * car.cs_f * rear_val);

        dfslip_da = (car.friction_coeff / (v * (car.l_r + car.l_f))) *
                (-car.cs_f*car.h_cg*steer - (car.cs_r*car.h_cg - car.cs_f*car.h_cg)*slip_angle +
                (car.cs_r*car.h_cg*car.l_r + car.cs_f*car.h_cg*car.l_f)*(yaw_dot/v));

        dfslip_dsteer = (car.friction_coeff / (v * (car.l_r + car.l_f))) *
                (car.cs_f * rear_val);


        A <<    0.0, 0.0, -v*sin(yaw+slip_angle), cos(yaw+slip_angle),                 0.0,  -v*sin(yaw+slip_angle),
                0.0, 0.0,  v*cos(yaw+slip_angle), sin(yaw+slip_angle),                 0.0,   v*cos(yaw+slip_angle),
                0.0, 0.0,                       0.0,                   0.0,                 1.0,                       0.0,
                0.0, 0.0,                       0.0,                   0.0,                 0.0,                       0.0,
                0.0, 0.0,                       0.0,           dfyawdot_dv,     dfyawdot_dyawdot,  dfyawdot_dslip,
                0.0, 0.0,                       0.0,             dfslip_dv,       dfslip_dyawdot,    dfslip_dslip;

        B <<    0.0, 0.0,
                0.0, 0.0,
                0.0, 0.0,
                1.0, 0.0,
                dfyawdot_da, dfyawdot_dsteer,
                dfslip_da,     dfslip_dsteer;
    }

    /**  Discretize using Zero-Order Hold **/
    Matrix<double,nx+nx,nx+nx> aux, M;
    aux.setZero();
    aux.block<nx,nx>(0,0) << A;
    aux.block<nx,nx>(0, nx) << Matrix<double,nx,nx>::Identity();
    M = (aux*Ts).exp();
    M12 = M.block<nx,nx>(0,nx);
    h = states_dot - (A*x_op + B*u_op);

    Ad = (A*Ts).exp();
    Bd = M12*B;
    hd = M12*h;

}

void wrap_angle(double & angle, const double & angle_ref) {
    while(angle - angle_ref > M_PI) {
        angle -= 2*M_PI;
    }
    while(angle - angle_ref < -M_PI) {
        angle += 2*M_PI;
    }
}

void LMPC::solve_MPC (const Matrix<double,nx,1>& terminal_candidate) {
    vector<Sample> terminal_CSS; // declare a Convex Safety Set (CSS)
    double s_terminal = track_->findTheta(terminal_candidate(0), terminal_candidate(1)); //get progress of selected terminal candidate by providing (x, y)
    select_convex_safe_set(terminal_CSS, iter_ - SAFETY_SET_ITERS, iter_ - 1, s_terminal);

    /** MPC variables: z = [x0, ..., xN, u0, ..., uN-1, s0, ..., sN, lambda0, ....., lambda(SAFETY_SET_ITERS * K_NEAR), s_t1, s_t2, s_t3, s_t4, s_t5, s_t6]*
     *  constraints: dynamics, track bounds, input limits, acceleration limit, slack, lambdas, terminal state, sum of lambda's*/
    // nx: terminal state soft constraint

    int number_of_decision_variables = (N+1)*nx+ N*nu + (N+1) + (SAFETY_SET_ITERS*K_NEAR) + nx;
    int number_of_constraints = (N+1)*nx + 2*(N+1) + N*nu + (N+1) + (N+1) + (SAFETY_SET_ITERS*K_NEAR) + 2*nx + 1;
    // HessianMatrix is a square positive definite matrix, dimension of it is the same as the # of decision variables
    // x-x_ref((N+1)*nx)), u-u_ref(N*nu), slack variable(N+1), lambda(SAFETY_SET_ITERS*K_NEAR), terminal state (nx)
    SparseMatrix<double> HessianMatrix(number_of_decision_variables, number_of_decision_variables);
    // column of constraintMatrix has to match the # of decision variables, constraintMatrix*x <= b
    SparseMatrix<double> constraintMatrix(number_of_constraints, number_of_decision_variables);

    VectorXd gradient(number_of_decision_variables);

    VectorXd lower(number_of_constraints);
    VectorXd upper(number_of_constraints);

    gradient.setZero();
    lower.setZero(); upper.setZero();

    Matrix<double,nx,1> x_k_1;
    Matrix<double,nu,1> u_k_1;
    Matrix<double,nx,nx> Ad;
    Matrix<double,nx,nu> Bd;
    Matrix<double,nx,1> x0, hd; // x0 is current x
    border_lines_.clear();

    if (use_dyn_) { // dynamic model
        x0 <<car_pos_.x(), car_pos_.y(), yaw_, vel_, yawdot_, slip_angle_;
    } else { // kinematic model, do not need yawdot_, slip_angle_
        x0 <<car_pos_.x(), car_pos_.y(), yaw_, vel_, 0.0, 0.0;
    }
    /** make sure there are no discontinuities in yaw**/
    // first check terminal safe_set
    for (Sample & cur_safety_set_point : terminal_CSS) {
        wrap_angle(cur_safety_set_point.x(2), x0(2));
    }
    // also check for previous QPSolution
    for (int i = 0; i < N+1; i++) {
        wrap_angle(QPSolution_(i*nx+2), x0(2));
    }

    for (int i = 0; i < N+1; i++) {        //0 to N
        x_k_1 = QPSolution_.segment<nx>(i*nx); //x in previous iter
        u_k_1 = QPSolution_.segment<nu>((N+1)*nx + i*nu);
        double s_k_1 = track_->findTheta(x_k_1(0), x_k_1(1));
        get_linearized_dynamics(Ad, Bd, hd, x_k_1, u_k_1, use_dyn_);
        /* form Hessian entries*/
        // cost does not depend on x0, only 1 to N
        HessianMatrix.insert((N+1)*nx + N*nu + i, (N+1)*nx + N*nu + i) = q_s;

        if (i<N){ // cost of control input u
            for (int row = 0; row < nu; row++) {
                HessianMatrix.insert((N+1)*nx + i*nu + row, (N+1)*nx + i*nu + row) = R(row, row);
            }
        }

        /* form constraint matrix */
        if (i<N){ // N dynamics constraints
            // Ad
            for (int row = 0; row < nx; row++) {
                for(int col = 0; col < nx; col++) {
                    constraintMatrix.insert((i+1)*nx+row, i*nx+col) = Ad(row,col);
                }
            }
            // Bd
            for (int row=0; row<nx; row++){
                for(int col=0; col<nu; col++){
                    constraintMatrix.insert((i+1)*nx+row, (N+1)*nx+ i*nu+col) = Bd(row,col);
                }
            }
            lower.segment<nx>((i+1)*nx) = -hd;// constant term from dynamics
            upper.segment<nx>((i+1)*nx) = -hd;
        }

        // -I for each x_k+1
        for (int row=0; row<nx; row++) {
            constraintMatrix.insert(i*nx+row, i*nx+row) = -1.0;
        }

        double dx_dtheta = track_->x_eval_d(s_k_1); //gradient, slope of x wrt s
        double dy_dtheta = track_->y_eval_d(s_k_1); //gradient, slope of y wrt s

        constraintMatrix.insert((N+1)*nx+ 2*i, i*nx) = -dy_dtheta;      // a*x
        constraintMatrix.insert((N+1)*nx+ 2*i, i*nx+1) = dx_dtheta;     // b*y
        constraintMatrix.insert((N+1)*nx+ 2*i, (N+1)*nx +N*nu +i) = 1.0;   // min(C1,C2) <= a*x + b*y + s_k <= inf

        constraintMatrix.insert((N+1)*nx+ 2*i+1, i*nx) = -dy_dtheta;      // a*x
        constraintMatrix.insert((N+1)*nx+ 2*i+1, i*nx+1) = dx_dtheta;     // b*y
        constraintMatrix.insert((N+1)*nx+ 2*i+1, (N+1)*nx +N*nu +i) = -1.0;   // -inf <= a*x + b*y - s_k <= max(C1,C2)

        //get upper line and lower line
        Vector2d left_tangent_p, right_tangent_p, center_p;
        Vector2d right_line_p1, right_line_p2, left_line_p1, left_line_p2;
        geometry_msgs::Point r_p1, r_p2, l_p1, l_p2;

        center_p << track_->x_eval(s_k_1), track_->y_eval(s_k_1);
        right_tangent_p = center_p + track_->getRightHalfWidth(s_k_1) * Vector2d(dy_dtheta, -dx_dtheta).normalized();
        left_tangent_p  = center_p + track_->getLeftHalfWidth(s_k_1) * Vector2d(-dy_dtheta, dx_dtheta).normalized();

        // For visualizing track boundaries
        right_line_p1 = right_tangent_p + 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized(); // expand around the point for 0.3m
        right_line_p2 = right_tangent_p - 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
        left_line_p1 = left_tangent_p + 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();
        left_line_p2 = left_tangent_p - 0.15*Vector2d(dx_dtheta, dy_dtheta).normalized();

        r_p1.x = right_line_p1(0);  r_p1.y = right_line_p1(1);
        r_p2.x = right_line_p2(0);  r_p2.y = right_line_p2(1);
        l_p1.x = left_line_p1(0);   l_p1.y = left_line_p1(1);
        l_p2.x = left_line_p2(0);   l_p2.y = left_line_p2(1);
        border_lines_.push_back(r_p1);  border_lines_.push_back(r_p2);
        border_lines_.push_back(l_p1); border_lines_.push_back(l_p2);
        // END of visualiing track boundaries

        // BOUNDARY interception for right tangent line, and left tangent line
        double C1 =  - dy_dtheta*right_tangent_p(0) + dx_dtheta*right_tangent_p(1);
        double C2 = - dy_dtheta*left_tangent_p(0) + dx_dtheta*left_tangent_p(1);

        lower((N+1)*nx+ 2*i) =  min(C1, C2);
        upper((N+1)*nx+ 2*i) = OsqpEigen::INFTY;

        lower((N+1)*nx+ 2*i+1) = -OsqpEigen::INFTY;
        upper((N+1)*nx+ 2*i+1) = max(C1, C2);

        // u_min < u < u_max
        if (i < N) {
            for (int row = 0; row < nu; row++) {
                constraintMatrix.insert((N+1)*nx+ 2*(N+1) +i*nu+row, (N+1)*nx+i*nu+row) = 1.0;
            }
            // input bounds: speed and steer
            lower.segment<nu>((N+1)*nx+ 2*(N+1) +i*nu) <<  -DECELERATION_MAX, -STEER_MAX;
            upper.segment<nu>((N+1)*nx+ 2*(N+1) +i*nu) << ACCELERATION_MAX, STEER_MAX;
        }

        //max velocity
        constraintMatrix.insert((N+1)*nx+ 2*(N+1) + N*nu +i, i*nx+3) = 1;
        lower((N+1)*nx+ 2*(N+1) + N*nu +i) = 0;
        upper((N+1)*nx+ 2*(N+1) + N*nu +i) = SPEED_MAX;

        // slack variables_k >= 0
        constraintMatrix.insert((N+1)*nx + 2*(N+1) + N*nu + (N+1) + i, (N+1)*nx+N*nu +i) = 1.0;
        lower((N+1)*nx + 2*(N+1) + N*nu  + (N+1) + i) = 0;
        upper((N+1)*nx + 2*(N+1) + N*nu  + (N+1) + i) = OsqpEigen::INFTY;

    }
    int numOfConstraintsSoFar = (N+1)*nx + 2*(N+1) + N*nu + (N+1) + (N+1);

    // labmda's >= 0
    for (int i = 0; i < SAFETY_SET_ITERS*K_NEAR; i++) {
        constraintMatrix.insert(numOfConstraintsSoFar + i, (N+1)*nx+ N*nu + (N+1) + i) = 1.0;
        lower(numOfConstraintsSoFar + i) = 0;
        upper(numOfConstraintsSoFar + i) = OsqpEigen::INFTY;
    }
    numOfConstraintsSoFar += SAFETY_SET_ITERS*K_NEAR;

    // terminal state constraints, relaxed:  -s_terminal <= -x_N+1 + linear_combination(lambda's) <= s_terminal
    // 0 <= s_terminal -x_N+1 + linear_combination(lambda's) <= inf
    for (int i = 0; i < SAFETY_SET_ITERS*K_NEAR; i++) {
        for (int state_ind = 0; state_ind < nx; state_ind++) {
            constraintMatrix.insert(numOfConstraintsSoFar + state_ind, (N+1)*nx+ N*nu + (N+1) + i) = terminal_CSS[i].x(state_ind);
        }
    }
    for (int state_ind=0; state_ind<nx; state_ind++) {
        constraintMatrix.insert(numOfConstraintsSoFar + state_ind, N*nx + state_ind) = -1;
        constraintMatrix.insert(numOfConstraintsSoFar+state_ind, (N+1)*nx+ N*nu + (N+1) + SAFETY_SET_ITERS*K_NEAR + state_ind) = 1;
        lower(numOfConstraintsSoFar+state_ind) = 0;
        upper(numOfConstraintsSoFar+state_ind) = OsqpEigen::INFTY;
    }
    numOfConstraintsSoFar += nx;

    //-inf <= -x_N+1 + linear_combination(lambda's) - s_terminal <= 0
    for (int i = 0; i < SAFETY_SET_ITERS*K_NEAR; i++) {
        for (int state_ind = 0; state_ind < nx; state_ind++) {
            constraintMatrix.insert(numOfConstraintsSoFar + state_ind, (N+1)*nx+ N*nu + (N+1) + i) = terminal_CSS[i].x(state_ind);
        }
    }
    for (int state_ind = 0; state_ind < nx; state_ind++) {
        constraintMatrix.insert(numOfConstraintsSoFar + state_ind, N*nx + state_ind) = -1;
        constraintMatrix.insert(numOfConstraintsSoFar+state_ind, (N+1)*nx+ N*nu + (N+1) + SAFETY_SET_ITERS*K_NEAR + state_ind) = -1;
        lower(numOfConstraintsSoFar+state_ind) = -OsqpEigen::INFTY;
        upper(numOfConstraintsSoFar+state_ind) = 0;
    }

    numOfConstraintsSoFar += nx;
   // cout<<"con dim: "<< (N+1)*nx+ 2*(N+1)*nx + N*nu + (N-1) + (N+1)*nx + (SAFETY_SET_ITERS*K_NEAR) + nx+1 <<endl;
    // sum of lamda's = 1;
    for (int i=0; i<SAFETY_SET_ITERS*K_NEAR; i++){
        constraintMatrix.insert(numOfConstraintsSoFar, (N+1)*nx+ N*nu + (N+1) + i) = 1;
    }

    lower(numOfConstraintsSoFar) = 1.0;
    upper(numOfConstraintsSoFar) = 1.0;
    numOfConstraintsSoFar++;
    if (numOfConstraintsSoFar != (N+1)*nx+ 2*(N+1) + N*nu + (N+1) + (N+1) + (SAFETY_SET_ITERS*K_NEAR) + 2*nx+1) throw;  // for debug

    // cost J
    for (int i = 0; i < SAFETY_SET_ITERS*K_NEAR; i++) {
        gradient((N+1)*nx+ N*nu + (N+1) + i) = terminal_CSS[i].cost;
    }

    //
    for (int i = 0; i < nx; i++) {
        HessianMatrix.insert((N+1)*nx+ N*nu + (N+1) + SAFETY_SET_ITERS*K_NEAR + i, (N+1)*nx+ N*nu + (N+1) + SAFETY_SET_ITERS*K_NEAR + i) = q_s;
    }

    //x0 constraint
    lower.head(nx) = -x0;
    upper.head(nx) = -x0;


    SparseMatrix<double> H_t = HessianMatrix.transpose();
    SparseMatrix<double> sparse_I(number_of_decision_variables,  number_of_decision_variables);
    sparse_I.setIdentity();
    HessianMatrix = 0.5*(HessianMatrix + H_t) + 0.0000001*sparse_I;

    OsqpEigen::Solver solver;
    solver.settings()->setWarmStart(true);
    solver.data()->setNumberOfVariables(number_of_decision_variables);
    solver.data()->setNumberOfConstraints(number_of_constraints);

    if (!solver.data()->setHessianMatrix(HessianMatrix)) throw "fail set Hessian";
    if (!solver.data()->setGradient(gradient)){throw "fail to set gradient";}
    if (!solver.data()->setLinearConstraintsMatrix(constraintMatrix)) throw"fail to set constraint matrix";
    if (!solver.data()->setLowerBound(lower)){throw "fail to set lower bound";}
    if (!solver.data()->setUpperBound(upper)){throw "fail to set upper bound";}

    if(!solver.initSolver()) {
        cout<< "fail to initialize solver"<<endl;
    }

    if(!solver.solve()) {
        return;
    }
    QPSolution_ = solver.getSolution();
    visualize_mpc_solution(terminal_CSS, terminal_candidate);

    solver.clearSolver();

    if (use_dyn_) {
        ROS_INFO("using dynamics");
    } else {
        ROS_INFO("using kinematics");
    }
}

void LMPC::applyControl() {
    float accel = QPSolution_((N+1)*nx);
    float steer = QPSolution_((N+1)*nx+1);
    cout<<"accel_cmd: "<<accel<<endl;
    cout<<"steer_cmd: "<<steer<<endl;
    cout << "slip_angle: "<<slip_angle_<<endl;


    steer = min(steer, 0.41f);
    steer = max(steer, -0.41f);

    ackermann_msgs::AckermannDriveStamped ack_msg;
    ack_msg.drive.acceleration = accel;
    ack_msg.drive.steering_angle = steer;
    ack_msg.drive.steering_angle_velocity = 1.0;
    drive_pub_.publish(ack_msg);
}

void LMPC::visualize_mpc_solution(const vector<Sample>& convex_safe_set, const Matrix<double,nx,1>& terminal_candidate) {
    visualization_msgs::MarkerArray markers;

    visualization_msgs::Marker pred_dots;
    pred_dots.header.stamp = ros::Time::now();
    pred_dots.header.frame_id = "map";
    pred_dots.id = rviz_id::PREDICTION;
    pred_dots.ns = "predicted_positions";
    pred_dots.type = visualization_msgs::Marker::POINTS;
    pred_dots.scale.x = pred_dots.scale.y = pred_dots.scale.z = 0.05;
    pred_dots.action = visualization_msgs::Marker::ADD;
    pred_dots.pose.orientation.w = 1.0;
    pred_dots.color.g = 1.0;
    pred_dots.color.a = 1.0;
    for (int i=0; i<N+1; i++){
        geometry_msgs::Point p;
        p.x = QPSolution_(i*nx);
        p.y = QPSolution_(i*nx+1);
        pred_dots.points.push_back(p);
    }
    markers.markers.push_back(pred_dots);

    visualization_msgs::Marker borderlines;
    borderlines.header.stamp = ros::Time::now();
    borderlines.header.frame_id = "map";
    borderlines.id = rviz_id::BORDERLINES;
    borderlines.ns = "borderlines";
    borderlines.type = visualization_msgs::Marker::LINE_LIST;
    borderlines.scale.x = 0.03;
    borderlines.action = visualization_msgs::Marker::ADD;
    borderlines.pose.orientation.w = 1.0;
    borderlines.color.r = 1.0;
    borderlines.color.a = 1.0;

    borderlines.points = border_lines_;
    markers.markers.push_back(borderlines);

    visualization_msgs::Marker css_dots;
    css_dots.header.stamp = ros::Time::now();
    css_dots.header.frame_id = "map";
    css_dots.id = rviz_id::SAFE_SET;
    css_dots.ns = "safe_set";
    css_dots.type = visualization_msgs::Marker::POINTS;
    css_dots.scale.x = css_dots.scale.y = css_dots.scale.z = 0.04;
    css_dots.action = visualization_msgs::Marker::ADD;
    css_dots.pose.orientation.w = 1.0;
    css_dots.color.g = 1.0;
    css_dots.color.b = 1.0;
    css_dots.color.a = 1.0;
    VectorXd costs = VectorXd(convex_safe_set.size());
    for (int i = 0; i < convex_safe_set.size(); i++) {
        geometry_msgs::Point p;
        p.x = convex_safe_set[i].x(0);
        p.y = convex_safe_set[i].x(1);
        css_dots.points.push_back(p);
        costs(i) = convex_safe_set[i].cost;
    }
    //cout<<"costs: "<<costs<< endl;
    markers.markers.push_back(css_dots);

    visualization_msgs::Marker terminal_dot;
    terminal_dot.header.stamp = ros::Time::now();
    terminal_dot.header.frame_id = "map";
    terminal_dot.id = rviz_id::TERMINAL_CANDIDATE;
    terminal_dot.ns = "terminal_candidate";
    terminal_dot.type = visualization_msgs::Marker::SPHERE;
    terminal_dot.scale.x = terminal_dot.scale.y = terminal_dot.scale.z = 0.1;
    terminal_dot.action = visualization_msgs::Marker::ADD;
    terminal_dot.pose.orientation.w = 1.0;
    terminal_dot.pose.position.x = terminal_candidate(0);
    terminal_dot.pose.position.y = terminal_candidate(1);
    terminal_dot.color.r = 0.5;
    terminal_dot.color.b = 0.8;
    terminal_dot.color.a = 1.0;
    markers.markers.push_back(terminal_dot);

    LMPC_viz_pub_.publish(markers);
}

int main(int argc, char **argv) {
    ros::init(argc, argv, "LMPC");
    ros::NodeHandle nh;
    LMPC lmpc(nh);
    ros::Rate rate(20);
    while(ros::ok()) {
        ros::spinOnce();
        lmpc.run();
        rate.sleep();
    }
    return 0;
}