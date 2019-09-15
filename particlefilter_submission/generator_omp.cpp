#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <sstream>
#include <math.h>
#include <time.h>
#include "helper_functions.h"
#include <omp.h>



int num_particles = 25000;
	//Set up parameters here
double delta_t = 0.1; // Time elapsed between measurements [sec]
double sensor_range = 50; // Sensor range [m]


double sigma_pos [3] = {0.3, 0.3, 0.01}; // GPS measurement uncertainty [x [m], y [m], theta [rad]]
double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]


std::default_random_engine gen;

using namespace std;


struct Particle {

	int id;
	double x;
	double y;
	double theta;
	double weight;
};



bool generate_gt(std::vector<control_s> position_meas, std::vector<ground_truth>& grnd_trth, ground_truth init){

	int n = position_meas.size();
	ground_truth grt;
	grt.x = init.x;
	grt.y = init.y;
	grt.theta = init.theta;
	

	for (int i = 0; i<n; i++){
		grnd_trth.push_back(grt);

		std::normal_distribution<double> dist_x(0, sigma_pos[0]);
		std::normal_distribution<double> dist_y(0, sigma_pos[1]);
		std::normal_distribution<double> dist_theta(0, sigma_pos[2]);
		double yaw_rate = position_meas[i].yawrate;
		double velocity = position_meas[i].velocity;




		if (fabs(yaw_rate) > 0.001) {
			grt.x += velocity/yaw_rate * (sin(grt.theta + yaw_rate * delta_t) - sin(grt.theta));
			grt.y += velocity/yaw_rate * (cos(grt.theta)  - cos(grt.theta + yaw_rate * delta_t));
			
		} 
		else {
			grt.x += velocity * delta_t * cos(grt.theta);
			grt.y+= velocity * delta_t * sin(grt.theta);
		}
		grt.theta  += yaw_rate * delta_t;

		grt.x =  grt.x ;//+ dist_x(gen);
		grt.y =  grt.y ;//+ dist_y(gen);
		grt.theta = grt.theta ;//+ dist_theta(gen);
		
	}
	grnd_trth.push_back(grt);
	
	

	return true;
}

bool comparator(ground_truth pos,LandmarkObs land ){
	if(((pos.x-land.x)*(pos.x-land.x)+(pos.y-land.y)*(pos.y-land.y))<(sensor_range*sensor_range)){
		return true;
	}
	else return false;
}

bool generate_obs(ground_truth pos,std::vector<LandmarkObs>& observ, Map map){
	int n = map.landmark_list.size();

	LandmarkObs Lmark;


	for (int i = 0; i<n; i++){

		//Lmark.id=map.landmark_list[i].id_i;
		Lmark.x= map.landmark_list[i].x_f ;
		Lmark.y= map.landmark_list[i].y_f;
		if(comparator(pos,Lmark)){
			Lmark.x= (map.landmark_list[i].x_f - pos.x)*cos(pos.theta) + (map.landmark_list[i].y_f - pos.y)*sin(pos.theta);
			Lmark.y= (pos.x - map.landmark_list[i].x_f)*sin(pos.theta) + (map.landmark_list[i].y_f - pos.y)*cos(pos.theta);
			observ.push_back(Lmark);
		}
	}
	return true;
}












void init(double x, double y, double theta, double std[], std::vector<Particle>& particles, std::vector<double>& weights) {
	//   Sets the number of particles. Initializes all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1 (see main)
	// 	 Adds random Gaussian noise to each particle.
	
	
	weights.resize(num_particles, 1.0);

	// This line creates a normal (Gaussian) distribution for x
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);

	for (int i = 0; i < num_particles; ++i) {
		
		double sample_x, sample_y, sample_theta;
		
		sample_x = dist_x(gen);
		sample_y = dist_y(gen);
		sample_theta = dist_theta(gen);

		Particle particle;
		particle.id = i;		
		particle.x = sample_x;
		particle.y = sample_y;
		particle.theta = sample_theta;
		particle.weight = 1.0;

		particles.push_back(particle);

	}

	

	return;

}

void prediction(double delta_t, double std_pos[], double velocity, double yaw_rate, std::vector<Particle>& particles) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	omp_set_num_threads(8);

	int i =0;

	#pragma omp parallel for default(none) shared(delta_t, std_pos, velocity, yaw_rate,particles,gen)

	for (i = 0; i<particles.size(); i++){
		Particle p = particles[i];

		if (fabs(yaw_rate) > 0.001) {
			p.x += velocity/yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
			p.y += velocity/yaw_rate * (cos(p.theta)  - cos(p.theta + yaw_rate * delta_t));
			
		} 
		else {
			p.x += velocity * delta_t * cos(p.theta);
			p.y += velocity * delta_t * sin(p.theta);
		}
		p.theta += yaw_rate*delta_t;

		std::normal_distribution<double> dist_x(p.x, std_pos[0]);
		std::normal_distribution<double> dist_y(p.y, std_pos[1]);
		std::normal_distribution<double> dist_theta(p.theta, std_pos[2]);

		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);

		particles[i] = p;
		
	}

	return;
}

std::vector<LandmarkObs> associate_data(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	// observed measurement to this particular landmark.
	
	std::vector<LandmarkObs> associated_landmarks;
	LandmarkObs closest;
	
	for (auto obs: observations){
		
		double shortest = 1E10; // some number larger than any possible measurement 

		for (auto pred: predicted){
			double distance = dist(obs.x,obs.y,pred.x,pred.y);
			if (distance < shortest) {
				shortest = distance;
				closest = pred;
			}
		}

		associated_landmarks.push_back(closest);
	}
	
	return associated_landmarks;
}

void updateWeights(double sensor_range, double std_landmark[], std::vector<LandmarkObs> observations, Map map_landmarks, std::vector<Particle>& particles,std::vector<double>& weights) {
	//  Updates the weights of each particle using a mult-variate Gaussian distribution. 

	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	omp_set_num_threads(8);

	int i =0;

	#pragma omp parallel for default(none) shared(sensor_range, sigma_x, sigma_y, observations, map_landmarks, particles, weights)

	for(int i=0; i < particles.size(); ++i) {

		// collect all landmarks within sensor range of the current particle in a vector predicted.
	    Particle p = particles[i];

	    // transform observations from the particle coordinate system to the MAP system
		std::vector<LandmarkObs> transformed_observations;
		for (auto observation: observations){

			LandmarkObs transformed_observation;
			transformed_observation.x = p.x + observation.x * cos(p.theta) - observation.y * sin(p.theta);
			transformed_observation.y = p.y + observation.x * sin(p.theta) + observation.y * cos(p.theta);
			//transformed_observation.id = observation.id;

			transformed_observations.push_back(transformed_observation);

		}

		// get all landmarks that are within sight of the particle
		std::vector<LandmarkObs> predicted;
		for (auto landmark: map_landmarks.landmark_list){

			double distance = dist(p.x,p.y,landmark.x_f,landmark.y_f);
			if (distance < sensor_range) {
				LandmarkObs one_landmark;
				one_landmark.id = landmark.id_i;
				one_landmark.x = landmark.x_f;
				one_landmark.y = landmark.y_f;
				predicted.push_back(one_landmark);		
			}
		}

		// then associate the nearest landmark to every observation of the particle 
		std::vector<LandmarkObs> associated_landmarks;
		associated_landmarks = associate_data(predicted, transformed_observations);

		double probability = 1;		
		for (int j=0; j < associated_landmarks.size(); ++j){

			double dx = transformed_observations.at(j).x - associated_landmarks.at(j).x;
			double dy = transformed_observations.at(j).y - associated_landmarks.at(j).y;
			probability *= 1.0/(2*M_PI*sigma_x*sigma_y) * exp(-dx*dx / (2*sigma_x*sigma_x))* exp(-dy*dy / (2*sigma_y*sigma_y));			
		}

		p.weight = probability;
		weights[i] = probability;

	}

	return;
}


void resample(std::vector<Particle>& particles, std::vector<double>& weights) {

	std::discrete_distribution<int> d(weights.begin(), weights.end());
	std::vector<Particle> weighted_sample(num_particles);

	for(int i = 0; i < num_particles; ++i){
		int j = d(gen);
		weighted_sample.at(i) = particles.at(j);
	}

	particles = weighted_sample;

	return;

}









int main() {

	bool is_initialized = false;

	// Start timer.
	int start = clock();
	double starttime = omp_get_wtime();
	


	// noise generation

	normal_distribution<double> N_x_init(0, sigma_pos[0]);
	normal_distribution<double> N_y_init(0, sigma_pos[1]);
	normal_distribution<double> N_theta_init(0, sigma_pos[2]);
	normal_distribution<double> N_obs_x(0, sigma_landmark[0]);
	normal_distribution<double> N_obs_y(0, sigma_landmark[1]);
	double n_x, n_y, n_theta, n_range, n_heading;

	std::vector<double> weights;
	std::vector<Particle> particles;
	// Read map data
	Map map;
	if (!read_map_data("data/map_data.txt", map)) {
		cout << "Error: Could not open map file" << endl;
		return -1;
	}

	// Read position data
	vector<control_s> position_meas;
	if (!read_control_data("data/control_data.txt", position_meas)) {
		cout << "Error: Could not open position/control measurement file" << endl;
		return -1;
	}
	
	// Read ground truth data
	vector<ground_truth> gt;

	ground_truth initial;
	initial.x = 6.2785;
	initial.y = 1.9598;
	initial.theta = 0.0;
	generate_gt( position_meas, gt, initial);

	
	// Run particle filter!
	int num_time_steps = position_meas.size();

	double total_error[3] = {0,0,0};
	double cum_mean_error[3] = {0,0,0};
	
	for (int i = 0; i < num_time_steps; ++i) {
		cout << "Time step: " << i << endl;
		// Read in landmark observations for current time step.

		ostringstream file;
		file << "data/observation/observations_" << setfill('0') << setw(6) << i+1 << ".txt";
		vector<LandmarkObs> observations;
/*		if (!read_landmark_data(file.str(), observations)) {
			cout << "Error: Could not open observation file " << i+1 << endl;
			return -1;
		}*/
		generate_obs(gt[i],observations,map);
		

		
		// Initialize particle filter if this is the first time step.
		if (!is_initialized) {
			n_x = N_x_init(gen);
			n_y = N_y_init(gen);
			n_theta = N_theta_init(gen);
			init(gt[i].x + n_x, gt[i].y + n_y, gt[i].theta + n_theta, sigma_pos, particles, weights);
			is_initialized = true;
		}
		else {
			// Predict the vehicle's next state (noiseless).

			prediction(delta_t, sigma_pos, position_meas[i-1].velocity, position_meas[i-1].yawrate, particles);
		}
		// simulate the addition of noise to noiseless observation data.
		vector<LandmarkObs> noisy_observations;
		LandmarkObs obs;
		for (int j = 0; j < observations.size(); ++j) {
			n_x = N_obs_x(gen);
			n_y = N_obs_y(gen);
			obs = observations[j];
			obs.x = obs.x + n_x;
			obs.y = obs.y + n_y;
			noisy_observations.push_back(obs);
		}

		// Update the weights and resample
		updateWeights(sensor_range, sigma_landmark, noisy_observations, map, particles, weights);
		resample(particles,weights);
		
		// Calculate and output the average weighted error of the particle filter over all time steps so far.

		int num_particles = particles.size();
		double highest_weight = 0.0;
		Particle best_particle;
		for (int i = 0; i < num_particles; ++i) {
			if (particles[i].weight > highest_weight) {
				highest_weight = particles[i].weight;
				best_particle = particles[i];
			}
		}
		double *avg_error = getError(gt[i].x, gt[i].y, gt[i].theta, best_particle.x, best_particle.y, best_particle.theta);

		// for plotting the results of the filter
		std::ofstream dataFile;
		dataFile.open("best_particle_omp.dat", std::ios::app);
		dataFile << best_particle.x << " " << best_particle.y << " " << best_particle.theta << "\n";
		dataFile.close();
		

		for (int j = 0; j < 3; ++j) {
			total_error[j] += avg_error[j]*avg_error[j];
			cum_mean_error[j] = total_error[j] / (double)(i + 1);
		}
		
		// Print the cumulative weighted error
		cout << "Cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2] << endl;
		


	}
	
	// Output the runtime for the filter.
	int stop = clock();
	double endtime = omp_get_wtime();
	double runtime = (stop - start) / double(CLOCKS_PER_SEC);
cout<<"Mean square error :"<< sqrt(total_error[0])/num_time_steps<<" "<<sqrt(total_error[1])/num_time_steps<<" "<< sqrt(total_error[2])/num_time_steps<<endl;
	cout << "Runtime (sec): " << runtime << endl;
	cout << "omp Runtime (sec): " << -1*(starttime-endtime) << endl;
	

	return 0;
}
