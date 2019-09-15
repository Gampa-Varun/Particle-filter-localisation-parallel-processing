#include <iostream>
#include <iomanip>
#include <random>
#include <fstream>
#include <sstream>
#include <math.h>
#include <time.h>
#include "helper_functions.h"
#include <curand.h>
#include <curand_kernel.h>
#include<cuda.h>

using namespace std;

const int num_particles = 30000;
	//Set up parameters here
double delta_t = 0.1; // Time elapsed between measurements [sec]
double sensor_range = 50; // Sensor range [m]
double sigma_pos [3] = {0.3, 0.3, 0.01}; // Uncertainty [x [m], y [m], theta [rad]]
double sigma_landmark [2] = {0.3, 0.3}; // Landmark measurement uncertainty [x [m], y [m]]

default_random_engine gen;



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

		Lmark.id=map.landmark_list[i].id_i;
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




__device__ void setup_kernel(curandState *state, int off){

  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  curand_init(1234, idx, off, &state[0]);
}





bool is_initialized = false;


__global__ void init(double x, double y, double theta, double* std, Particle* particles, int num_particles, double* weights) {
	//   Sets the number of particles. Initializes all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1 (see main)
	// 	 Adds random Gaussian noise to each particle.

	
	int id = blockIdx.x*blockDim.x + threadIdx.x;

	
	

	
	double sample_x, sample_y, sample_theta;

	

	

	if(id < num_particles){

		curandState *devStates;
		devStates = (curandState*)malloc(sizeof(curandState));
		setup_kernel(devStates,0);
		double distx = curand_normal_double(devStates)*std[0] ;
		setup_kernel(devStates,1);
		double disty = curand_normal_double(devStates)*std[1];
		setup_kernel(devStates,2);
		double disttheta = curand_normal_double(devStates)*std[2] ;
		sample_x = x+distx;
		sample_y = y + disty;
		sample_theta = theta+disttheta;

		
		particles[id].id = id;		
		particles[id].x = sample_x;
		particles[id].y = sample_y;
		particles[id].theta = sample_theta;
		particles[id].weight = 1.0;
		weights[id] = 1.0;

		free(devStates);

	}

	return;

}



__global__ void prediction(double del_t, double* std_pos, double velocity, double yaw_rate, Particle* particles, int num_particles) {


	

	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<num_particles){


		if (fabs(yaw_rate) > 0.001) {
			particles[id].x += velocity/yaw_rate * (sin(particles[id].theta + yaw_rate * del_t) - sin(particles[id].theta));
			particles[id].y += velocity/yaw_rate * (cos(particles[id].theta)  - cos(particles[id].theta + yaw_rate * del_t));
			
		} 
		else {
			particles[id].x += velocity * del_t * cos(particles[id].theta);
			particles[id].y += velocity * del_t * sin(particles[id].theta);
		}
		particles[id].theta  += yaw_rate * del_t;
		
		curandState *devStates;
		devStates = (curandState*)malloc(sizeof(curandState));
		setup_kernel(devStates,0);
		double distx = curand_normal_double(devStates)*std_pos[0] ;
		setup_kernel(devStates,1);
		double disty = curand_normal_double(devStates)*std_pos[1];
		setup_kernel(devStates,2);
		double disttheta = curand_normal_double(devStates)*std_pos[2] ;


		particles[id].x = particles[id].x + distx;
		particles[id].y = particles[id].y + disty;
		particles[id].theta = particles[id].theta+disttheta;

		free(devStates);
		
	}	

}

__device__ double distnc(double x1, double y1, double x2, double y2) {
	return sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}






__device__ void associate_data(LandmarkObs *predicted, LandmarkObs *observations, LandmarkObs *associated_landmarks, int lengthobs, int lengthpred) {

	LandmarkObs closest;
	
	
	for (int i = 0;i<lengthobs;i++){
		LandmarkObs obs = observations[i];
		
		double shortest = 1E10; // some number larger than any possible measurement 

		for (int j = 0; j< lengthpred;j++){
			LandmarkObs pred = predicted[j];
			double distance = distnc(obs.x,obs.y,pred.x,pred.y);
			if (distance < shortest) {
				shortest = distance;
				closest = pred;
			}
		}

		associated_landmarks[i] = closest;

	}
	return;

}






__global__ void updateWeights(double sensor_range, double* std_landmark, LandmarkObs *observations, LandmarkObs* map_landmarks, Particle* particles, double *weights, int num_particles, int lengthobs, int lengthmap) {
	//  Updates the weights of each particle using a mult-variate Gaussian distribution. 


	double sigma_x = std_landmark[0];
	double sigma_y = std_landmark[1];

	int id = blockIdx.x*blockDim.x + threadIdx.x;

	if(id<num_particles){
		// collect all landmarks within sensor range of the current particle in a vector predicted.
	    Particle p = particles[id];
	    

	    // transform observations from the particle coordinate system to the MAP system
	
		LandmarkObs *transformed_observations;
		int sizeT = sizeof(LandmarkObs)*lengthobs;
		transformed_observations = (LandmarkObs*)malloc(sizeT );

		for (int i = 0; i < lengthobs;i++){
	
			transformed_observations[i].x = p.x + observations[i].x * cos(p.theta) - observations[i].y * sin(p.theta);
			transformed_observations[i].y = p.y + observations[i].x * sin(p.theta) + observations[i].y * cos(p.theta);
			transformed_observations[i].id = observations[i].id;

		}

		// get all landmarks that are within sight of the particle
		LandmarkObs *predicted;
		int sizePr = sizeof(LandmarkObs)*lengthmap;

		predicted = (LandmarkObs*)malloc(sizePr);
		int lengthpred = 0;
		for (int k =0; k<lengthmap;k++){


			double distance = distnc(p.x,p.y,map_landmarks[k].x,map_landmarks[k].y);
			if (distance < sensor_range) {
				predicted[lengthpred] = map_landmarks[k];
				lengthpred = lengthpred+1;
			}
		}

		// then associate the nearest landmark to every observation of the particle 
		LandmarkObs *associated_landmarks;
		int sizeA = sizeof(LandmarkObs)*lengthobs;
		associated_landmarks =  (LandmarkObs*)malloc(sizeA) ;
		//LandmarkObs *associated_landmarks = 
		associate_data(predicted, transformed_observations,associated_landmarks, lengthobs, lengthpred);

		double probability = 1.0;		
		for (int j=0; j < lengthobs; ++j){

			double dx = transformed_observations[j].x - associated_landmarks[j].x;
			double dy = transformed_observations[j].y - associated_landmarks[j].y;
			probability *= 1.0/(2*M_PI*sigma_x*sigma_y) * exp(-dx*dx / (2*sigma_x*sigma_x))* exp(-dy*dy / (2*sigma_y*sigma_y));			
		}

		p.weight = probability;
		weights[id] = probability;

		particles[id] = p;

		free(transformed_observations);
		free(predicted);
		free(associated_landmarks);

	}

}




void resample(Particle* particles, double* weights ) {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	// http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution


	std::vector<double> weightvec;
	for(int i =0;i<num_particles;i++){
		weightvec.push_back(weights[i]);
	}


	std::discrete_distribution<int> d(weightvec.begin(), weightvec.end());
	
	Particle weighted_sample[num_particles];

	for(int i = 0; i < num_particles; ++i){
		int j = d(gen);
		weighted_sample[i] = particles[j];
	}

	for (int i = 0; i<num_particles;i++){
		particles[i] = weighted_sample[i];
	}

	return;

}
void my_cudasafe( cudaError_t error, char* message)
{
	if(error!=cudaSuccess) 
	{ 
		fprintf(stderr,"ERROR: %s : %s\n",message,cudaGetErrorString(error)); 
		exit(-1); 
	}
}





int main() {

	bool is_initialized = false;

	// Start timer.
	int start = clock();


	double *weights;
	int sizeW = sizeof(double)*num_particles;
	weights = (double*)malloc(sizeW);

	double *d_weights ;
	cudaMalloc(&d_weights,sizeW);






	Particle  *particles;
	int sizeP = sizeof(Particle)*num_particles;
	particles = (Particle*)malloc( sizeP );

	Particle *d_particles;
	cudaMalloc(&d_particles,sizeP);


	Map map;
	if (!read_map_data("data/map_data.txt", map)) {
		cout << "Error: Could not open map file" << endl;
		return -1;
	}
	int lengthmap = map.landmark_list.size();



	LandmarkObs *M;
	int sizeM = sizeof(LandmarkObs)*map.landmark_list.size();
	M = (LandmarkObs*)malloc(sizeM);
	cudaMallocManaged(&M,sizeM);

	for( int t = 0;t<map.landmark_list.size();t++){
		M[t].id = map.landmark_list[t].id_i;
		M[t].x = map.landmark_list[t].x_f;
		M[t].y = map.landmark_list[t].y_f;

	}


	// Read position data
	vector<control_s> position_meas;
	if (!read_control_data("data/control_data.txt", position_meas)) {
		cout << "Error: Could not open position/control measurement file" << endl;
		return -1;
	}


	double *d_sigma_landmark;
	int sizeL = sizeof(sigma_landmark);
	d_sigma_landmark = (double*)malloc(sizeL);
	cudaMalloc(&d_sigma_landmark,sizeL);
	cudaMemcpy(d_sigma_landmark,sigma_landmark,sizeL,cudaMemcpyHostToDevice);


	double *d_sigma_pos;
	int  sizePos = sizeof(sigma_pos);
	d_sigma_pos = (double*)malloc(sizePos);
	cudaMalloc(&d_sigma_pos,sizeof(sigma_pos));
	cudaMemcpy(d_sigma_pos,sigma_pos,sizePos,cudaMemcpyHostToDevice);


	vector<ground_truth> gt;

	ground_truth initial;
	initial.x = 6.2785;
	initial.y = 1.9598;
	initial.theta = 0.0;
	generate_gt( position_meas, gt, initial);

	int num_time_steps = position_meas.size();



	dim3 DimGrid(100,1);
	dim3 DimBlock(512,1);

	double total_error[3] = {0,0,0};
	double cum_mean_error[3] = {0,0,0};

	cudaMalloc(&d_weights,sizeW);
	cudaMalloc(&d_particles,sizeP);

	for (int i = 0; i < num_time_steps; ++i) {



		cout << "Time step: " << i << endl;
	
		// Read in landmark observations for current time step.
		vector<LandmarkObs> observations;
	

		generate_obs( gt[i], observations, map);

		int sizeO = sizeof(LandmarkObs)*observations.size();
		LandmarkObs *observ;
		observ = (LandmarkObs*)malloc(sizeO);
		for (int l = 0;l<observations.size();l++){
			observ[l].id = observations[l].id;
			observ[l].x = observations[l].x;
			observ[l].y = observations[l].y;

		}
		LandmarkObs *d_observe;
		cudaMalloc(&d_observe,sizeO);
		cudaMemcpy(d_observe,observ,sizeO,cudaMemcpyHostToDevice);
		int lengthobs = observations.size();
		observations.clear();



		if (!is_initialized) {

			init<<<DimGrid,DimBlock >>>(gt[i].x , gt[i].y , gt[i].theta, d_sigma_pos, d_particles, num_particles,d_weights);
			my_cudasafe(cudaGetLastError(), " init");
			
			is_initialized = true;
		}

	else {

			cudaMemcpy(d_particles,particles,sizeP,cudaMemcpyHostToDevice);
			prediction<<< DimGrid,DimBlock >>>(delta_t, d_sigma_pos, position_meas[i-1].velocity, position_meas[i-1].yawrate, d_particles, num_particles);
			my_cudasafe(cudaGetLastError(), " kernel");
		}



		
		updateWeights<<< DimGrid,DimBlock>>>(sensor_range, d_sigma_landmark, d_observe, M, d_particles,d_weights,num_particles,lengthobs, lengthmap);
		my_cudasafe(cudaGetLastError(), " update");

		cudaDeviceSynchronize();

		cudaMemcpy(particles,d_particles,sizeP,cudaMemcpyDeviceToHost);
		cudaMemcpy(weights,d_weights,sizeW,cudaMemcpyDeviceToHost);
		resample(particles, weights);


		double highest_weight = 0.0;
		Particle best_particle;

		for (int j = 0; j < num_particles; ++j) {
			if (particles[j].weight > highest_weight) {
				highest_weight = particles[j].weight;
				best_particle = particles[j];
			}
		}


		std::ofstream dataFile;
		dataFile.open("best_particle_parallel.dat", std::ios::app);
		dataFile << best_particle.x << " " << best_particle.y << " " << best_particle.theta << "\n";
		dataFile.close();
		
		double *avg_error = getError(gt[i].x, gt[i].y, gt[i].theta, best_particle.x, best_particle.y, best_particle.theta);
		

		for (int j = 0; j < 3; ++j) {
			total_error[j] += avg_error[j]*avg_error[j];
			cum_mean_error[j] = total_error[j] / (double)(i + 1);
		}
		
		// Print the cumulative weighted error
		cout << "Cumulative mean weighted error: x " << cum_mean_error[0] << " y " << cum_mean_error[1] << " yaw " << cum_mean_error[2] << endl;
		cout<<"error in x:"<<(best_particle.x-gt[i].x)<<endl;

		cudaFree(d_observe);
		free(observ);


	}

int stop = clock();

	cout<<"Mean square error :"<< sqrt(total_error[0])/num_time_steps<<" "<<sqrt(total_error[1])/num_time_steps<<" "<< sqrt(total_error[2])/num_time_steps<<endl;
	double runtime = (stop - start) / double(CLOCKS_PER_SEC);
	cout << "Runtime (sec): " << runtime << endl;

}






