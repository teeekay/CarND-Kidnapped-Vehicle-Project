/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *  Additional code and errors 
 *  Added: Sep 9, 2017
 *  by: Anthony Knight 
 *
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

#include "debugging.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	//set up normal distributions for x,y, and theta using default_random_engine
	D1(cout << "ParticleFilter::init hello" << endl;)
	random_device rd;
	default_random_engine gen(rd());
	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std[0];
	std_y = std[1];
	std_theta = std[2];

	D1(cout << "ParticleFilter::init (x, y, theta) = (" << x << ", " << y << ", " << theta << ")" << endl;)
	D1(cout << "ParticleFilter::init std_devs " << std_x <<" : "<<  std_y << " : " << std_theta  << endl;)

	//dist_x centred on x with std dev of std_x
	normal_distribution<double> dist_x(x, std_x);  
	//dist_y centred on y with std dev of std_y
	normal_distribution<double> dist_y(y, std_y);
	//dist_theta centred on theta with std dev of std_theta
	normal_distribution<double> dist_theta(theta, std_theta);


	D1(cout << "ParticleFilter::init num_particles " << num_particles << " , " << particles.size()<< "." << endl;)
	for (int i = 0; i < num_particles; i++)
	{
		Particle p = {};
		p.id = i;
		p.x = dist_x(gen);
		p.y = dist_y(gen);
		p.theta = dist_theta(gen);
		p.weight = 1.0;
		particles.push_back(p);
		//also load up weights vector - although I think I'll just clear it 
		weights.push_back(1.0);
	}
	D1(cout << "ParticleFilter::init num_particles " << num_particles << " , " << particles.size() << "." << endl;)
	is_initialized = true;
	D1(printParticles();)
	D1(cout << "ParticleFilter::init bye" << endl;)
	return;
}

void ParticleFilter::printParticles()
{
	for (int i = 0; i < num_particles; i++)
	{
		Particle p = particles[i];
		cout << p.id << ": X: " << p.x << ": Y: " << p.y <<
			": Theta: " << p.theta << ": Weight: " << p.weight << endl;
	}
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	
	D1(cout << "ParticleFilter::prediction Hi" << endl;)
	
	// Add measurements to each particle
	for (int i = 0; i < num_particles; i++)
	{
		if (fabs(yaw_rate) > 0.00001) {
			particles[i].x += velocity / yaw_rate*(sin(particles[i].theta + yaw_rate*delta_t) - sin(particles[i].theta));
			particles[i].y += velocity / yaw_rate*(cos(particles[i].theta) - cos(particles[i].theta + yaw_rate*delta_t));
			particles[i].theta += yaw_rate * delta_t;
		}
		else {
			D1(cout << "yaw_rate of " << yaw_rate << " not used since too close to zero." << endl;)
			particles[i].x += velocity*cos(particles[i].theta)*delta_t;
			particles[i].y += velocity*sin(particles[i].theta)*delta_t;
		}

	}

	// add random Gaussian noise.
	default_random_engine gen;

	double std_x, std_y, std_theta; // Standard deviations for x, y, and theta
	std_x = std_pos[0];
	std_y = std_pos[1];
	std_theta = std_pos[2];

	for (int i = 0; i < num_particles; i++)
	{
		//dist_x centred on x with std dev of std_x
		normal_distribution<double> dist_x(particles[i].x, std_x);
		//dist_y centred on y with std dev of std_y
		normal_distribution<double> dist_y(particles[i].y, std_y);
		//dist_theta centred on theta with std dev of std_theta
		normal_distribution<double> dist_theta(particles[i].theta, std_theta);
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
	}

	D1(printParticles();)

	D1(cout << "ParticleFilter::prediction bye" << endl;)
	return;
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

	for (unsigned int i = 0; i < observations.size(); i++)
	{
		double old_distance = __DBL_MAX__;
		double distance;
		for (unsigned int j = 0; j < predicted.size(); j++)
		{
			distance = dist(predicted[j].x, predicted[j].y, observations[i].x, observations[i].y);
			if (distance < old_distance) {
				observations[i].id = j;  // predicted[j].id;
				old_distance = distance;
			}
		}
		D1(cout << "Observation[" << i << "] at (" << observations[i].x << ", " << observations[i].y << ") closest to landmark[" << observations[i].id << "] at (" <<
			predicted[observations[i].id].x << "," << predicted[observations[i].id].y << "), distance " << old_distance << " m." << endl;)
	}

}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		std::vector<LandmarkObs> observations, Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html
	weights.clear();
	double sig_x = std_landmark[0];
	double sig_y = std_landmark[1];
	D1(cout << "(sig_x,sig_y) = (" << sig_x << ", " << sig_y << ")." << endl;)
	double gauss_norm = 1.0 / (2.0 * M_PI * sig_x * sig_y);
	D1(cout << "gauss_norm = " << gauss_norm << endl;)
	//sum of weights for all particles
	double total_weight = 0.0;
	for (unsigned int i = 0; i < particles.size(); i++)
	{
		D1(cout << "Particle[" << i << "], location ("<< particles[i].x <<", "<< particles[i].y <<"), theta = " << particles[i].theta << "." << endl;)
		//first - translate car co-ordinates of observations to map co-ords
		std::vector<LandmarkObs> map_observations;
		for (unsigned int j = 0; j < observations.size(); j++)
		{

			LandmarkObs map_obs;
			map_obs.x = particles[i].x + (cos(particles[i].theta)*observations[j].x) - (sin(particles[i].theta)*observations[j].y);
			map_obs.y = particles[i].y + (sin(particles[i].theta)*observations[j].x) + (cos(particles[i].theta)*observations[j].y);
			map_obs.id = observations[j].id;
			D1(cout << "Observations from car: (x,y) = (" << observations[j].x <<", " << observations[j].y << "). ";)
			map_observations.push_back(map_obs);
			D1(cout << "Map Observations:      (x,y) = (" << map_observations[j].x << ", " << map_observations[j].y << ")."<<endl;)

		}
		//second find map landmarks within sensor range and store in LandmarkObs vector
		std::vector<LandmarkObs> map_marks;
		for (unsigned int j = 0; j < map_landmarks.landmark_list.size(); j++)
		{

			if (dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[j].x_f, map_landmarks.landmark_list[j].y_f) < sensor_range)
			{
				LandmarkObs map_mark;
				map_mark.x = double(map_landmarks.landmark_list[j].x_f);
				map_mark.y = double(map_landmarks.landmark_list[j].y_f);
				map_mark.id = map_landmarks.landmark_list[j].id_i;
				map_marks.push_back(map_mark);
			}
			//else {
			//	cout << "map landmark " << map_landmarks.landmark_list[j].id_i <<", "<< j << "out of range" << endl;
			//}
		}
		// run data association
		dataAssociation(map_marks, map_observations);

		double cum_prob = 1.0;
		for (unsigned int j = 0; j < map_observations.size(); j++)
		{
			double powerx = pow((map_observations[j].x - map_marks[map_observations[j].id].x), 2.0) / (2 * sig_x*sig_x);
			double powery = pow((map_observations[j].y - map_marks[map_observations[j].id].y), 2.0) / (2 * sig_y*sig_y);
			double prob = gauss_norm * exp(-powerx - powery);
			D1(cout << "particle["<<i<<"] observations["<< j <<"] powerx = " << powerx <<", powery = " << powery << "."<< endl;)

			cum_prob *= prob;
		}
		particles[i].weight = cum_prob;
		total_weight += particles[i].weight;
		D1(cout << "Particles[" << i << "].weight = " << cum_prob << ".  Total_weight = " << total_weight << "." << endl;)

	}
	for (unsigned int i = 0; i < particles.size(); i++)
	{
		particles[i].weight = particles[i].weight / total_weight;
		weights.push_back(particles[i].weight);
	}
	D1(printParticles();)
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
	random_device seed;
	mt19937 random_generator(seed());
	discrete_distribution<> distribution(weights.begin(), weights.end());
	std::vector<Particle> new_particles;
	D1(cout << "ParticleFilter::resample Hi" << endl;)
	for (unsigned int i = 0; i < num_particles; i++)
	{
		new_particles.push_back(particles[distribution(random_generator)]);
		new_particles[i].id = i;
	}
	particles = new_particles;
	D1(printParticles();)
	D1(cout << "ParticleFilter::resample Bye" << endl;)
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
	//particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
	// associations: The landmark id that goes along with each listed association
	// sense_x: the associations x mapping already converted to world coordinates
	// sense_y: the associations y mapping already converted to world coordinates

	//Clear the previous associations
	particle.associations.clear();
	particle.sense_x.clear();
	particle.sense_y.clear();

	particle.associations= associations;
 	particle.sense_x = sense_x;
 	particle.sense_y = sense_y;

 	return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
