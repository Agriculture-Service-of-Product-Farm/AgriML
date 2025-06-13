#include "AgriML.h"
#include <iostream>
#include <Eigen/Dense>
#include <Eigen/SVD>

AgriML::AgriML() : sensorData(10, 0.0)
{
    std::cout << "AgriML constructor called" << std::endl;
}

AgriML::~AgriML()
{
    std::cout << "AgriML destructor called" << std::endl;
}

void AgriML::run()
{
    std::cout << "AgriML run method called" << std::endl;
}

void AgriML::addSensorData(double data)
{
    sensorData.push_back(data);
    std::cout << "Added sensor data: " << data << std::endl;
}

#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

Eigen::MatrixXd AgriML::performPCA(const Eigen::MatrixXd &data, int numComponents)
{
    Eigen::MatrixXd centeredData = data.rowwise() - data.colwise().mean();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(centeredData, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd principalComponents = svd.matrixV().leftCols(numComponents);
    return principalComponents;
}

void AgriML::performAutoencoder(const Eigen::MatrixXd &data)
{
    // TODO: Implement Autoencoder using TensorFlow C++ API
    std::cout << "Autoencoder implementation not yet complete." << std::endl;
}
