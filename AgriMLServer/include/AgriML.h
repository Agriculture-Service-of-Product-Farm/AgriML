#ifndef AGRIML_H
#define AGRIML_H

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <xgboost/c_api.h>
#include <tensorflow/core/platform/env.h>
#include <tensorflow/core/public/session.h>

class AgriML
{
public:
    AgriML();
    void run();
    void addSensorData(double data);
    Eigen::MatrixXd performPCA(const Eigen::MatrixXd &data, int numComponents);
    void performAutoencoder(const Eigen::MatrixXd &data);

private:
    std::vector<double> sensorData;
};

#endif
