cmake_minimum_required(VERSION 3.0)
project(AgriMLServer)

set(CMAKE_CXX_STANDARD 14)

find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

find_package(XGBoost REQUIRED)
include_directories(${XGBoost_INCLUDE_DIR})

find_package(TensorFlow REQUIRED)
include_directories(${TensorFlow_INCLUDE_DIR})

include_directories(${CMAKE_SOURCE_DIR}/include)

link_directories(${XGBoost_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${XGBoost_LIBRARIES})

link_directories(${TensorFlow_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${TensorFlow_LIBRARIES})

# Add database dependencies
find_package(MySQL REQUIRED)
include_directories(${MySQL_INCLUDE_DIR})
link_directories(${MySQL_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${MySQL_LIBRARIES})

find_package(Couchbase REQUIRED)
include_directories(${Couchbase_INCLUDE_DIR})
link_directories(${Couchbase_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${Couchbase_LIBRARIES})

find_package(Redis REQUIRED)
include_directories(${Redis_INCLUDE_DIR})
link_directories(${Redis_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${Redis_LIBRARIES})

find_package(CouchDB REQUIRED)
include_directories(${CouchDB_INCLUDE_DIR})
link_directories(${CouchDB_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${CouchDB_LIBRARIES})

find_package(AWSSDK REQUIRED COMPONENTS s3)
include_directories(${AWSSDK_INCLUDE_DIR})
link_directories(${AWSSDK_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${AWSSDK_LIBRARIES})

find_package(AzureCosmosDB REQUIRED)
include_directories(${AzureCosmosDB_INCLUDE_DIR})
link_directories(${AzureCosmosDB_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${AzureCosmosDB_LIBRARIES})

find_package(InfluxDB REQUIRED)
include_directories(${InfluxDB_INCLUDE_DIR})
link_directories(${InfluxDB_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${InfluxDB_LIBRARIES})

find_package(PostgreSQL REQUIRED)
include_directories(${PostgreSQL_INCLUDE_DIR})
link_directories(${PostgreSQL_LIBRARY_DIR})
target_link_libraries(AgriMLServer ${PostgreSQL_LIBRARIES})

message(STATUS "Eigen include directory: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "XGBoost include directory: ${XGBoost_INCLUDE_DIR}")
message(STATUS "TensorFlow include directory: ${TensorFlow_INCLUDE_DIR}")
message(STATUS "MySQL include directory: ${MySQL_INCLUDE_DIR}")
message(STATUS "Couchbase include directory: ${Couchbase_INCLUDE_DIR}")
message(STATUS "Redis include directory: ${Redis_INCLUDE_DIR}")
message(STATUS "CouchDB include directory: ${CouchDB_INCLUDE_DIR}")
message(STATUS "AWS SDK include directory: ${AWSSDK_INCLUDE_DIR}")
message(STATUS "Azure Cosmos DB include directory: ${AzureCosmosDB_INCLUDE_DIR}")
message(STATUS "InfluxDB include directory: ${InfluxDB_INCLUDE_DIR}")
message(STATUS "PostgreSQL include directory: ${PostgreSQL_INCLUDE_DIR}")

add_executable(AgriMLServer src/main.cpp src/AgriML.cpp)
