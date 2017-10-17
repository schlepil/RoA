cmake_minimum_required(VERSION 2.8)
project(myOMPL)

#Libraries
#find_package(Eigen REQUIRED)
find_package(OMPL REQUIRED)
FIND_PACKAGE(GSL REQUIRED)
#find_package(Eigen3 REQUIRED)

include_directories( /usr/include/Eigen3 )
include_directories( /usr/include/eigen3 )
#find_package(ODE REQUIRED)

set(CMAKE_CXX_FLAGS "-std=c++11 -O3 ${CMAKE_CXX_FLAGS}")
#set(LIBRARY_OUTPUT_PATH lib/${CMAKE_BUILD_TYPE})
#set(OSG_LIBRARIES  osg osgAnimation osgDB osgFX osgGA osgManipulator osgParticle osgPresentation osgQt osgShadow osgSim osgTerrain osgText osgUtil osgViewer osgVolume osgWidget)

#include_directories(include)

install(DIRECTORY include/  DESTINATION /usr/local/include/${PROJECT_NAME})

add_executable(main main.cpp)
target_link_libraries(main ${OMPL_LIBRARIES})
add_executable(main2 main2.cpp)
target_link_libraries(main2 ${OMPL_LIBRARIES})
add_executable(main3 main3.cpp)
target_link_libraries(main3 ${OMPL_LIBRARIES})
add_executable(main3_ode main3_ode.cpp)
target_link_libraries(main3_ode ${OMPL_LIBRARIES} ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES})
add_executable(main_SO main_SO.cpp)
target_link_libraries(main_SO ${OMPL_LIBRARIES})
add_executable(main_Acro main_Acro.cpp)
target_link_libraries(main_Acro ${OMPL_LIBRARIES} ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES})
