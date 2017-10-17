#include <iostream>
#include <fstream>

#include <ompl/control/SpaceInformation.h>
#include <ompl/base/goals/GoalState.h>

#include <ompl/control/planners/kpiece/KPIECE1.h>
#include <ompl/control/planners/rrt/RRT.h>

#include <ompl/base/spaces/SO2StateSpace.h>
#include <ompl/base/spaces/RealVectorStateSpace.h>
#include <ompl/base/StateSpace.h>

#include <ompl/control/spaces/RealVectorControlSpace.h>

#include <ompl/control/PathControl.h>

#include <ompl/control/SimpleSetup.h>

#include <math.h>

#include <Eigen/Core>

using namespace std;
using namespace Eigen;


//Quick n Dirty: system definitions
double m = 2.0;
double l = 3.0;
double g = 9.81;
double dtA = 0.00025;

inline double so2Err(const double alpha) {
    //Keep error value in ]-pi, pi]
    double out = fmod(alpha, 2.*M_PI);
    if (out>M_PI) {
        out -= 2.+M_PI;
    }
    return out;
}
inline void so2Err(double& alpha) {
    //Keep error value in ]-pi, pi]
    alpha = fmod(alpha, 2.*M_PI);
    if (alpha>M_PI) {
        alpha -= 2.+M_PI;
    }
    return;
}


class MyProjection : public ompl::base::ProjectionEvaluator
{
public:
MyProjection(const ompl::base::StateSpacePtr &space) : ompl::base::ProjectionEvaluator(space){
}
virtual unsigned int getDimension(void) const{
    return 2;
}
virtual void defaultCellSizes(void){
    cellSizes_.resize(2);
    cellSizes_[0] = 2.*M_PI/50;
    cellSizes_[1] = 0.2;
}
virtual void project(const ompl::base::State *st, ompl::base::EuclideanProjection &projection) const{
    const ompl::base::CompoundState *ainvPendState = st->as<ompl::base::CompoundState>();
    const double phi = ainvPendState->as<ompl::base::SO2StateSpace::StateType>(0)->value;
    const double phiDot = ainvPendState->as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0];
    projection(0) = so2Err(phi);
    projection(1) = phiDot;
}
};


// Get a userdefined goal function
class MyGoal : public ompl::base::Goal {
public:
    double eps;
    Matrix2d P;
    Vector2d goal;
public:
    MyGoal(const ompl::base::SpaceInformationPtr &si) : ompl::base::Goal(si) {
        eps = 0.2;
        P << 2.91, 1.44, 1.44, 1.05;
        goal << 0.,0.;
    }

    inline double getDistance( double phi, double phiDot)const {
        Vector2d pos;
        pos << phi, phiDot;
        pos -= goal;
        pos[0] = so2Err((double)pos[0]);
        return (double) (pos.transpose()*P*pos);
    }

    virtual bool isSatisfied(const ompl::base::State *st) const {
        //Check if system is upright with small velocity
        const ompl::base::CompoundState *ainvPendState = st->as<ompl::base::CompoundState>();
        const double phi = ainvPendState->as<ompl::base::SO2StateSpace::StateType>(0)->value;
        const double phiDot = ainvPendState->as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0];

        double dist = getDistance(phi, phiDot);

        return (bool) (dist < eps);
    }
    virtual bool isSatisfied(const ompl::base::State *st, double *distance) const {
        const ompl::base::CompoundState *ainvPendState = st->as<ompl::base::CompoundState>();
        const double phi = ainvPendState->as<ompl::base::SO2StateSpace::StateType>(0)->value;
        const double phiDot = ainvPendState->as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0];

        *distance = getDistance(phi, phiDot);

        bool result = (bool) (*distance < eps);

        return result;
    }
};

bool isStateValid(const ompl::control::SpaceInformation *si, const ompl::base::State *state) {
    //TBD: Actually do sth here
    return si->satisfiesBounds(state);
}
/*
void propagate2(const ompl::base::State *start, const ompl::control::Control *control, const double duration, ompl::base::State *result){
    const ompl::base::CompoundStateSpace::StateType *ainvPendState = start->as<ompl::base::CompoundStateSpace::StateType>();

    const double phi = ainvPendState->getSubspace("phi")->as<ompl::base::SO2StateSpace::StateType>()->value;
    const double phiDot = (*ainvPendState).getSubspace("phiDot")->as<ompl::base::RealVectorStateSpace::StateType>()->values[0];
    const double ctrl = control->as<ompl::control::RealVectorControlSpace::ControlType>()->values[0];

    double phiT = phi;
    double phiDotT = phiDot;

    int N = duration/dtA;
    double dt = duration/N;

    for (int i=0; i<N; ++i){
        phiT += dt*phiDot;
        phiDotT += (sin(phiT)*(g/l) + 1./(m*l*l)*ctrl);
    }

    result->as<ompl::base::CompoundStateSpace>()->getSubspace("phi")->as<ompl::base::SO2StateSpace::StateType>()->value = phiT;
    result->as<ompl::base::CompoundStateSpace>()->getSubspace("phiDot")->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = phiDotT;
}
*/

void propagate(const ompl::base::State *start, const ompl::control::Control *control, const double duration, ompl::base::State *result) {
    const ompl::base::CompoundState *ainvPendState = start->as<ompl::base::CompoundState>();

    const double phi = ainvPendState->as<ompl::base::SO2StateSpace::StateType>(0)->value;
    const double phiDot = ainvPendState->as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0];
    const double ctrl = control->as<ompl::control::RealVectorControlSpace::ControlType>()->values[0];

    double phiT = phi;
    double phiDotT = phiDot;

    int N = duration/dtA;
    double dt = duration/N;

    for (int i=0; i<N; ++i) {
        phiT += dt*phiDotT;
        phiDotT += dt*(sin(phiT)*(g/l) + 1./(m*l*l)*ctrl);
    }

    result->as<ompl::base::CompoundState>()->as<ompl::base::SO2StateSpace::StateType>(0)->value = phiT;
    result->as<ompl::base::CompoundState>()->as<ompl::base::RealVectorStateSpace::StateType>(1)->values[0] = phiDotT;
}

int main() {
    cout << "Performing planning for inverted pendulum" << endl;

    //get the state-space: SO2+R^1
    ompl::base::StateSpacePtr phi(new ompl::base::SO2StateSpace());
    ompl::base::StateSpacePtr phiDot(new ompl::base::RealVectorStateSpace(1));
    ompl::base::RealVectorBounds bounds(1);
    bounds.setLow(-50.);
    bounds.setHigh(50.);
    phiDot->as<ompl::base::RealVectorStateSpace>()->setBounds(bounds);
    phi->setName("phi");
    phiDot->setName("phiDot");
    //phi->setName("phi");
    //phiDot->setName("phiDot");
    ompl::base::StateSpacePtr invPendStateSpace = phi+phiDot;
    invPendStateSpace->registerProjection("myProjection", ompl::base::ProjectionEvaluatorPtr(new MyProjection(invPendStateSpace)));

    //Get the control space -> real vector representing tau
    ompl::control::ControlSpacePtr cspace(new ompl::control::RealVectorControlSpace(invPendStateSpace, 1));
    //Set bounds
    ompl::base::RealVectorBounds cbounds(1);
    cbounds.setLow(-30.);
    cbounds.setHigh(30.);

    cspace->as<ompl::control::RealVectorControlSpace>()->setBounds(cbounds);

    // construct an instance of  space information from this control space
    ompl::control::SpaceInformationPtr si(new ompl::control::SpaceInformation(invPendStateSpace, cspace));
    // set state validity checking for this space
    si->setStateValidityChecker(std::bind(&isStateValid, si.get(),  std::placeholders::_1));

    si->setPropagationStepSize(0.02);
    si->setMinMaxControlDuration(1,20);

    // set the state propagation routine
    si->setStatePropagator(std::bind(&propagate, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
    //
    si->setup();

    //Get start and goal
    ompl::base::ScopedState<ompl::base::CompoundStateSpace> start(invPendStateSpace);
    //start->as<ompl::base::CompoundStateSpace>()->getSubspace("phi")->as<ompl::base::SO2StateSpace::StateType>()->value = M_PI;//This corresponds to stable eq
    //start->getSpace()->getSubspace("phi")->as<ompl::base::SO2StateSpace::StateType>()->value = M_PI;//This corresponds to stable eq
    //start->as<ompl::base::CompoundStateSpace>()->getSubspace("phiDot")->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = 0.0;
    start[0]=M_PI-0.001;
    start[1]=0.;

    // create a problem instance ob::ProblemDefinitionPtr pdef(new ob::ProblemDefinition(si));
    ompl::base::ProblemDefinitionPtr pdef(new ompl::base::ProblemDefinition(si));
    // set the start and goal states
        //Get the goal
    ompl::base::GoalPtr thisGoal(new MyGoal(si));

    pdef->addStartState(start);
    pdef->setGoal(thisGoal);

    //simple
    //ompl::control::SimpleSetup ss(cspace);

    // set the state propagation routine
    //ss.setStatePropagator(std::bind(&propagate, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));

    // set state validity checking for this space
    //ss.setStateValidityChecker(std::bind(&isStateValid, ss.getSpaceInformation().get(), std::placeholders::_1));

    //Solverob::PlannerPtr planner(new oc::KPIECE1(si));
    si->setup();
    ompl::base::PlannerPtr planner(new ompl::control::RRT(si));

    // set the problem we are trying to solve for the planner
    planner->setProblemDefinition(pdef);
    planner->as<ompl::control::KPIECE1>()->setProjectionEvaluator("myProjection");

    // perform setup steps for the planner
    planner->setup();

    // print the settings for this space
    si->printSettings(std::cout);

    // print the problem settings
    pdef->print(std::cout);

    // attempt to solve the problem within one second of planning time
    planner->setup();
    ompl::base::PlannerStatus solved = planner->solve(1000.);

    if (solved) {
        // get the goal representation from the problem definition (not the same as the goal state)
        // and inquire about the found path
        ompl::base::PathPtr path = pdef->getSolutionPath();
        std::cout << "Found solution:" << std::endl;

        // print the path to screen
        path->print(std::cout);
        path->as<ompl::control::PathControl>()->printAsMatrix(std::cout);
        ofstream resFile;
        resFile.open("invPendSwingUp.txt");
        path->as<ompl::control::PathControl>()->printAsMatrix(resFile);
        resFile.close();
        //path->printAsMatrix(std::cout);
    } else {
        std::cout << "No solution found" << std::endl;
    }

    return 0;
}
