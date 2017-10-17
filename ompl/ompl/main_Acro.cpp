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
 #include <Eigen/LU>

#include <stdio.h>
#include <gsl/gsl_errno.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_odeiv.h>

using namespace std;
using namespace Eigen;


//Quick n Dirty: system definitions
//State space is [phi0,phi1, omega0, omega1
//Taken from Drake - RobotLocomotion @ CSAIL
//Geometry
const double l1 = 1.;
const double l2 = 2.;
//Mass
const double m1 = 1.;
const double m2 = 1.;
//Damping
const double b1=.1;
const double b2=.1;
//Gravity center and inertial moment
const double lc1 = .5;
const double lc2 = 1.;
const double Ic1 = .083;
const double Ic2 = .33;
//Gravity
const double g = 9.81;

//helper
const double I1 = Ic1 + m1*lc1*lc1;
const double I2 = Ic2 + m2*lc2*lc2;

double ctrlODE;


inline double getH12(const double phi1){
    //I2 + m2*l1*lc2*sympy.cos(x1);
    return I2 + m2*l1*lc2*cos(phi1);
}

inline Matrix2d getM(const double phi1){
    Matrix2d M;
    //[[ I1 + I2 + m2*l1**2 + 2*m2*l1*lc2*sympy.cos(x1), h12], [h12, I2]]
    M(0,0) = I1 + I2 + m2*l1*l1 + 2*m2*l1*lc2*cos(phi1);
    M(0,1) = getH12(phi1);
    M(1,0) = M(0,1);
    M(1,1) = I2;
    return M;
}
inline Vector2d getC(const double phi1, const double phiDot0, const double phiDot1){
    Vector2d C;
    //    [[ -2*m2*l1*lc2*sympy.sin(x1)*x3, -m2*l1*lc2*sympy.sin(x1)*x3], [m2*l1*lc2*sympy.sin(x1)*x2, 0 ]]
    C(0) = -2.*m2*l1*lc2*sin(phi1)*phiDot1*phiDot0 -m2*l1*lc2*sin(phi1)*phiDot1*phiDot1 + b1*phiDot0;
    C(1) = m2*l1*lc2*sin(phi1)*phiDot0*phiDot0 + b2*phiDot1;
    return C;
}
inline Vector2d getG(const double phi0, const double phi1){
    Vector2d G;
    G(0) = g*( m1*lc1*sin(phi0) + m2*(l1*sin(phi0)+lc2*sin(phi0+phi1)));
    G(1) = g*m2*lc2*sin(phi0+phi1);
    return G;
}

int fDyn(double t, const double X[], double f[], void *params){
    //Input throughput matri
    Vector2d B;
    B << 0.,1.;

    Matrix<double, 1, 1> u;
    u << *(double *)params;

    //The velocity
    f[0] = X[2];
    f[1] = X[3];
    //The acceleration
    Vector2d qd;
    qd << X[2], X[3];
    Vector2d qdd;
    qdd = getM(X[1]).inverse()*( -getC(X[1], X[2], X[3])-getG(X[0], X[1]) + B*u);
    f[2] = qdd(0);
    f[3] = qdd(1);

    return GSL_SUCCESS;
}

int jacDyn(double t, const double X[], double *dfdX, double dfdt[], void *params){
    dfdX = nullptr;
    dfdt = nullptr;

    return GSL_SUCCESS;
 }


//Set up the integrator
const gsl_odeiv_step_type * T = gsl_odeiv_step_rkf45;

gsl_odeiv_step * s = gsl_odeiv_step_alloc (T, 4);
gsl_odeiv_control * c = gsl_odeiv_control_y_new (1e-9, 0.0);
gsl_odeiv_evolve * e = gsl_odeiv_evolve_alloc (4);

gsl_odeiv_system sys = {fDyn, jacDyn, 4, &ctrlODE};
//Define a integration function
void integrateAcro(double X[], double t, double tEnd){
    //Do the integration loop; Result will be in X
    double h = 1e-6;
    while(t<tEnd){
        int status = gsl_odeiv_evolve_apply (e, c, s, &sys, &t, tEnd, &h, X);
        assert ( status == GSL_SUCCESS && "Failed integrating!" );
    }
}


inline double so2Var(const double alpha){
    //Keep error value in ]-pi, pi]
    double out = remainder(alpha, 2.*M_PI);
    return out;
}
inline void so2Var(double& alpha){
    //Keep error value in ]-pi, pi]
    alpha = remainder(alpha, 2.*M_PI);
    return;
}


//Projection taken into account the so2 part
class MyProjection : public ompl::base::ProjectionEvaluator
{
public:
MyProjection(const ompl::base::StateSpacePtr &space) : ompl::base::ProjectionEvaluator(space)
{
}
virtual unsigned int getDimension(void) const{
    return 4;
}
virtual void defaultCellSizes(void){
    cellSizes_.resize(4);
    cellSizes_[0] = 2.*M_PI/50;
    cellSizes_[1] = 2.*M_PI/50;
    cellSizes_[2] = 0.5;
    cellSizes_[3] = 0.5;
}
virtual void project(const ompl::base::State *state, ompl::base::EuclideanProjection &projection) const{
    const double *values = state->as<ompl::base::RealVectorStateSpace::StateType>()->values;
    projection(0) = so2Var(values[0]);
    projection(1) = so2Var(values[1]);
    projection(2) = values[2];
    projection(3) = values[3];
}
};

//
// Get a userdefined goal function
class MyGoal : public ompl::base::Goal
{
public:
    double eps;
    Matrix4d P;
    Vector4d goal;
public:
    MyGoal(const ompl::base::SpaceInformationPtr &si) : ompl::base::Goal(si)
    {
        eps = .5;
        P << 7.06569588, -2.49293786, -0.02423357, -1.51483124, -2.49293786,  1.99769639,  0.17239844,  0.99743335, -0.02423357,  0.17239844,  0.43822334, -0.1466905, -1.51483124,  0.99743335, -0.1466905 ,  0.93868745;
        P = P+0.05*Matrix4d::Identity();
        goal << M_PI-0.001,0.,0.,0.;//Check numerics for M_Pi
    }

    inline double getDistance( double phi0, double phi1, double phiDot0, double phiDot1)const{
        Vector4d pos;
        pos << phi0, phi1, phiDot0, phiDot1;
        pos -= goal;
        pos[0] = so2Var((double)pos[0]);
        pos[1] = so2Var((double)pos[1]);
        return (double) (pos.transpose()*P*pos);
    }

    virtual bool isSatisfied(const ompl::base::State *st) const{
        //Check if system is upright with small velocity
        const ompl::base::RealVectorStateSpace::StateType *ainvPendState = st->as<ompl::base::RealVectorStateSpace::StateType>();

        double dist = getDistance((double) (ainvPendState->values[0]), (double) (ainvPendState->values[1]), (double) (ainvPendState->values[3]), (double) (ainvPendState->values[4]));

        return (bool) (dist < eps);
    }
    virtual bool isSatisfied(const ompl::base::State *st, double *distance) const{
        const ompl::base::RealVectorStateSpace::StateType *ainvPendState = st->as<ompl::base::RealVectorStateSpace::StateType>();

        *distance = getDistance((double) (ainvPendState->values[0]), (double) (ainvPendState->values[1]), (double) (ainvPendState->values[3]), (double) (ainvPendState->values[4]));

        bool result = (bool) (*distance < eps);

        return result;
    }
};


bool isStateValid(const ompl::control::SpaceInformation *si, const ompl::base::State *state){
    //TBD: Actually do sth here
    return si->satisfiesBounds(state);
}

void propagate(const ompl::base::State *start, const ompl::control::Control *control, const double duration, ompl::base::State *result){
    const ompl::base::RealVectorStateSpace::StateType *ainvPendState = start->as<ompl::base::RealVectorStateSpace::StateType>();

    double X[4] = {ainvPendState->values[0], ainvPendState->values[1], ainvPendState->values[2], ainvPendState->values[3]};
    ctrlODE = control->as<ompl::control::RealVectorControlSpace::ControlType>()->values[0];

    integrateAcro(X, 0.0, duration);
    //Matrix<double, 1, 4> AA;
    for (size_t ii=0; ii<4; ++ii){
        result->as<ompl::base::RealVectorStateSpace::StateType>()->values[ii] = X[ii];
        //AA[0,ii] = X[ii];
    }
    //std::cout << AA << std::endl;
}

int main()
{
    cout << "Performing planning for inverted pendulum" << endl;

    //get the state-space: SO2+R^1
    ompl::base::StateSpacePtr acroStateSpace(new ompl::base::RealVectorStateSpace(4));
    ompl::base::RealVectorBounds bounds(4);
    bounds.low=std::vector<double>({-5., -10.,-30.,-40.});
    bounds.high=std::vector<double>({5.,  10., 30., 40.});
    acroStateSpace->as<ompl::base::RealVectorStateSpace>()->setBounds(bounds);
    acroStateSpace->registerProjection("myProjection", ompl::base::ProjectionEvaluatorPtr(new MyProjection(acroStateSpace)));
    //phi->setName("phi");
    //phiDot->setName("phiDot");

    //Get the control space -> real vector representing tau
    ompl::control::ControlSpacePtr cspace(new ompl::control::RealVectorControlSpace(acroStateSpace, 1));
    //Set bounds
    ompl::base::RealVectorBounds cbounds(1);
    cbounds.setLow(-10);
    cbounds.setHigh(10);

    cspace->as<ompl::control::RealVectorControlSpace>()->setBounds(cbounds);

    // construct an instance of  space information from this control space
    ompl::control::SpaceInformationPtr si(new ompl::control::SpaceInformation(acroStateSpace, cspace));
    // set state validity checking for this space
    si->setStateValidityChecker(std::bind(&isStateValid, si.get(),  std::placeholders::_1));

    si->setPropagationStepSize(0.05);//
    si->setMinMaxControlDuration(1,20);

    // set the state propagation routine
    si->setStatePropagator(std::bind(&propagate, std::placeholders::_1, std::placeholders::_2, std::placeholders::_3, std::placeholders::_4));
    //
    si->setup();

    //Get start and goal
    ompl::base::ScopedState<ompl::base::RealVectorStateSpace> start(acroStateSpace);
    //start->as<ompl::base::CompoundStateSpace>()->getSubspace("phi")->as<ompl::base::SO2StateSpace::StateType>()->value = M_PI;//This corresponds to stable eq
    //start->getSpace()->getSubspace("phi")->as<ompl::base::SO2StateSpace::StateType>()->value = M_PI;//This corresponds to stable eq
    //start->as<ompl::base::CompoundStateSpace>()->getSubspace("phiDot")->as<ompl::base::RealVectorStateSpace::StateType>()->values[0] = 0.0;
    start[0]=0.*M_PI-0.0;
    start[1]=0.;
    start[2]=0.;
    start[3]=0.;

    // create a problem instance ob::ProblemDefinitionPtr pdef(new ob::ProblemDefinition(si));
    ompl::base::ProblemDefinitionPtr pdef(new ompl::base::ProblemDefinition(si));

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
    ompl::base::PlannerPtr planner(new ompl::control::KPIECE1(si));

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

    if (solved){
        // get the goal representation from the problem definition (not the same as the goal state)
        // and inquire about the found path
        ompl::base::PathPtr path = pdef->getSolutionPath();
        std::cout << "Found solution:" << std::endl;

        // print the path to screen
        path->print(std::cout);
        path->as<ompl::control::PathControl>()->printAsMatrix(std::cout);
        ofstream resFile;
        resFile.open("acroSwingUp10New.txt");
        path->as<ompl::control::PathControl>()->printAsMatrix(resFile);
        resFile.close();
        //path->printAsMatrix(std::cout);
    }else{
        std::cout << "No solution found" << std::endl;
    }

    return 0;
}
