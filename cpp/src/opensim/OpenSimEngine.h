#pragma once
#include <string>
#include <vector>
#include <memory>
#include <Eigen/Dense>

namespace OpenSim { class Model; }
namespace SimTK { class State; }

// Wraps the OpenSim WristModel to compute muscle activations from joint angles.
class OpenSimEngine {
public:
    struct MuscleState {
        std::string name;
        double activation;     // current activation level [0,1]
        double force;          // muscle force (N)
        double fiberLength;    // muscle fiber length (m)
    };

    struct JointLimits {
        std::string name;
        double minDeg;
        double maxDeg;
        double defaultDeg;
    };

    OpenSimEngine();
    ~OpenSimEngine();

    // Load an .osim model file
    bool loadModel(const std::string& osimPath);

    // Queries
    int muscleCount() const;
    int coordinateCount() const;
    std::string muscleName(int idx) const;
    std::string coordinateName(int idx) const;
    std::vector<std::string> muscleNames() const;
    std::vector<std::string> coordinateNames() const;
    JointLimits coordinateLimits(int idx) const;

    // Set a single coordinate value (in degrees)
    void setCoordinateDeg(int idx, double valueDeg);
    void setCoordinateDeg(const std::string& name, double valueDeg);

    // Set all coordinates at once (in degrees)
    void setAllCoordinatesDeg(const Eigen::VectorXd& valuesDeg);

    // Realize the state and compute equilibrium
    void realize();

    // Get current muscle states (after realize)
    std::vector<MuscleState> muscleStates() const;

    // Get activation vector (nMuscles) — quick accessor
    Eigen::VectorXd activations() const;

    // Get moment arm matrix (nMuscles x nCoordinates)
    // Each entry is the moment arm of muscle m about coordinate c
    Eigen::MatrixXd momentArmMatrix() const;

    // Compute static optimization: given target joint torques,
    // find muscle activations that produce them (minimizing sum of squared activations)
    Eigen::VectorXd staticOptimization(const Eigen::VectorXd& targetTorques) const;

    bool isLoaded() const { return m_loaded; }

private:
    std::unique_ptr<OpenSim::Model> m_model;
    std::unique_ptr<SimTK::State> m_state;
    bool m_loaded = false;
};
