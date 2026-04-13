#include "OpenSimEngine.h"

#include <OpenSim/OpenSim.h>
#include <iostream>
#include <cmath>

static constexpr double DEG2RAD = 3.14159265358979323846 / 180.0;
static constexpr double RAD2DEG = 180.0 / 3.14159265358979323846;

OpenSimEngine::OpenSimEngine() = default;
OpenSimEngine::~OpenSimEngine() = default;

bool OpenSimEngine::loadModel(const std::string& osimPath)
{
    try {
        m_model = std::make_unique<OpenSim::Model>(osimPath);
        m_model->setUseVisualizer(false);

        // Initialize the system and get a default state
        auto& state = m_model->initSystem();
        m_state = std::make_unique<SimTK::State>(SimTK::State(state));

        // Realize to position stage
        m_model->realizePosition(*m_state);

        m_loaded = true;
        std::cout << "OpenSim model loaded: " << osimPath << "\n"
                  << "  Muscles: " << muscleCount() << "\n"
                  << "  Coordinates: " << coordinateCount() << "\n";

        // Print coordinate ranges
        for (int i = 0; i < coordinateCount(); ++i) {
            auto lim = coordinateLimits(i);
            std::cout << "  " << lim.name << ": ["
                      << lim.minDeg << ", " << lim.maxDeg << "] deg"
                      << " default=" << lim.defaultDeg << "\n";
        }

        return true;
    }
    catch (const std::exception& e) {
        std::cerr << "Failed to load OpenSim model: " << e.what() << "\n";
        m_loaded = false;
        return false;
    }
}

int OpenSimEngine::muscleCount() const
{
    if (!m_loaded) return 0;
    return static_cast<int>(m_model->getMuscles().getSize());
}

int OpenSimEngine::coordinateCount() const
{
    if (!m_loaded) return 0;
    return static_cast<int>(m_model->getCoordinateSet().getSize());
}

std::string OpenSimEngine::muscleName(int idx) const
{
    if (!m_loaded || idx < 0 || idx >= muscleCount()) return "";
    return m_model->getMuscles().get(idx).getName();
}

std::string OpenSimEngine::coordinateName(int idx) const
{
    if (!m_loaded || idx < 0 || idx >= coordinateCount()) return "";
    return m_model->getCoordinateSet().get(idx).getName();
}

std::vector<std::string> OpenSimEngine::muscleNames() const
{
    std::vector<std::string> names;
    for (int i = 0; i < muscleCount(); ++i)
        names.push_back(muscleName(i));
    return names;
}

std::vector<std::string> OpenSimEngine::coordinateNames() const
{
    std::vector<std::string> names;
    for (int i = 0; i < coordinateCount(); ++i)
        names.push_back(coordinateName(i));
    return names;
}

OpenSimEngine::JointLimits OpenSimEngine::coordinateLimits(int idx) const
{
    JointLimits lim{};
    if (!m_loaded || idx < 0 || idx >= coordinateCount()) return lim;

    const auto& coord = m_model->getCoordinateSet().get(idx);
    lim.name = coord.getName();
    lim.minDeg = coord.getRangeMin() * RAD2DEG;
    lim.maxDeg = coord.getRangeMax() * RAD2DEG;
    lim.defaultDeg = coord.getDefaultValue() * RAD2DEG;
    return lim;
}

void OpenSimEngine::setCoordinateDeg(int idx, double valueDeg)
{
    if (!m_loaded || idx < 0 || idx >= coordinateCount()) return;
    auto& coord = m_model->updCoordinateSet().get(idx);
    coord.setValue(*m_state, valueDeg * DEG2RAD);
}

void OpenSimEngine::setCoordinateDeg(const std::string& name, double valueDeg)
{
    if (!m_loaded) return;
    auto& coordSet = m_model->updCoordinateSet();
    for (int i = 0; i < coordSet.getSize(); ++i) {
        if (coordSet.get(i).getName() == name) {
            coordSet.get(i).setValue(*m_state, valueDeg * DEG2RAD);
            return;
        }
    }
    std::cerr << "OpenSim: coordinate '" << name << "' not found\n";
}

void OpenSimEngine::setAllCoordinatesDeg(const Eigen::VectorXd& valuesDeg)
{
    if (!m_loaded) return;
    int n = std::min(static_cast<int>(valuesDeg.size()), coordinateCount());
    for (int i = 0; i < n; ++i)
        setCoordinateDeg(i, valuesDeg(i));
}

void OpenSimEngine::realize()
{
    if (!m_loaded) return;

    // Realize through dynamics stage
    m_model->realizeVelocity(*m_state);
    m_model->realizeDynamics(*m_state);

    // Equilibrate muscles (compute tendon force equilibrium)
    m_model->equilibrateMuscles(*m_state);
}

std::vector<OpenSimEngine::MuscleState> OpenSimEngine::muscleStates() const
{
    std::vector<MuscleState> result;
    if (!m_loaded) return result;

    const auto& muscles = m_model->getMuscles();
    for (int i = 0; i < muscles.getSize(); ++i) {
        const auto& m = muscles.get(i);
        MuscleState ms;
        ms.name = m.getName();
        ms.activation = m.getActivation(*m_state);
        ms.force = m.getActuation(*m_state);
        ms.fiberLength = m.getFiberLength(*m_state);
        result.push_back(ms);
    }
    return result;
}

Eigen::VectorXd OpenSimEngine::activations() const
{
    int n = muscleCount();
    Eigen::VectorXd a(n);
    if (!m_loaded) { a.setZero(); return a; }

    const auto& muscles = m_model->getMuscles();
    for (int i = 0; i < n; ++i)
        a(i) = muscles.get(i).getActivation(*m_state);
    return a;
}

Eigen::MatrixXd OpenSimEngine::momentArmMatrix() const
{
    int nM = muscleCount();
    int nC = coordinateCount();
    Eigen::MatrixXd ma = Eigen::MatrixXd::Zero(nM, nC);
    if (!m_loaded) return ma;

    const auto& muscles = m_model->getMuscles();
    auto& coords = m_model->updCoordinateSet();

    for (int i = 0; i < nM; ++i) {
        const auto& muscle = muscles.get(i);
        for (int j = 0; j < nC; ++j) {
            auto& coord = coords.get(j);
            ma(i, j) = muscle.computeMomentArm(*m_state, coord);
        }
    }
    return ma;
}

Eigen::VectorXd OpenSimEngine::staticOptimization(
    const Eigen::VectorXd& targetTorques) const
{
    // Simple static optimization: minimize sum(a_i^2) subject to:
    //   M * a = tau (moment arm matrix * activations = joint torques)
    // Uses least-norm solution: a = M^T * (M * M^T)^{-1} * tau
    // Then clamp to [0, 1]

    Eigen::MatrixXd M = momentArmMatrix(); // (nMuscles x nCoords)
    int nM = static_cast<int>(M.rows());
    int nC = static_cast<int>(M.cols());

    if (targetTorques.size() != nC) {
        std::cerr << "staticOptimization: torque vector size mismatch\n";
        return Eigen::VectorXd::Zero(nM);
    }

    // M^T is (nCoords x nMuscles), M * M^T is (nMuscles x nMuscles) — wrong
    // Actually we need: M^T * a = tau where M is (nMuscles x nCoords)
    // So tau = M^T * a where M^T is (nCoords x nMuscles)
    // This is underdetermined: nCoords equations, nMuscles unknowns (nMuscles > nCoords)
    // Least-norm: a = M * (M^T * M)^{-1} * tau

    Eigen::MatrixXd Mt = M.transpose(); // (nCoords x nMuscles)
    Eigen::MatrixXd MtM = Mt * M;       // (nCoords x nCoords)

    Eigen::VectorXd a = M * MtM.ldlt().solve(targetTorques);

    // Clamp activations to [0, 1]
    for (int i = 0; i < nM; ++i)
        a(i) = std::clamp(a(i), 0.0, 1.0);

    return a;
}
