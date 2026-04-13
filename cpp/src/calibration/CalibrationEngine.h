#pragma once
#include "CalibrationData.h"
#include <vector>
#include <map>

class CalibrationEngine {
public:
    struct MuscleAngularPosition {
        std::string groupName;
        double angleDeg;  // [-180, +180] around forearm axis
    };

    struct CalibrationResult {
        double thetaOffsetDeg;
        double score;
        double maxPossibleScore;
        std::vector<double> scoreProfile;  // 360 entries, [-180..+179]
    };

    CalibrationEngine();

    void setMuscleAngularPositions(const std::vector<MuscleAngularPosition>& positions);
    void setStripDefinitions(const std::vector<StripInfo>& strips);
    void addObservation(const MovementObservation& obs);
    void clearObservations();

    const std::vector<StripInfo>& strips() const { return m_strips; }
    int observationCount() const { return static_cast<int>(m_observations.size()); }

    CalibrationResult solve() const;

private:
    double scoreTheta(double thetaDeg) const;
    bool stripCoversMuscle(const StripInfo& strip, double muscleDeg,
                           double thetaDeg) const;

    static constexpr double kStripHalfWidthDeg = 30.0;

    std::vector<MuscleAngularPosition> m_musclePositions;
    std::vector<StripInfo> m_strips;
    std::vector<MovementObservation> m_observations;
    std::vector<CalibrationMovement> m_movements;
};
