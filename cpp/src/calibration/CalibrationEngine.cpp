#include "CalibrationEngine.h"
#include <cmath>
#include <algorithm>
#include <iostream>
#include <limits>

static double wrapTo180(double deg)
{
    while (deg > 180.0) deg -= 360.0;
    while (deg <= -180.0) deg += 360.0;
    return deg;
}

CalibrationEngine::CalibrationEngine()
{
    m_movements = CalibrationData::getMovements();
}

void CalibrationEngine::setMuscleAngularPositions(
    const std::vector<MuscleAngularPosition>& positions)
{
    m_musclePositions = positions;
}

void CalibrationEngine::setStripDefinitions(const std::vector<StripInfo>& strips)
{
    m_strips = strips;
}

void CalibrationEngine::addObservation(const MovementObservation& obs)
{
    m_observations.push_back(obs);
}

void CalibrationEngine::clearObservations()
{
    m_observations.clear();
}

bool CalibrationEngine::stripCoversMuscle(const StripInfo& strip,
                                           double muscleDeg,
                                           double thetaDeg) const
{
    double adjustedCenter = wrapTo180(strip.centerAngleDeg + thetaDeg);
    double diff = wrapTo180(muscleDeg - adjustedCenter);
    return std::abs(diff) <= kStripHalfWidthDeg;
}

double CalibrationEngine::scoreTheta(double thetaDeg) const
{
    double totalScore = 0.0;

    for (const auto& obs : m_observations) {
        if (obs.movementIndex < 0 ||
            obs.movementIndex >= static_cast<int>(m_movements.size()))
            continue;

        const auto& expected = m_movements[obs.movementIndex].expectedMuscles;
        int nStrips = static_cast<int>(m_strips.size());

        for (int s = 0; s < nStrips; ++s) {
            if (s >= static_cast<int>(obs.stripActive.size()))
                continue;

            // Check if this strip covers any expected muscle at this theta
            bool coversExpected = false;
            for (const auto& mp : m_musclePositions) {
                if (expected.count(mp.groupName) > 0 &&
                    stripCoversMuscle(m_strips[s], mp.angleDeg, thetaDeg))
                {
                    coversExpected = true;
                    break;
                }
            }

            bool active = obs.stripActive[s];

            if (active && coversExpected)
                totalScore += 1.0;   // true positive
            else if (!active && !coversExpected)
                totalScore += 0.5;   // true negative
            else if (active && !coversExpected)
                totalScore -= 0.5;   // false positive
            else // !active && coversExpected
                totalScore -= 1.0;   // false negative
        }
    }
    return totalScore;
}

CalibrationEngine::CalibrationResult CalibrationEngine::solve() const
{
    CalibrationResult result;
    result.scoreProfile.resize(360, 0.0);
    result.thetaOffsetDeg = 0.0;
    result.score = -std::numeric_limits<double>::max();

    // Compute max possible score: TP(+1) for active strips, TN(+0.5) for inactive
    result.maxPossibleScore = 0.0;
    for (const auto& obs : m_observations) {
        int nStrips = std::min(static_cast<int>(m_strips.size()),
                               static_cast<int>(obs.stripActive.size()));
        for (int s = 0; s < nStrips; ++s)
            result.maxPossibleScore += obs.stripActive[s] ? 1.0 : 0.5;
    }

    // Brute-force scan
    for (int deg = -180; deg < 180; ++deg) {
        double theta = static_cast<double>(deg);
        double s = scoreTheta(theta);
        result.scoreProfile[deg + 180] = s;

        if (s > result.score) {
            result.score = s;
            result.thetaOffsetDeg = theta;
        }
    }

    std::cout << "Calibration solved: theta=" << result.thetaOffsetDeg
              << " deg, score=" << result.score
              << "/" << result.maxPossibleScore << "\n";

    return result;
}
