#pragma once
#include <string>
#include <vector>
#include <cmath>
#include <Eigen/Dense>

#include <vtkSmartPointer.h>
#include <vtkPolyData.h>
#include <vtkPolyDataMapper.h>
#include <vtkActor.h>
#include <vtkRenderer.h>
#include <vtkLookupTable.h>
#include <vtkScalarBarActor.h>
#include <vtkFloatArray.h>
#include <vtkBillboardTextActor3D.h>
#include <vtkStaticCellLocator.h>

struct MuscleInfo {
    std::string name;
    int vertexStart;
    int vertexEnd;
    bool visible = true;
};

struct PartGroup {
    std::string name;
    std::vector<vtkSmartPointer<vtkActor>> actors;
    bool visible = false;
};

class HeatmapScene {
public:
    struct Params {
        double sigma   = 0.1;
        double radius  = 0.01;
        double climMin = 0.0;
        double climMax = 50.0;
    };

    HeatmapScene(vtkRenderer* renderer, const Params& params);

    // Load all STL files from directory, merge, compute RBF weights
    void loadModel(const std::string& modelDir,
                   const std::vector<int>& channels,
                   const Eigen::MatrixXd& sensorPoints,
                   const Eigen::Vector3d& axisOrigin = Eigen::Vector3d::Zero(),
                   const Eigen::Vector3d& axisDir = Eigen::Vector3d(0, 0, 1),
                   bool hasAxis = false);

    // Load an auxiliary part group (bones, ligaments, nerves) from a directory
    void loadPartGroup(const std::string& dir, const std::string& name,
                       double r, double g, double b, double opacity = 0.6);

    // Toggle visibility of a part group by name
    void setPartVisible(const std::string& name, bool visible);
    bool isPartVisible(const std::string& name) const;

    // Update vertex scalars from per-sensor values. Called each frame.
    void updateScalars(const Eigen::VectorXd& sensorValues);

    // Muscle visibility
    void setMuscleVisible(int idx, bool visible);
    bool isMuscleVisible(int idx) const;
    int  muscleCount() const { return static_cast<int>(m_muscles.size()); }
    const std::string& muscleName(int idx) const { return m_muscles[idx].name; }
    int  findMuscleAtVertex(int vertexId) const;

    // Update the scalar bar title (when viz function changes)
    void updateScalarBarTitle(const std::string& title);
    // Rebuild the actor (needed when switching viz function to reset scalar bar)
    void rebuildMeshActor();

    // Sensor marker visibility
    void setSensorMarkersVisible(bool visible);
    bool sensorMarkersVisible() const { return m_sensorMarkersVisible; }

    // Fascia rotation: rotate sensors around forearm axis and recompute RBF
    void setFasciaRotation(double angleDeg);

    // Camera helpers
    Eigen::Vector3d sensorCentroid() const;

    // Calibration: compute angular position of each muscle centroid
    // Returns (muscle_display_name, angle_degrees) pairs
    std::vector<std::pair<std::string, double>> computeMuscleAngularPositions() const;

    // Accessors for calibration
    const Eigen::Vector3d& forearmAxisOrigin() const { return m_forearmAxisOrigin; }
    const Eigen::Vector3d& forearmAxisDir() const { return m_forearmAxisDir; }
    const Eigen::MatrixXd& baseSensorPoints() const { return m_baseSensorPoints; }

    int sensorCount() const { return static_cast<int>(m_channels.size()); }
    const std::vector<int>& channels() const { return m_channels; }

private:
    void buildLookupTable();
    void computeRbfWeights(const Eigen::MatrixXd& meshPoints,
                           const Eigen::MatrixXd& sensorPoints);
    void addSensorMarkers(const Eigen::MatrixXd& sensorPoints,
                          const std::vector<int>& channels);
    static std::string formatMuscleName(const std::string& filename);

    vtkRenderer* m_renderer;
    Params m_params;

    // Merged mesh
    vtkSmartPointer<vtkPolyData>       m_mesh;
    vtkSmartPointer<vtkPolyDataMapper> m_mapper;
    vtkSmartPointer<vtkActor>          m_actor;
    vtkSmartPointer<vtkLookupTable>    m_lut;
    vtkSmartPointer<vtkScalarBarActor> m_scalarBar;
    vtkSmartPointer<vtkFloatArray>     m_scalarsArray;

    // Radial raycast: project sensor positions onto muscle mesh surface
    Eigen::Vector3d projectRadiallyToMesh(const Eigen::Vector3d& pt) const;

    // RBF precomputed data
    Eigen::MatrixXd m_weightsInRange; // (numInRange x nSensors) - sub-matrix of in-range rows
    std::vector<int> m_inRangeIndices;
    int m_nMeshPoints = 0;

    // Cell locator for radial raycast
    vtkSmartPointer<vtkStaticCellLocator> m_cellLocator;

    // Muscles
    std::vector<MuscleInfo> m_muscles;

    // Auxiliary part groups (bones, ligaments, nerves)
    std::vector<PartGroup> m_partGroups;

    // Sensors
    std::vector<int> m_channels;
    Eigen::MatrixXd  m_sensorPoints;        // current (rotated) positions
    Eigen::MatrixXd  m_baseSensorPoints;    // original unrotated positions
    Eigen::MatrixXd  m_meshPoints;          // cached mesh vertex positions
    Eigen::Vector3d  m_forearmAxisOrigin;
    Eigen::Vector3d  m_forearmAxisDir;
    Eigen::Vector3d  m_basisU;   // perpendicular frame for angular computation
    Eigen::Vector3d  m_basisV;
    double           m_currentFasciaAngle = 0.0;
    vtkSmartPointer<vtkActor> m_sensorGlyphActor;
    std::vector<vtkSmartPointer<vtkBillboardTextActor3D>> m_sensorLabels;
    bool m_sensorMarkersVisible = true;
};
