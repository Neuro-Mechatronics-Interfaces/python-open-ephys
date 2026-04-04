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
                   const Eigen::MatrixXd& sensorPoints);

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

    // Camera helpers
    Eigen::Vector3d sensorCentroid() const;

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

    // RBF precomputed data
    Eigen::MatrixXd m_weightsInRange; // (numInRange x nSensors) - sub-matrix of in-range rows
    std::vector<int> m_inRangeIndices;
    int m_nMeshPoints = 0;

    // Muscles
    std::vector<MuscleInfo> m_muscles;

    // Auxiliary part groups (bones, ligaments, nerves)
    std::vector<PartGroup> m_partGroups;

    // Sensors
    std::vector<int> m_channels;
    Eigen::MatrixXd  m_sensorPoints;
    vtkSmartPointer<vtkActor> m_sensorGlyphActor;
    std::vector<vtkSmartPointer<vtkBillboardTextActor3D>> m_sensorLabels;
    bool m_sensorMarkersVisible = true;
};
