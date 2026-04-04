#pragma once
#include <QMainWindow>
#include <QTimer>
#include <memory>
#include <Eigen/Dense>

#include <vtkSmartPointer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkOrientationMarkerWidget.h>
#include <QVTKOpenGLNativeWidget.h>

class ControlPanel;
class HeatmapScene;
class ZmqClient;

class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    struct AppConfig {
        std::string modelPath   = "./models/forearm/muscles/";
        std::string configPath  = "./sensor_config.json";
        std::string host        = "127.0.0.1";
        std::string port        = "5556";
        int    windowMs         = 200;
        int    frameRate        = 60;
        double sigma            = 0.015;
        double radius           = 0.04;
        double climMin          = 0.0;
        double climMax          = 50.0;
    };

    explicit MainWindow(const AppConfig& config, QWidget* parent = nullptr);
    ~MainWindow() override;

    vtkRenderer* renderer() { return m_renderer; }
    HeatmapScene* scene() { return m_scene.get(); }
    bool isSelectionLocked() const;

public slots:
    void onUpdate();
    void onMuscleToggled(int idx, bool visible);
    void onVizFunctionChanged(const QString& name);
    void onPartToggled(const QString& name, bool visible);
    void onWindowMsChanged(int ms);

private:
    void setupVtk();
    void setupScene();
    void setupCamera();
    void setupPicking();

    Eigen::VectorXd applyVizFunction(const Eigen::MatrixXf& Y) const;

    AppConfig m_config;

    // VTK
    QVTKOpenGLNativeWidget*                     m_vtkWidget = nullptr;
    vtkSmartPointer<vtkGenericOpenGLRenderWindow> m_renderWindow;
    vtkSmartPointer<vtkRenderer>                 m_renderer;

    // App components
    std::unique_ptr<ZmqClient>    m_client;
    std::unique_ptr<HeatmapScene> m_scene;
    ControlPanel*                 m_controlPanel = nullptr;

    // State
    QTimer*  m_timer = nullptr;
    int64_t  m_lastSampleIdx = 0;

    enum class VizFunc { RMS, AbsSpikeAmplitude };
    VizFunc m_vizFunc = VizFunc::RMS;

    // Sensor config
    std::vector<int> m_sensorChannels;
    int m_nSensors = 0;

    // Keep alive
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axesWidget;
};
