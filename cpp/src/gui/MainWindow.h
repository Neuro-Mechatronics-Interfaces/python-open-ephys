#pragma once
#include <QMainWindow>
#include <QTimer>
#include <QElapsedTimer>
#include <QDockWidget>
#include <memory>
#include <Eigen/Dense>

#include <vtkSmartPointer.h>
#include <vtkGenericOpenGLRenderWindow.h>
#include <vtkRenderer.h>
#include <vtkOrientationMarkerWidget.h>
#include <QVTKOpenGLNativeWidget.h>

class ControlPanel;
class CalibrationPanel;
class HeatmapScene;
class ZmqClient;
class CalibrationEngine;

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
    std::unique_ptr<ZmqClient>      m_client;
    std::unique_ptr<HeatmapScene>   m_scene;
    ControlPanel*                   m_controlPanel = nullptr;

    // State
    QTimer*  m_timer = nullptr;
    int64_t  m_lastSampleIdx = 0;

    enum class VizFunc { RMS, AbsSpikeAmplitude };
    VizFunc m_vizFunc = VizFunc::RMS;

    // Sensor config
    std::vector<int> m_sensorChannels;
    int m_nSensors = 0;

    // Calibration
    CalibrationPanel*                       m_calibrationPanel = nullptr;
    QDockWidget*                            m_calibrationDock = nullptr;
    std::unique_ptr<CalibrationEngine>      m_calibEngine;

    enum class CalibState { Off, WaitingForRecord, Recording, Solving, Done };
    CalibState m_calibState = CalibState::Off;
    int        m_calibMovementIdx = 0;
    QElapsedTimer m_calibRecordTimer;
    std::vector<Eigen::VectorXd> m_calibRmsAccumulator;

    static constexpr int kCalibRecordDurationMs = 3000;
    static constexpr double kStripActiveThreshold = 10.0;

    void setupCalibration();
    void startCalibration();
    void onCalibRecordRequested();
    void onCalibMovementSkipped();
    void onCalibAccepted(double thetaDeg);
    void onCalibRetry();
    void onCalibCancelled();
    void finishCalibRecording();
    void solveCalibration();

    // Keep alive
    vtkSmartPointer<vtkOrientationMarkerWidget> m_axesWidget;
};
