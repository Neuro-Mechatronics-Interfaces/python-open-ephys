#include "MainWindow.h"
#include "ControlPanel.h"
#include "HeatmapScene.h"
#include "SensorConfig.h"
#include "ZmqClient.h"

#include <QDockWidget>
#include <iostream>
#include <cmath>
#include <algorithm>
#include <set>
#include <filesystem>

#include <vtkCamera.h>
#include <vtkAxesActor.h>
#include <vtkOrientationMarkerWidget.h>
#include <vtkRenderWindowInteractor.h>
#include <vtkInteractorStyleTrackballCamera.h>
#include <vtkCellPicker.h>
#include <vtkCommand.h>
#include <vtkCell.h>

// ─── Cell-pick callback ───

class PickCallback : public vtkCommand {
public:
    static PickCallback* New() { return new PickCallback; }
    vtkTypeMacro(PickCallback, vtkCommand);
    void setMainWindow(MainWindow* w) { m_window = w; }

    void Execute(vtkObject* caller, unsigned long /*eventId*/, void*) override {
        if (m_window->isSelectionLocked()) return;

        auto* interactor = static_cast<vtkRenderWindowInteractor*>(caller);
        int x, y;
        interactor->GetEventPosition(x, y);

        auto picker = vtkSmartPointer<vtkCellPicker>::New();
        picker->SetTolerance(0.005);
        if (picker->Pick(x, y, 0, m_window->renderer())) {
            vtkIdType cellId = picker->GetCellId();
            if (cellId >= 0 && picker->GetDataSet()) {
                vtkCell* cell = picker->GetDataSet()->GetCell(cellId);
                if (cell && cell->GetNumberOfPoints() > 0) {
                    int vertexId = static_cast<int>(cell->GetPointId(0));
                    int muscleIdx = m_window->scene()->findMuscleAtVertex(vertexId);
                    if (muscleIdx >= 0) {
                        bool newState = !m_window->scene()->isMuscleVisible(muscleIdx);
                        m_window->onMuscleToggled(muscleIdx, newState);
                    }
                }
            }
        }
    }
private:
    MainWindow* m_window = nullptr;
};

// ─── MainWindow ───

MainWindow::MainWindow(const AppConfig& config, QWidget* parent)
    : QMainWindow(parent), m_config(config)
{
    setWindowTitle("EMG 3D Heatmap");
    resize(1200, 800);

    setupVtk();
    setupScene();
    setupCamera();
    setupPicking();

    // Control panel dock
    m_controlPanel = new ControlPanel(m_config.windowMs);
    auto* dock = new QDockWidget("Display Settings", this);
    dock->setWidget(m_controlPanel);
    dock->setMinimumWidth(400);
    addDockWidget(Qt::RightDockWidgetArea, dock);

    // Populate muscle buttons
    std::vector<std::string> names;
    for (int i = 0; i < m_scene->muscleCount(); ++i)
        names.push_back(m_scene->muscleName(i));
    m_controlPanel->setMuscleNames(names);

    // Sensor markers hidden by default
    m_scene->setSensorMarkersVisible(false);

    // Connect signals
    connect(m_controlPanel, &ControlPanel::muscleToggled,
            this, &MainWindow::onMuscleToggled);
    connect(m_controlPanel, &ControlPanel::vizFunctionChanged,
            this, &MainWindow::onVizFunctionChanged);
    connect(m_controlPanel, &ControlPanel::partToggled,
            this, &MainWindow::onPartToggled);
    connect(m_controlPanel, &ControlPanel::sensorMarkersToggled,
            this, [this](bool visible) {
                m_scene->setSensorMarkersVisible(visible);
                m_renderWindow->Render();
            });
    connect(m_controlPanel, &ControlPanel::windowMsChanged,
            this, &MainWindow::onWindowMsChanged);

    // Update timer
    m_timer = new QTimer(this);
    connect(m_timer, &QTimer::timeout, this, &MainWindow::onUpdate);
    m_timer->start(1000 / m_config.frameRate);

    std::cout << "\n3D viewer is running. Close the window to stop.\n" << std::endl;
}

MainWindow::~MainWindow()
{
    m_timer->stop();
    if (m_client) m_client->stop();
}

// ─── VTK Setup ───

void MainWindow::setupVtk()
{
    m_vtkWidget = new QVTKOpenGLNativeWidget(this);
    m_renderWindow = vtkSmartPointer<vtkGenericOpenGLRenderWindow>::New();
    m_vtkWidget->setRenderWindow(m_renderWindow);

    m_renderer = vtkSmartPointer<vtkRenderer>::New();
    m_renderer->SetBackground(1.0, 1.0, 1.0);
    m_renderer->SetUseDepthPeeling(1);
    m_renderer->SetMaximumNumberOfPeels(4);
    m_renderer->SetOcclusionRatio(0.1);
    m_renderWindow->AddRenderer(m_renderer);

    setCentralWidget(m_vtkWidget);

    // Axes widget
    auto axes = vtkSmartPointer<vtkAxesActor>::New();
    m_axesWidget = vtkSmartPointer<vtkOrientationMarkerWidget>::New();
    m_axesWidget->SetOrientationMarker(axes);
    m_axesWidget->SetInteractor(m_renderWindow->GetInteractor());
    m_axesWidget->SetViewport(0.0, 0.0, 0.2, 0.2);
    m_axesWidget->EnabledOn();
    m_axesWidget->InteractiveOff();

    m_renderWindow->GetInteractor()->SetInteractorStyle(
        vtkSmartPointer<vtkInteractorStyleTrackballCamera>::New());
}

// ─── Scene Setup ───

void MainWindow::setupScene()
{
    // Load sensor config
    SensorConfig sc = SensorConfig::load(m_config.configPath);
    m_sensorChannels = sc.channels;
    m_nSensors = static_cast<int>(sc.channels.size());
    std::cout << "Loaded " << m_nSensors << " sensors from " << m_config.configPath << "\n";

    // Create scene
    HeatmapScene::Params sp;
    sp.sigma   = m_config.sigma;
    sp.radius  = m_config.radius;
    sp.climMin = m_config.climMin;
    sp.climMax = m_config.climMax;
    m_scene = std::make_unique<HeatmapScene>(m_renderer, sp);
    m_scene->loadModel(m_config.modelPath, sc.channels, sc.positions);

    // Load auxiliary part groups from sibling dirs of muscle path
    namespace fs = std::filesystem;
    fs::path muscleDir = fs::path(m_config.modelPath);
    // Strip trailing separator: "muscles/" -> "muscles", then parent gives "forearm"
    if (muscleDir.filename().empty())
        muscleDir = muscleDir.parent_path();
    fs::path baseDir = muscleDir.parent_path();
    std::cerr << "[MainWindow] modelPath='" << m_config.modelPath
              << "' muscleDir='" << muscleDir.string()
              << "' baseDir='" << baseDir.string() << "'\n";
    std::cerr.flush();
    m_scene->loadPartGroup((baseDir / "bones").string(),      "Bones",      0.9, 0.9, 0.85, 0.5);
    m_scene->loadPartGroup((baseDir / "ligaments").string(), "Ligaments",  0.4, 0.7, 0.4,  0.5);
    m_scene->loadPartGroup((baseDir / "membranes").string(), "Membranes",  0.6, 0.5, 0.7,  0.4);
    m_scene->loadPartGroup((baseDir / "arteries").string(),  "Arteries",   0.8, 0.15, 0.15, 0.6);
    m_scene->loadPartGroup((baseDir / "cartilages").string(),"Cartilages", 0.5, 0.7, 0.9,  0.5);

    // Start ZMQ client
    ZmqClient::Config zc;
    zc.host     = m_config.host;
    zc.dataPort = m_config.port;
    zc.verbose  = true;
    m_client = std::make_unique<ZmqClient>(zc);
    m_client->start();
    std::cout << "ZMQ client started, connecting to " << m_config.host
              << ":" << m_config.port << " ...\n";
}

// ─── Camera Setup ───

void MainWindow::setupCamera()
{
    Eigen::Vector3d center = m_scene->sensorCentroid();
    auto* cam = m_renderer->GetActiveCamera();
    cam->SetFocalPoint(center.x(), center.y(), center.z());
    cam->SetPosition(center.x() + 0.2, center.y() + 0.1, center.z() + 0.1);
    m_renderer->ResetCameraClippingRange();
}

// ─── Cell Picking Setup ───

void MainWindow::setupPicking()
{
    auto callback = vtkSmartPointer<PickCallback>::New();
    callback->setMainWindow(this);
    m_renderWindow->GetInteractor()->AddObserver(
        vtkCommand::LeftButtonPressEvent, callback);
}

// ─── Update Loop (called by QTimer) ───

void MainWindow::onUpdate()
{
    if (!m_client || !m_client->isReady())
        return;

    // Auto-detect channels on first data
    if (m_client->channelIndex().empty()) {
        auto seen = m_client->seenChannels();
        std::vector<int> sorted(seen.begin(), seen.end());
        std::sort(sorted.begin(), sorted.end());
        m_client->setChannelIndex(sorted);
        std::cout << "Channels detected: " << sorted.size()
                  << ", sample rate: " << m_client->sampleRate() << " Hz\n";
    }

    int windowSamples = static_cast<int>(
        m_client->sampleRate() * m_config.windowMs / 1000.0);

    ZmqClient::Snapshot snap;
    try {
        snap = m_client->getLatest(windowSamples);
    } catch (...) {
        return;
    }

    if (snap.Y.size() == 0) return;

    // Skip if same data as last frame
    int64_t currentIdx = static_cast<int64_t>(
        snap.t(snap.t.size() - 1) * m_client->sampleRate());
    if (currentIdx == m_lastSampleIdx) return;
    m_lastSampleIdx = currentIdx;

    // Apply visualization function
    Eigen::VectorXd data = applyVizFunction(snap.Y);

    // Map channel values to sensor order
    Eigen::VectorXd sensorValues = Eigen::VectorXd::Zero(m_nSensors);
    for (int i = 0; i < m_nSensors; ++i) {
        int ch = m_sensorChannels[i];
        if (ch < static_cast<int>(data.size()))
            sensorValues(i) = data(ch);
    }

    m_scene->updateScalars(sensorValues);
    m_renderWindow->Render();
}

// ─── Viz Function ───

Eigen::VectorXd MainWindow::applyVizFunction(const Eigen::MatrixXf& Y) const
{
    int nCh = static_cast<int>(Y.rows());
    Eigen::VectorXd result(nCh);

    switch (m_vizFunc) {
    case VizFunc::RMS:
        for (int i = 0; i < nCh; ++i)
            result(i) = std::sqrt(
                Y.row(i).cast<double>().squaredNorm() / Y.cols());
        break;
    case VizFunc::AbsSpikeAmplitude:
        for (int i = 0; i < nCh; ++i)
            result(i) = Y.row(i).cast<double>().cwiseAbs().maxCoeff();
        break;
    }
    return result;
}

// ─── Slots ───

void MainWindow::onMuscleToggled(int idx, bool visible)
{
    m_scene->setMuscleVisible(idx, visible);
    m_controlPanel->setMuscleChecked(idx, visible);
    m_renderWindow->Render();
}

bool MainWindow::isSelectionLocked() const
{
    return m_controlPanel && m_controlPanel->isLocked();
}

void MainWindow::onVizFunctionChanged(const QString& name)
{
    if (name == "RMS Amplitude")
        m_vizFunc = VizFunc::RMS;
    else if (name == "Abs Spike Amplitude")
        m_vizFunc = VizFunc::AbsSpikeAmplitude;

    m_controlPanel->setActiveVizFunction(name);
    m_scene->rebuildMeshActor();
    m_renderWindow->Render();
}

void MainWindow::onPartToggled(const QString& name, bool visible)
{
    m_scene->setPartVisible(name.toStdString(), visible);
    m_renderWindow->Render();
}

void MainWindow::onWindowMsChanged(int ms)
{
    m_config.windowMs = ms;
}
