#include <QApplication>
#include <QSurfaceFormat>
#include <QVTKOpenGLNativeWidget.h>
#include "gui/MainWindow.h"

#include <iostream>
#include <string>
#include <cstring>
#include <filesystem>

static void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [options]\n"
              << "  --model PATH      Path to model directory (default: ./models/forearm/muscles/)\n"
              << "  --config PATH     Path to sensor_config.json (default: ./sensor_config.json)\n"
              << "  --host IP         Open Ephys host (default: 127.0.0.1)\n"
              << "  --port PORT       ZMQ data port (default: 5556)\n"
              << "  --window-ms MS    RMS window size in ms (default: 200)\n"
              << "  --frame-rate HZ   Update frequency in Hz (default: 60)\n"
              << "  --sigma METERS    Gaussian spread (default: 0.1)\n"
              << "  --radius METERS   Max sensor influence radius (default: 0.01)\n"
              << "  --clim-min VAL    Colormap minimum (default: 0.0)\n"
              << "  --clim-max VAL    Colormap maximum (default: 50.0)\n"
              << "  --help            Show this message\n";
}

static std::string getArg(int argc, char* argv[], const char* flag, const std::string& def) {
    for (int i = 1; i < argc - 1; ++i)
        if (std::strcmp(argv[i], flag) == 0)
            return argv[i + 1];
    return def;
}

static double getArgD(int argc, char* argv[], const char* flag, double def) {
    std::string s = getArg(argc, argv, flag, "");
    return s.empty() ? def : std::stod(s);
}

static int getArgI(int argc, char* argv[], const char* flag, int def) {
    std::string s = getArg(argc, argv, flag, "");
    return s.empty() ? def : std::stoi(s);
}

static bool hasFlag(int argc, char* argv[], const char* flag) {
    for (int i = 1; i < argc; ++i)
        if (std::strcmp(argv[i], flag) == 0) return true;
    return false;
}

int main(int argc, char* argv[])
{
    if (hasFlag(argc, argv, "--help") || hasFlag(argc, argv, "-h")) {
        printUsage(argv[0]);
        return 0;
    }

    // Auto-detect Qt plugin path relative to executable
    namespace fs = std::filesystem;
    fs::path exePath = fs::canonical(argv[0]);
    fs::path pluginPath = exePath.parent_path() / ".." / ".." / "vcpkg_installed" / "x64-windows" / "Qt6" / "plugins";
    if (fs::exists(pluginPath)) {
        std::string p = fs::canonical(pluginPath).string();
        qputenv("QT_PLUGIN_PATH", QByteArray::fromStdString(p));
    }

    // CRITICAL: Must set surface format before QApplication construction
    QSurfaceFormat::setDefaultFormat(QVTKOpenGLNativeWidget::defaultFormat());

    QApplication app(argc, argv);

    MainWindow::AppConfig cfg;
    cfg.modelPath  = getArg(argc, argv, "--model",      "./models/forearm/muscles/");
    cfg.configPath = getArg(argc, argv, "--config",     "./sensor_config.json");
    cfg.host       = getArg(argc, argv, "--host",       "127.0.0.1");
    cfg.port       = getArg(argc, argv, "--port",       "5556");
    cfg.windowMs   = getArgI(argc, argv, "--window-ms", 200);
    cfg.frameRate  = getArgI(argc, argv, "--frame-rate", 60);
    cfg.sigma      = getArgD(argc, argv, "--sigma",      0.1);
    cfg.radius     = getArgD(argc, argv, "--radius",     0.01);
    cfg.climMin    = getArgD(argc, argv, "--clim-min",   0.0);
    cfg.climMax    = getArgD(argc, argv, "--clim-max",   50.0);
    std::cerr << "[main] Starting MainWindow..." << std::endl;
    try {
        MainWindow window(cfg);
        std::cerr << "[main] MainWindow created, showing..." << std::endl;
        window.show();
        return app.exec();
    } catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Fatal error: unknown exception" << std::endl;
        return 1;
    }
}
