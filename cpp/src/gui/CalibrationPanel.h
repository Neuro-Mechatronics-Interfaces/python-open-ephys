#pragma once
#include <QWidget>
#include <QPushButton>
#include <QLabel>
#include <QProgressBar>
#include <QVBoxLayout>
#include <vector>
#include <string>

class CalibrationPanel : public QWidget {
    Q_OBJECT
public:
    explicit CalibrationPanel(QWidget* parent = nullptr);

    void setMovements(const std::vector<std::pair<std::string, std::string>>& movements);

    // State transitions
    void setCurrentMovement(int index);
    void setMovementComplete(int index);
    void setRecording(bool recording);
    void setProgress(double fraction);  // 0.0 to 1.0

    // Show result after solving
    void showResult(double thetaDeg, double confidence);
    void hideResult();

signals:
    void recordingRequested();
    void movementSkipped();
    void calibrationAccepted(double thetaDeg);
    void calibrationRetryRequested();
    void calibrationCancelled();

private:
    struct MovementRow {
        QWidget* container;
        QLabel*  statusIcon;
        QLabel*  nameLabel;
        QLabel*  instructionLabel;
        QProgressBar* progressBar;
    };

    std::vector<MovementRow> m_rows;
    int m_currentIndex = -1;
    double m_resultTheta = 0.0;

    // Buttons
    QPushButton* m_recordBtn = nullptr;
    QPushButton* m_skipBtn = nullptr;
    QPushButton* m_cancelBtn = nullptr;

    // Result widgets
    QWidget*     m_resultWidget = nullptr;
    QLabel*      m_resultLabel = nullptr;
    QLabel*      m_confidenceLabel = nullptr;
    QPushButton* m_acceptBtn = nullptr;
    QPushButton* m_retryBtn = nullptr;

    QVBoxLayout* m_mainLayout = nullptr;
};
