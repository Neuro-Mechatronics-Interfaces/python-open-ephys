#include "CalibrationPanel.h"
#include <QFrame>

static QFrame* makeSep(QWidget* parent) {
    auto* line = new QFrame(parent);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    line->setStyleSheet("color: #555;");
    return line;
}

CalibrationPanel::CalibrationPanel(QWidget* parent)
    : QWidget(parent)
{
    m_mainLayout = new QVBoxLayout(this);
    m_mainLayout->setSpacing(6);
    m_mainLayout->setContentsMargins(12, 12, 12, 12);

    auto* title = new QLabel("Calibration Mode", this);
    title->setStyleSheet("font-weight: bold; font-size: 14px;");
    m_mainLayout->addWidget(title);
    m_mainLayout->addWidget(makeSep(this));

    // Movement rows will be added by setMovements()

    // Button row
    auto* btnLayout = new QHBoxLayout();
    m_recordBtn = new QPushButton("Begin Recording", this);
    m_recordBtn->setMinimumHeight(32);
    m_recordBtn->setStyleSheet(
        "QPushButton { background-color: #2196F3; color: white; font-weight: bold; "
        "border-radius: 4px; padding: 4px 12px; }"
        "QPushButton:disabled { background-color: #666; }");
    connect(m_recordBtn, &QPushButton::clicked, this, &CalibrationPanel::recordingRequested);
    btnLayout->addWidget(m_recordBtn);

    m_skipBtn = new QPushButton("Skip", this);
    m_skipBtn->setMinimumHeight(32);
    connect(m_skipBtn, &QPushButton::clicked, this, &CalibrationPanel::movementSkipped);
    btnLayout->addWidget(m_skipBtn);
    m_mainLayout->addLayout(btnLayout);

    m_cancelBtn = new QPushButton("Cancel Calibration", this);
    m_cancelBtn->setMinimumHeight(28);
    m_cancelBtn->setStyleSheet("color: #ff6666;");
    connect(m_cancelBtn, &QPushButton::clicked, this, &CalibrationPanel::calibrationCancelled);
    m_mainLayout->addWidget(m_cancelBtn);

    m_mainLayout->addWidget(makeSep(this));

    // Result area (hidden initially)
    m_resultWidget = new QWidget(this);
    auto* resLayout = new QVBoxLayout(m_resultWidget);
    resLayout->setContentsMargins(0, 0, 0, 0);
    resLayout->setSpacing(4);

    m_resultLabel = new QLabel("", m_resultWidget);
    m_resultLabel->setStyleSheet("font-size: 13px; font-weight: bold;");
    resLayout->addWidget(m_resultLabel);

    m_confidenceLabel = new QLabel("", m_resultWidget);
    m_confidenceLabel->setStyleSheet("font-size: 11px;");
    resLayout->addWidget(m_confidenceLabel);

    auto* resBtnLayout = new QHBoxLayout();
    m_acceptBtn = new QPushButton("Accept && Apply", m_resultWidget);
    m_acceptBtn->setMinimumHeight(32);
    m_acceptBtn->setStyleSheet(
        "QPushButton { background-color: #4CAF50; color: white; font-weight: bold; "
        "border-radius: 4px; padding: 4px 12px; }");
    connect(m_acceptBtn, &QPushButton::clicked, this, [this]() {
        emit calibrationAccepted(m_resultTheta);
    });
    resBtnLayout->addWidget(m_acceptBtn);

    m_retryBtn = new QPushButton("Retry", m_resultWidget);
    m_retryBtn->setMinimumHeight(32);
    connect(m_retryBtn, &QPushButton::clicked, this, &CalibrationPanel::calibrationRetryRequested);
    resBtnLayout->addWidget(m_retryBtn);
    resLayout->addLayout(resBtnLayout);

    m_mainLayout->addWidget(m_resultWidget);
    m_resultWidget->hide();

    m_mainLayout->addStretch();
}

void CalibrationPanel::setMovements(
    const std::vector<std::pair<std::string, std::string>>& movements)
{
    // Clear existing rows
    for (auto& row : m_rows) {
        m_mainLayout->removeWidget(row.container);
        delete row.container;
    }
    m_rows.clear();

    // Insert movement rows before the button row (index 2 = after title + separator)
    int insertIdx = 2;
    for (int i = 0; i < static_cast<int>(movements.size()); ++i) {
        MovementRow row;
        row.container = new QWidget(this);
        auto* hLayout = new QVBoxLayout(row.container);
        hLayout->setContentsMargins(4, 4, 4, 4);
        hLayout->setSpacing(2);

        // Top row: status icon + name
        auto* topRow = new QHBoxLayout();
        row.statusIcon = new QLabel(QString("(%1)").arg(i + 1), row.container);
        row.statusIcon->setFixedWidth(28);
        row.statusIcon->setStyleSheet("font-size: 12px; color: #888;");
        topRow->addWidget(row.statusIcon);

        row.nameLabel = new QLabel(QString::fromStdString(movements[i].first), row.container);
        row.nameLabel->setStyleSheet("font-size: 12px; font-weight: bold; color: #888;");
        topRow->addWidget(row.nameLabel, 1);
        hLayout->addLayout(topRow);

        // Instruction
        row.instructionLabel = new QLabel(
            QString::fromStdString(movements[i].second), row.container);
        row.instructionLabel->setStyleSheet("font-size: 10px; color: #999; padding-left: 32px;");
        row.instructionLabel->setWordWrap(true);
        hLayout->addWidget(row.instructionLabel);

        // Progress bar (hidden until recording)
        row.progressBar = new QProgressBar(row.container);
        row.progressBar->setRange(0, 100);
        row.progressBar->setValue(0);
        row.progressBar->setMaximumHeight(14);
        row.progressBar->setStyleSheet(
            "QProgressBar { border: 1px solid #555; border-radius: 3px; text-align: center; "
            "font-size: 9px; }"
            "QProgressBar::chunk { background-color: #2196F3; }");
        row.progressBar->hide();
        hLayout->addWidget(row.progressBar);

        row.container->setStyleSheet(
            "QWidget { border: 1px solid transparent; border-radius: 4px; }");

        m_mainLayout->insertWidget(insertIdx++, row.container);
        m_rows.push_back(row);
    }
}

void CalibrationPanel::setCurrentMovement(int index)
{
    m_currentIndex = index;
    for (int i = 0; i < static_cast<int>(m_rows.size()); ++i) {
        auto& row = m_rows[i];
        if (i < index) {
            // Completed
            row.statusIcon->setText(QString::fromUtf8("\xe2\x9c\x93"));
            row.statusIcon->setStyleSheet("font-size: 14px; color: #4CAF50;");
            row.nameLabel->setStyleSheet("font-size: 12px; font-weight: bold; color: #4CAF50;");
            row.instructionLabel->setStyleSheet("font-size: 10px; color: #777; padding-left: 32px;");
            row.container->setStyleSheet(
                "QWidget { border: 1px solid transparent; border-radius: 4px; }");
            row.progressBar->hide();
        } else if (i == index) {
            // Current
            row.statusIcon->setText(QString::fromUtf8("\xe2\x97\x8f"));
            row.statusIcon->setStyleSheet("font-size: 14px; color: #2196F3;");
            row.nameLabel->setStyleSheet("font-size: 12px; font-weight: bold; color: #ffffff;");
            row.instructionLabel->setStyleSheet("font-size: 10px; color: #ccc; padding-left: 32px;");
            row.container->setStyleSheet(
                "QWidget { border: 1px solid #2196F3; border-radius: 4px; }");
            row.progressBar->setValue(0);
        } else {
            // Pending
            row.statusIcon->setText(QString("(%1)").arg(i + 1));
            row.statusIcon->setStyleSheet("font-size: 12px; color: #888;");
            row.nameLabel->setStyleSheet("font-size: 12px; font-weight: bold; color: #888;");
            row.instructionLabel->setStyleSheet("font-size: 10px; color: #999; padding-left: 32px;");
            row.container->setStyleSheet(
                "QWidget { border: 1px solid transparent; border-radius: 4px; }");
            row.progressBar->hide();
        }
    }

    bool allDone = (index >= static_cast<int>(m_rows.size()));
    m_recordBtn->setEnabled(!allDone);
    m_skipBtn->setEnabled(!allDone);
}

void CalibrationPanel::setMovementComplete(int index)
{
    if (index < 0 || index >= static_cast<int>(m_rows.size())) return;
    auto& row = m_rows[index];
    row.statusIcon->setText(QString::fromUtf8("\xe2\x9c\x93"));
    row.statusIcon->setStyleSheet("font-size: 14px; color: #4CAF50;");
    row.nameLabel->setStyleSheet("font-size: 12px; font-weight: bold; color: #4CAF50;");
    row.progressBar->hide();
}

void CalibrationPanel::setRecording(bool recording)
{
    if (m_currentIndex < 0 || m_currentIndex >= static_cast<int>(m_rows.size())) return;
    auto& row = m_rows[m_currentIndex];

    if (recording) {
        row.progressBar->setValue(0);
        row.progressBar->show();
        m_recordBtn->setEnabled(false);
        m_recordBtn->setText("Recording...");
        m_skipBtn->setEnabled(false);
    } else {
        row.progressBar->hide();
        m_recordBtn->setEnabled(true);
        m_recordBtn->setText("Begin Recording");
        m_skipBtn->setEnabled(true);
    }
}

void CalibrationPanel::setProgress(double fraction)
{
    if (m_currentIndex < 0 || m_currentIndex >= static_cast<int>(m_rows.size())) return;
    m_rows[m_currentIndex].progressBar->setValue(static_cast<int>(fraction * 100));
}

void CalibrationPanel::showResult(double thetaDeg, double confidence)
{
    m_resultTheta = thetaDeg;
    m_resultLabel->setText(QString("Result: \xce\xb8 = %1%2")
        .arg(thetaDeg >= 0 ? "+" : "").arg(thetaDeg, 0, 'f', 0) + QString::fromUtf8("\xc2\xb0"));
    m_confidenceLabel->setText(QString("Confidence: %1%").arg(confidence, 0, 'f', 0));

    m_recordBtn->setEnabled(false);
    m_skipBtn->setEnabled(false);
    m_resultWidget->show();
}

void CalibrationPanel::hideResult()
{
    m_resultWidget->hide();
}
