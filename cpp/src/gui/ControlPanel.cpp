#include "ControlPanel.h"
#include <QAction>
#include <QHBoxLayout>
#include <QFrame>

static QFrame* makeSeparator(QWidget* parent) {
    auto* line = new QFrame(parent);
    line->setFrameShape(QFrame::HLine);
    line->setFrameShadow(QFrame::Sunken);
    line->setStyleSheet("color: #555;");
    return line;
}

ControlPanel::ControlPanel(int initialWindowMs, QWidget* parent)
    : QWidget(parent)
{
    auto* layout = new QVBoxLayout(this);
    layout->setSpacing(8);
    layout->setContentsMargins(10, 10, 10, 10);

    // Viz function dropdown button
    m_vizFuncBtn = new QPushButton("Set Visualization Function", this);
    m_vizFuncBtn->setMinimumHeight(32);
    m_vizFuncMenu = new QMenu(m_vizFuncBtn);

    auto* rmsAction = m_vizFuncMenu->addAction("RMS Amplitude");
    connect(rmsAction, &QAction::triggered, this, [this]() {
        emit vizFunctionChanged("RMS Amplitude");
    });

    auto* spikeAction = m_vizFuncMenu->addAction("Absolute Spike Amplitude");
    connect(spikeAction, &QAction::triggered, this, [this]() {
        emit vizFunctionChanged("Abs Spike Amplitude");
    });

    m_vizFuncBtn->setMenu(m_vizFuncMenu);
    layout->addWidget(m_vizFuncBtn);

    // Active viz function label
    m_activeVizLabel = new QLabel("Active: RMS Amplitude (mV)", this);
    m_activeVizLabel->setStyleSheet("color: #ffffff; font-size: 11px; padding-left: 4px;");
    layout->addWidget(m_activeVizLabel);

    // Window size spinner
    auto* windowRow = new QHBoxLayout();
    windowRow->setContentsMargins(4, 0, 0, 0);
    auto* windowLabel = new QLabel("Window:", this);
    windowLabel->setStyleSheet("font-size: 11px;");
    windowRow->addWidget(windowLabel);
    m_windowMsSpin = new QSpinBox(this);
    m_windowMsSpin->setRange(10, 5000);
    m_windowMsSpin->setSingleStep(50);
    m_windowMsSpin->setSuffix(" ms");
    m_windowMsSpin->setValue(initialWindowMs);
    m_windowMsSpin->setMinimumHeight(26);
    connect(m_windowMsSpin, QOverload<int>::of(&QSpinBox::valueChanged),
            this, &ControlPanel::windowMsChanged);
    windowRow->addWidget(m_windowMsSpin);
    layout->addLayout(windowRow);

    layout->addWidget(makeSeparator(this));

    // Muscles label + lock button row
    auto* labelRow = new QHBoxLayout();
    auto* musclesLabel = new QLabel("Selected Muscles", this);
    musclesLabel->setStyleSheet("font-weight: bold; font-size: 12px;");
    labelRow->addWidget(musclesLabel);

    m_selectAllBtn = new QPushButton(QString::fromUtf8("\xE2\x98\x91"), this);
    m_selectAllBtn->setFixedSize(28, 28);
    m_selectAllBtn->setToolTip("Select / Deselect all muscles");
    m_selectAllBtn->setStyleSheet(
        "QPushButton { border: 1px solid #888; border-radius: 4px; font-size: 14px; }");
    connect(m_selectAllBtn, &QPushButton::clicked, this, [this]() {
        if (m_locked) return;
        bool anyUnchecked = false;
        for (auto* btn : m_muscleButtons)
            if (!btn->isChecked()) { anyUnchecked = true; break; }
        for (int i = 0; i < static_cast<int>(m_muscleButtons.size()); ++i) {
            m_muscleButtons[i]->setChecked(anyUnchecked);
            emit muscleToggled(i, anyUnchecked);
        }
    });
    labelRow->addWidget(m_selectAllBtn);

    m_lockBtn = new QPushButton(QString::fromUtf8("\xF0\x9F\x94\x93"), this);
    m_lockBtn->setFixedSize(28, 28);
    m_lockBtn->setToolTip("Lock muscle selections");
    m_lockBtn->setCheckable(true);
    m_lockBtn->setStyleSheet(
        "QPushButton { border: 1px solid #888; border-radius: 4px; font-size: 14px; }"
        "QPushButton:checked { background-color: #ff6666; border-color: #cc0000; }");
    connect(m_lockBtn, &QPushButton::toggled, this, [this](bool checked) {
        m_locked = checked;
        m_lockBtn->setText(checked ? QString::fromUtf8("\xF0\x9F\x94\x92") : QString::fromUtf8("\xF0\x9F\x94\x93"));
        m_lockBtn->setToolTip(checked ? "Unlock muscle selections" : "Lock muscle selections");
        for (auto* btn : m_muscleButtons)
            btn->setEnabled(!checked);
    });
    labelRow->addWidget(m_lockBtn);
    layout->addLayout(labelRow);

    // Scroll area for muscle toggle buttons — capped height
    m_scrollArea = new QScrollArea(this);
    m_scrollArea->setWidgetResizable(true);
    m_scrollArea->setMaximumHeight(350);
    auto* scrollWidget = new QWidget();
    m_scrollLayout = new QVBoxLayout(scrollWidget);
    m_scrollLayout->setContentsMargins(0, 0, 0, 0);
    m_scrollLayout->setSpacing(3);
    m_scrollLayout->setAlignment(Qt::AlignTop);
    m_scrollArea->setWidget(scrollWidget);
    layout->addWidget(m_scrollArea);

    layout->addWidget(makeSeparator(this));

    // Toggle Parts dropdown with checkable items
    m_partsBtn = new QPushButton("Toggle Parts", this);
    m_partsBtn->setMinimumHeight(32);
    m_partsMenu = new QMenu(m_partsBtn);

    auto addPartAction = [this](const QString& name) {
        auto* action = m_partsMenu->addAction(name);
        action->setCheckable(true);
        action->setChecked(false);
        connect(action, &QAction::triggered, this, [this, action, name](bool) {
            emit partToggled(name, action->isChecked());
        });
    };
    addPartAction("Bones");
    addPartAction("Ligaments");
    addPartAction("Membranes");
    addPartAction("Arteries");
    addPartAction("Cartilages");

    m_partsBtn->setMenu(m_partsMenu);
    layout->addWidget(m_partsBtn);

    // Toggle Sensors button (standalone checkable button)
    m_sensorsBtn = new QPushButton("Toggle Sensors", this);
    m_sensorsBtn->setMinimumHeight(32);
    m_sensorsBtn->setCheckable(true);
    m_sensorsBtn->setChecked(false);
    connect(m_sensorsBtn, &QPushButton::toggled, this, [this](bool checked) {
        emit sensorMarkersToggled(checked);
    });
    layout->addWidget(m_sensorsBtn);

    layout->addStretch();
}

void ControlPanel::setMuscleNames(const std::vector<std::string>& names)
{
    for (auto* btn : m_muscleButtons) {
        m_scrollLayout->removeWidget(btn);
        delete btn;
    }
    m_muscleButtons.clear();

    for (int i = 0; i < static_cast<int>(names.size()); ++i) {
        auto* btn = new QPushButton(QString::fromStdString(names[i]));
        btn->setCheckable(true);
        btn->setChecked(true);
        btn->setMinimumHeight(28);
        connect(btn, &QPushButton::clicked, this, [this, i](bool checked) {
            emit muscleToggled(i, checked);
        });
        m_scrollLayout->addWidget(btn);
        m_muscleButtons.push_back(btn);
    }
}

void ControlPanel::setMuscleChecked(int idx, bool checked)
{
    if (idx >= 0 && idx < static_cast<int>(m_muscleButtons.size()))
        m_muscleButtons[idx]->setChecked(checked);
}

void ControlPanel::setActiveVizFunction(const QString& name)
{
    m_activeVizLabel->setText("Active: " + name + " (mV)");
}
