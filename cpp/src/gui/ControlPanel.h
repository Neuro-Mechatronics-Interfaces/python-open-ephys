#pragma once
#include <QWidget>
#include <QPushButton>
#include <QVBoxLayout>
#include <QScrollArea>
#include <QMenu>
#include <QLabel>
#include <QSpinBox>
#include <vector>
#include <string>

class ControlPanel : public QWidget {
    Q_OBJECT
public:
    explicit ControlPanel(int initialWindowMs = 200, QWidget* parent = nullptr);

    void setMuscleNames(const std::vector<std::string>& names);
    void setMuscleChecked(int idx, bool checked);
    void setActiveVizFunction(const QString& name);
    bool isLocked() const { return m_locked; }

signals:
    void muscleToggled(int idx, bool visible);
    void vizFunctionChanged(const QString& name);
    void partToggled(const QString& name, bool visible);
    void sensorMarkersToggled(bool visible);
    void windowMsChanged(int ms);

private:
    QPushButton*  m_vizFuncBtn = nullptr;
    QMenu*        m_vizFuncMenu = nullptr;
    QLabel*       m_activeVizLabel = nullptr;
    QPushButton*  m_partsBtn = nullptr;
    QMenu*        m_partsMenu = nullptr;
    QPushButton*  m_sensorsBtn = nullptr;
    QSpinBox*     m_windowMsSpin = nullptr;
    QPushButton*  m_lockBtn = nullptr;
    QPushButton*  m_selectAllBtn = nullptr;
    bool          m_locked = false;
    QScrollArea*  m_scrollArea = nullptr;
    QVBoxLayout*  m_scrollLayout = nullptr;
    std::vector<QPushButton*> m_muscleButtons;
};
