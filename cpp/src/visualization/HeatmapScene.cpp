#include "HeatmapScene.h"

#include <filesystem>
#include <algorithm>
#include <iostream>
#include <limits>
#include <cctype>

#include <vtkSTLReader.h>
#include <vtkTriangleFilter.h>
#include <vtkAppendPolyData.h>
#include <vtkPolyDataNormals.h>
#include <vtkPointData.h>
#include <vtkProperty.h>
#include <vtkSphereSource.h>
#include <vtkGlyph3D.h>
#include <vtkPoints.h>
#include <vtkTextProperty.h>

namespace fs = std::filesystem;

// ─── Helpers ───

static Eigen::MatrixXd computeCdist(const Eigen::MatrixXd& A,
                                     const Eigen::MatrixXd& B)
{
    // A: (N x 3), B: (S x 3) → result: (N x S) Euclidean distances
    Eigen::VectorXd sqA = A.rowwise().squaredNorm();
    Eigen::VectorXd sqB = B.rowwise().squaredNorm();
    Eigen::MatrixXd cross = A * B.transpose();
    Eigen::MatrixXd dists2 = sqA.replicate(1, B.rows())
                            + sqB.transpose().replicate(A.rows(), 1)
                            - 2.0 * cross;
    return dists2.cwiseMax(0.0).cwiseSqrt();
}

std::string HeatmapScene::formatMuscleName(const std::string& filename)
{
    // Strip ".r.stl" suffix (last 6 chars), then title-case words (except "of")
    std::string base = filename;
    if (base.size() > 6) base = base.substr(0, base.size() - 6);

    std::string result;
    bool newWord = true;
    for (size_t i = 0; i < base.size(); ++i) {
        char c = base[i];
        if (c == ' ' || c == '_') {
            result += ' ';
            newWord = true;
        } else {
            if (newWord) {
                // Check if this word is "of"
                if (i + 2 <= base.size() &&
                    std::tolower(base[i]) == 'o' &&
                    std::tolower(base[i + 1]) == 'f' &&
                    (i + 2 == base.size() || base[i + 2] == ' '))
                {
                    result += 'o';
                } else {
                    result += static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
                }
                newWord = false;
            } else {
                result += c;
            }
        }
    }
    return result;
}

// ─── Construction ───

HeatmapScene::HeatmapScene(vtkRenderer* renderer, const Params& params)
    : m_renderer(renderer), m_params(params)
{
    buildLookupTable();
}

// ─── Lookup Table (white → yellow → orange → red) ───

void HeatmapScene::buildLookupTable()
{
    m_lut = vtkSmartPointer<vtkLookupTable>::New();
    m_lut->SetNumberOfTableValues(256);
    m_lut->SetRange(m_params.climMin, m_params.climMax);

    // Control points: t → (r, g, b)
    struct CP { double t, r, g, b; };
    std::vector<CP> cps = {
        {0.0,    1.0, 1.0, 1.0},     // white
        {0.333,  1.0, 1.0, 0.0},     // yellow
        {0.667,  1.0, 0.647, 0.0},   // orange
        {1.0,    1.0, 0.0, 0.0},     // red
    };

    for (int i = 0; i < 256; ++i) {
        double t = i / 255.0;
        // Find bounding control points
        int seg = 0;
        for (int s = 0; s < static_cast<int>(cps.size()) - 1; ++s) {
            if (t >= cps[s].t && t <= cps[s + 1].t) { seg = s; break; }
        }
        double frac = (cps[seg + 1].t - cps[seg].t) > 0.0
            ? (t - cps[seg].t) / (cps[seg + 1].t - cps[seg].t)
            : 0.0;
        double r = cps[seg].r + frac * (cps[seg + 1].r - cps[seg].r);
        double g = cps[seg].g + frac * (cps[seg + 1].g - cps[seg].g);
        double b = cps[seg].b + frac * (cps[seg + 1].b - cps[seg].b);
        m_lut->SetTableValue(i, r, g, b, 1.0);
    }
    m_lut->SetNanColor(0.0, 0.0, 0.0, 0.0);  // transparent NaN
    m_lut->Build();
}

// ─── Model loading ───

void HeatmapScene::loadModel(const std::string& modelDir,
                              const std::vector<int>& channels,
                              const Eigen::MatrixXd& sensorPoints)
{
    m_channels = channels;
    m_sensorPoints = sensorPoints;

    auto appendFilter = vtkSmartPointer<vtkAppendPolyData>::New();
    int vertexStart = 0;

    // Collect and sort STL files
    std::vector<fs::path> stlFiles;
    for (auto& entry : fs::directory_iterator(modelDir)) {
        if (entry.is_regular_file()) {
            auto ext = entry.path().extension().string();
            std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
            if (ext == ".stl" || ext == ".obj" || ext == ".ply")
                stlFiles.push_back(entry.path());
        }
    }
    std::sort(stlFiles.begin(), stlFiles.end());

    for (auto& fpath : stlFiles) {
        auto reader = vtkSmartPointer<vtkSTLReader>::New();
        reader->SetFileName(fpath.string().c_str());
        reader->Update();

        auto triFilter = vtkSmartPointer<vtkTriangleFilter>::New();
        triFilter->SetInputConnection(reader->GetOutputPort());
        triFilter->Update();

        vtkPolyData* pd = triFilter->GetOutput();
        int nPts = static_cast<int>(pd->GetNumberOfPoints());

        appendFilter->AddInputData(pd);

        MuscleInfo mi;
        mi.name = formatMuscleName(fpath.filename().string());
        mi.vertexStart = vertexStart;
        mi.vertexEnd = vertexStart + nPts;
        mi.visible = true;
        m_muscles.push_back(mi);

        vertexStart += nPts;
    }

    appendFilter->Update();

    // Generate smooth normals
    auto normals = vtkSmartPointer<vtkPolyDataNormals>::New();
    normals->SetInputConnection(appendFilter->GetOutputPort());
    normals->ComputePointNormalsOn();
    normals->ComputeCellNormalsOff();
    normals->SplittingOff();
    normals->ConsistencyOn();
    normals->SetFeatureAngle(60.0);
    normals->Update();

    m_mesh = normals->GetOutput();
    m_nMeshPoints = static_cast<int>(m_mesh->GetNumberOfPoints());

    // Initialize scalar array with climMin
    m_scalarsArray = vtkSmartPointer<vtkFloatArray>::New();
    m_scalarsArray->SetName("amplitude");
    m_scalarsArray->SetNumberOfTuples(m_nMeshPoints);
    for (int i = 0; i < m_nMeshPoints; ++i)
        m_scalarsArray->SetValue(i, static_cast<float>(m_params.climMin));
    m_mesh->GetPointData()->SetScalars(m_scalarsArray);

    // RBF weight computation
    Eigen::MatrixXd meshPts(m_nMeshPoints, 3);
    for (int i = 0; i < m_nMeshPoints; ++i) {
        double p[3];
        m_mesh->GetPoint(i, p);
        meshPts(i, 0) = p[0];
        meshPts(i, 1) = p[1];
        meshPts(i, 2) = p[2];
    }
    computeRbfWeights(meshPts, sensorPoints);

    // Create mapper + actor
    rebuildMeshActor();

    // Sensor markers and labels
    addSensorMarkers(sensorPoints, channels);

    std::cout << "Loaded " << m_muscles.size() << " muscles, "
              << m_nMeshPoints << " total vertices\n";
}

void HeatmapScene::rebuildMeshActor()
{
    // Remove old actor if exists
    if (m_actor) m_renderer->RemoveActor(m_actor);
    if (m_scalarBar) m_renderer->RemoveActor2D(m_scalarBar);

    m_mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    m_mapper->SetInputData(m_mesh);
    m_mapper->SetScalarModeToUsePointData();
    m_mapper->SetColorModeToMapScalars();
    m_mapper->SetLookupTable(m_lut);
    m_mapper->SetScalarRange(m_params.climMin, m_params.climMax);
    m_mapper->UseLookupTableScalarRangeOn();

    m_actor = vtkSmartPointer<vtkActor>::New();
    m_actor->SetMapper(m_mapper);
    m_actor->GetProperty()->SetInterpolationToPhong();
    m_actor->GetProperty()->SetOpacity(1.0);
    m_renderer->AddActor(m_actor);

    // Scalar bar
    m_scalarBar = vtkSmartPointer<vtkScalarBarActor>::New();
    m_scalarBar->SetLookupTable(m_lut);
    m_scalarBar->SetTitle("");
    m_scalarBar->SetNumberOfLabels(5);
    m_scalarBar->SetOrientationToVertical();
    m_scalarBar->SetPosition(0.88, 0.1);
    m_scalarBar->SetWidth(0.08);
    m_scalarBar->SetHeight(0.8);
    m_renderer->AddActor2D(m_scalarBar);
}

// ─── RBF weight computation ───

void HeatmapScene::computeRbfWeights(const Eigen::MatrixXd& meshPts,
                                      const Eigen::MatrixXd& sensorPts)
{
    int N = static_cast<int>(meshPts.rows());
    int S = static_cast<int>(sensorPts.rows());

    Eigen::MatrixXd dists = computeCdist(meshPts, sensorPts); // (N x S)

    // in_range: vertex within radius of at least one sensor
    Eigen::VectorXd minDists = dists.rowwise().minCoeff();

    // Full weight matrix
    Eigen::MatrixXd weights = (-0.5 * (dists.array() / m_params.sigma).square()).exp().matrix();

    // Hard radius cutoff
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < S; ++j)
            if (dists(i, j) > m_params.radius)
                weights(i, j) = 0.0;

    // Row-normalize
    Eigen::VectorXd rowSums = weights.rowwise().sum();
    for (int i = 0; i < N; ++i)
        if (rowSums(i) > 0.0)
            weights.row(i) /= rowSums(i);

    // Build in-range index list and extract sub-matrix
    m_inRangeIndices.clear();
    for (int i = 0; i < N; ++i)
        if (minDists(i) <= m_params.radius)
            m_inRangeIndices.push_back(i);

    int numInRange = static_cast<int>(m_inRangeIndices.size());
    m_weightsInRange.resize(numInRange, S);
    for (int i = 0; i < numInRange; ++i)
        m_weightsInRange.row(i) = weights.row(m_inRangeIndices[i]);

    std::cout << "RBF weights ready. " << numInRange << " / " << N
              << " vertices in range.\n";
}

// ─── Per-frame scalar update ───

void HeatmapScene::updateScalars(const Eigen::VectorXd& sensorValues)
{
    float* ptr = static_cast<float*>(m_scalarsArray->GetVoidPointer(0));
    float fillVal = static_cast<float>(m_params.climMin);

    // Fill with climMin
    std::fill(ptr, ptr + m_nMeshPoints, fillVal);

    // Matrix-vector multiply for in-range vertices
    Eigen::VectorXd interp = m_weightsInRange * sensorValues;
    int numInRange = static_cast<int>(m_inRangeIndices.size());
    for (int i = 0; i < numInRange; ++i)
        ptr[m_inRangeIndices[i]] = static_cast<float>(interp(i));

    // NaN out hidden muscles
    for (const auto& m : m_muscles) {
        if (!m.visible) {
            for (int i = m.vertexStart; i < m.vertexEnd; ++i)
                ptr[i] = std::numeric_limits<float>::quiet_NaN();
        }
    }

    m_scalarsArray->Modified();
    m_mesh->Modified();
}

// ─── Muscle visibility ───

void HeatmapScene::setMuscleVisible(int idx, bool visible)
{
    if (idx < 0 || idx >= static_cast<int>(m_muscles.size())) return;
    m_muscles[idx].visible = visible;
}

bool HeatmapScene::isMuscleVisible(int idx) const
{
    if (idx < 0 || idx >= static_cast<int>(m_muscles.size())) return false;
    return m_muscles[idx].visible;
}

int HeatmapScene::findMuscleAtVertex(int vertexId) const
{
    for (int i = 0; i < static_cast<int>(m_muscles.size()); ++i) {
        if (vertexId >= m_muscles[i].vertexStart && vertexId < m_muscles[i].vertexEnd)
            return i;
    }
    return -1;
}

void HeatmapScene::updateScalarBarTitle(const std::string& title)
{
    if (m_scalarBar)
        m_scalarBar->SetTitle(title.c_str());
}

// ─── Sensor markers ───

void HeatmapScene::addSensorMarkers(const Eigen::MatrixXd& pts,
                                     const std::vector<int>& channels)
{
    int n = static_cast<int>(pts.rows());

    // Glyph spheres for sensor positions
    auto sensorPoints = vtkSmartPointer<vtkPoints>::New();
    for (int i = 0; i < n; ++i)
        sensorPoints->InsertNextPoint(pts(i, 0), pts(i, 1), pts(i, 2));
    auto sensorPoly = vtkSmartPointer<vtkPolyData>::New();
    sensorPoly->SetPoints(sensorPoints);

    auto sphere = vtkSmartPointer<vtkSphereSource>::New();
    sphere->SetRadius(0.001);
    sphere->SetThetaResolution(12);
    sphere->SetPhiResolution(12);

    auto glyph = vtkSmartPointer<vtkGlyph3D>::New();
    glyph->SetInputData(sensorPoly);
    glyph->SetSourceConnection(sphere->GetOutputPort());
    glyph->Update();

    auto glyphMapper = vtkSmartPointer<vtkPolyDataMapper>::New();
    glyphMapper->SetInputConnection(glyph->GetOutputPort());
    glyphMapper->ScalarVisibilityOff();

    m_sensorGlyphActor = vtkSmartPointer<vtkActor>::New();
    m_sensorGlyphActor->SetMapper(glyphMapper);
    m_sensorGlyphActor->GetProperty()->SetColor(0.0, 0.0, 0.0);
    m_sensorGlyphActor->SetPickable(false);
    m_renderer->AddActor(m_sensorGlyphActor);

    // Billboard text labels — offset along averaged surface normal of nearby vertices
    auto* normals = m_mesh->GetPointData()->GetNormals();

    const double labelOffset = 0.004; // meters outward from surface
    const int kNearest = 20;          // average this many nearby normals
    for (int i = 0; i < n; ++i) {
        Eigen::Vector3d sensorPos(pts(i, 0), pts(i, 1), pts(i, 2));

        // Find K nearest mesh vertices to this sensor
        struct VertDist { int id; double dist2; };
        std::vector<VertDist> verts(m_nMeshPoints);
        for (int v = 0; v < m_nMeshPoints; ++v) {
            double p[3];
            m_mesh->GetPoint(v, p);
            double d2 = (p[0]-sensorPos.x())*(p[0]-sensorPos.x())
                       + (p[1]-sensorPos.y())*(p[1]-sensorPos.y())
                       + (p[2]-sensorPos.z())*(p[2]-sensorPos.z());
            verts[v] = {v, d2};
        }
        std::partial_sort(verts.begin(), verts.begin() + kNearest, verts.end(),
                          [](const VertDist& a, const VertDist& b) { return a.dist2 < b.dist2; });

        // Average the normals of the K nearest vertices for a smooth outward direction
        Eigen::Vector3d outward(0, 0, 1); // fallback
        if (normals) {
            Eigen::Vector3d avgNrm = Eigen::Vector3d::Zero();
            for (int k = 0; k < kNearest; ++k) {
                double nrm[3];
                normals->GetTuple(verts[k].id, nrm);
                avgNrm += Eigen::Vector3d(nrm[0], nrm[1], nrm[2]);
            }
            if (avgNrm.norm() > 1e-9)
                outward = avgNrm.normalized();
        }

        Eigen::Vector3d labelPos = sensorPos + outward * labelOffset;

        auto label = vtkSmartPointer<vtkBillboardTextActor3D>::New();
        std::string text = "CH" + std::to_string(channels[i]);
        label->SetInput(text.c_str());
        label->SetPosition(labelPos.x(), labelPos.y(), labelPos.z());
        label->GetTextProperty()->SetFontSize(14);
        label->GetTextProperty()->SetColor(0.0, 0.0, 0.0);
        label->SetPickable(false);
        m_renderer->AddActor(label);
        m_sensorLabels.push_back(label);
    }
}

void HeatmapScene::setSensorMarkersVisible(bool visible)
{
    m_sensorMarkersVisible = visible;
    if (m_sensorGlyphActor)
        m_sensorGlyphActor->SetVisibility(visible ? 1 : 0);
    for (auto& label : m_sensorLabels)
        label->SetVisibility(visible ? 1 : 0);
}

// ─── Auxiliary part groups ───

void HeatmapScene::loadPartGroup(const std::string& dir, const std::string& name,
                                  double r, double g, double b, double opacity)
{
    std::vector<fs::path> files;
    if (!fs::exists(dir) || !fs::is_directory(dir)) return;

    for (auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        auto ext = entry.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".stl" || ext == ".obj" || ext == ".ply")
            files.push_back(entry.path());
    }
    if (files.empty()) return;

    std::sort(files.begin(), files.end());

    PartGroup pg;
    pg.name = name;
    pg.visible = false;

    for (auto& fpath : files) {
        auto reader = vtkSmartPointer<vtkSTLReader>::New();
        reader->SetFileName(fpath.string().c_str());
        reader->Update();

        auto nrm = vtkSmartPointer<vtkPolyDataNormals>::New();
        nrm->SetInputData(reader->GetOutput());
        nrm->ComputePointNormalsOn();
        nrm->SplittingOff();
        nrm->Update();

        auto mapper = vtkSmartPointer<vtkPolyDataMapper>::New();
        mapper->SetInputData(nrm->GetOutput());
        mapper->ScalarVisibilityOff();

        auto actor = vtkSmartPointer<vtkActor>::New();
        actor->SetMapper(mapper);
        actor->GetProperty()->SetColor(r, g, b);
        actor->GetProperty()->SetOpacity(opacity);
        actor->GetProperty()->SetInterpolationToPhong();
        actor->SetVisibility(false);
        m_renderer->AddActor(actor);
        pg.actors.push_back(actor);
    }

    m_partGroups.push_back(pg);
    std::cout << "Loaded part group '" << name << "' from " << dir
              << " (" << files.size() << " meshes)\n";
}

void HeatmapScene::setPartVisible(const std::string& name, bool visible)
{
    for (auto& pg : m_partGroups) {
        if (pg.name == name) {
            pg.visible = visible;
            for (auto& actor : pg.actors)
                actor->SetVisibility(visible ? 1 : 0);
            return;
        }
    }
}

bool HeatmapScene::isPartVisible(const std::string& name) const
{
    for (const auto& pg : m_partGroups)
        if (pg.name == name) return pg.visible;
    return false;
}

Eigen::Vector3d HeatmapScene::sensorCentroid() const
{
    if (m_sensorPoints.rows() == 0)
        return Eigen::Vector3d::Zero();
    return m_sensorPoints.colwise().mean();
}
