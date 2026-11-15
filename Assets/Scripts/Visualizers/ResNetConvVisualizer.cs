using System.Collections.Generic;
using UnityEngine;
using DiGiorgioLabs.NeuralNetVR.Core;

namespace DiGiorgioLabs.NeuralNetVR.Visualizers
{
    /// <summary>
    /// 3D visualization of Convolutional Feature Maps + Sliding 3x3 Kernel + Residual Skip Connection F(x) + x.
    /// </summary>
    public class ResNetConvVisualizer : INeuralAlgorithmVisualizer
    {
        public string Title => "ResNet Residual Block & 3D Convolution";
        public string Subtitle => "Sliding 3×3 Kernel, Multi-Channel Feature Maps & Skip Connection F(x)+x";
        public string FormulaSummary =>
            "y = ReLU(F(x, {Wᵢ}) + x)\n" +
            "Toggle ON = Residual Skip Active | Toggle OFF = Plain Conv (no skip).";

        private readonly List<GameObject> _spawned = new List<GameObject>();
        private readonly Renderer[,] _inputGrid = new Renderer[5, 5];
        private readonly Renderer[,,] _conv1Grid = new Renderer[3, 3, 3];
        private readonly Renderer[,,] _conv2Grid = new Renderer[3, 3, 3];
        private readonly List<(Transform pulse, Renderer rend, Vector3[] path)> _skipPulses =
            new List<(Transform, Renderer, Vector3[])>();
        private readonly List<GameObject> _skipBeams = new List<GameObject>();

        private Transform _kernelCube;
        private Renderer _kernelRenderer;

        public void Build(Transform root, Material nodeMat, Material beamMat, NeuralNetController controller)
        {
            Cleanup();

            Vector3 inputOrigin = new Vector3(-1.35f, 0.65f, 0f);
            BuildGrid(root, "Input", inputOrigin, 5, 1, 0.14f, nodeMat, _inputGrid, null);
            _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, "Input x (5×5)", new Vector3(-1.35f, -0.05f, 0f), 0.022f).gameObject);

            var kObj = NeuralNetVisualFactory.CreateNodeCube(
                root, "ConvKernel_3x3", inputOrigin, new Vector3(0.42f, 0.42f, 0.16f), nodeMat,
                "3×3 Conv Kernel", "Slides across input x computing dot products W * x + b.");
            _kernelCube = kObj.transform;
            _kernelRenderer = kObj.GetComponent<Renderer>();
            NeuralNetVisualFactory.SetColorAndEmission(_kernelRenderer, new Color(1.0f, 0.72f, 0.15f, 0.45f), 1.3f);
            _spawned.Add(kObj);

            Vector3 conv1Origin = new Vector3(-0.25f, 0.65f, 0f);
            BuildGrid(root, "Conv1", conv1Origin, 3, 3, 0.16f, nodeMat, null, _conv1Grid);
            _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, "Conv1 (3×3×3)", new Vector3(-0.25f, -0.05f, 0f), 0.022f).gameObject);

            Vector3 conv2Origin = new Vector3(1.10f, 0.65f, 0f);
            BuildGrid(root, "F(x)+x", conv2Origin, 3, 3, 0.16f, nodeMat, null, _conv2Grid);
            _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, "Output F(x) + x", new Vector3(1.10f, -0.05f, 0f), 0.022f).gameObject);

            _spawned.Add(NeuralNetVisualFactory.CreateBeam(root, "ConvLink1", inputOrigin, conv1Origin, 0.015f, beamMat));
            _spawned.Add(NeuralNetVisualFactory.CreateBeam(root, "ConvLink2", conv1Origin, conv2Origin, 0.015f, beamMat));

            Vector3[] skipPath =
            {
                inputOrigin + new Vector3(0, 0.38f, 0),
                new Vector3(-0.80f, 1.48f, 0),
                new Vector3(0.42f, 1.48f, 0),
                conv2Origin + new Vector3(0, 0.35f, 0)
            };

            for (int i = 0; i < skipPath.Length - 1; i++)
            {
                var seg = NeuralNetVisualFactory.CreateBeam(root, $"SkipArc_{i}", skipPath[i], skipPath[i + 1], 0.022f, beamMat);
                NeuralNetVisualFactory.SetColorAndEmission(seg.GetComponent<Renderer>(), new Color(0.18f, 0.92f, 0.85f), 1.6f);
                _spawned.Add(seg);
                _skipBeams.Add(seg);
            }

            _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, "Identity Skip (+ x)", new Vector3(-0.18f, 1.65f, 0), 0.022f, 36, new Color(0.3f, 0.98f, 0.88f)).gameObject);

            for (int p = 0; p < 3; p++)
            {
                var pulse = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                pulse.name = $"SkipPulse_{p}";
                pulse.transform.SetParent(root, false);
                pulse.transform.localScale = Vector3.one * 0.065f;
                var col = pulse.GetComponent<Collider>();
                if (col != null) Object.Destroy(col);
                var pr = pulse.GetComponent<Renderer>();
                pr.sharedMaterial = nodeMat;
                _spawned.Add(pulse);
                _skipPulses.Add((pulse.transform, pr, skipPath));
            }
        }

        private void BuildGrid(
            Transform root,
            string prefix,
            Vector3 center,
            int size,
            int channels,
            float cellSpacing,
            Material mat,
            Renderer[,] grid2D,
            Renderer[,,] grid3D)
        {
            for (int c = 0; c < channels; c++)
            {
                float z = (c - (channels - 1) * 0.5f) * 0.28f;
                for (int r = 0; r < size; r++)
                {
                    for (int col = 0; col < size; col++)
                    {
                        float x = center.x + (col - (size - 1) * 0.5f) * cellSpacing;
                        float y = center.y + ((size - 1) * 0.5f - r) * cellSpacing;
                        var cell = NeuralNetVisualFactory.CreateNodeCube(
                            root, $"{prefix}_c{c}_{r}_{col}", new Vector3(x, y, z), Vector3.one * (cellSpacing * 0.72f),
                            mat, $"{prefix} Feature Cell", $"Channel {c}, Row {r}, Col {col}");
                        _spawned.Add(cell);

                        var rend = cell.GetComponent<Renderer>();
                        if (grid2D != null) grid2D[r, col] = rend;
                        if (grid3D != null) grid3D[c, r, col] = rend;
                    }
                }
            }
        }

        public void StepSimulation(float simTime, float deltaTime, NeuralNetController controller)
        {
            int step = Mathf.FloorToInt(simTime * 1.4f) % 9;
            int kRow = step / 3;
            int kCol = step % 3;

            Vector3 inputOrigin = new Vector3(-1.35f, 0.65f, 0f);
            float spacing = 0.14f;
            float kx = inputOrigin.x + (kCol - 1) * spacing;
            float ky = inputOrigin.y + (1 - kRow) * spacing;
            _kernelCube.localPosition = Vector3.Lerp(_kernelCube.localPosition, new Vector3(kx, ky, 0f), Mathf.Clamp01(deltaTime * 10f));

            for (int r = 0; r < 5; r++)
            {
                for (int c = 0; c < 5; c++)
                {
                    bool inKernel = r >= kRow && r < kRow + 3 && c >= kCol && c < kCol + 3;
                    Color col = inKernel ? new Color(1.0f, 0.78f, 0.20f) : new Color(0.22f, 0.34f, 0.50f);
                    NeuralNetVisualFactory.SetColorAndEmission(_inputGrid[r, c], col, inKernel ? 1.7f : 0.3f);
                }
            }

            bool skipActive = controller.AlgorithmToggle;
            for (int ch = 0; ch < 3; ch++)
            {
                Color chColor = ch == 0 ? new Color(0.22f, 0.78f, 1.0f) : (ch == 1 ? new Color(0.65f, 0.45f, 1.0f) : new Color(0.22f, 0.92f, 0.68f));
                for (int r = 0; r < 3; r++)
                {
                    for (int c = 0; c < 3; c++)
                    {
                        bool activeCell = (r == kRow && c == kCol);
                        NeuralNetVisualFactory.SetColorAndEmission(_conv1Grid[ch, r, c], chColor, activeCell ? 2.1f : 0.45f);
                        float outBoost = skipActive ? (activeCell ? 2.4f : 0.8f) : (activeCell ? 1.5f : 0.25f);
                        NeuralNetVisualFactory.SetColorAndEmission(_conv2Grid[ch, r, c], chColor, outBoost);
                    }
                }
            }

            foreach (var beam in _skipBeams)
            {
                beam.SetActive(skipActive);
            }

            for (int i = 0; i < _skipPulses.Count; i++)
            {
                var p = _skipPulses[i];
                if (!skipActive)
                {
                    p.pulse.localScale = Vector3.zero;
                    continue;
                }

                float t = ((simTime * 0.55f) + i * 0.33f) % 1f;
                p.pulse.localPosition = SamplePolyline(p.path, t);
                p.pulse.localScale = Vector3.one * 0.065f;
                NeuralNetVisualFactory.SetColorAndEmission(p.rend, new Color(0.25f, 1.0f, 0.88f), 2.5f);
            }
        }

        private static Vector3 SamplePolyline(Vector3[] pts, float t)
        {
            float scaled = t * (pts.Length - 1);
            int idx = Mathf.Clamp(Mathf.FloorToInt(scaled), 0, pts.Length - 2);
            return Vector3.Lerp(pts[idx], pts[idx + 1], scaled - idx);
        }

        public void Cleanup()
        {
            foreach (var go in _spawned)
            {
                if (go != null) Object.Destroy(go);
            }
            _spawned.Clear();
            _skipPulses.Clear();
            _skipBeams.Clear();
        }
    }
}
