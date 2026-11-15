using System.Collections.Generic;
using UnityEngine;
using DiGiorgioLabs.NeuralNetVR.Core;

namespace DiGiorgioLabs.NeuralNetVR.Visualizers
{
    /// <summary>
    /// 3D visualization of Forward Activation -> Loss -> Backpropagation & Adam Optimizer
    /// vs. Vanishing Gradients (toggleable via AlgorithmToggle).
    /// </summary>
    public class BackpropAdamVisualizer : INeuralAlgorithmVisualizer
    {
        public string Title => "Backpropagation & Adam Optimizer";
        public string Subtitle => "Forward Activations (Cyan) vs. Backward Gradients (Magenta)";
        public string FormulaSummary =>
            "θₜ = θₜ₋₁ - α · m̂ₜ / (√v̂ₜ + ε)\n" +
            "Toggle ON = Adam Adaptive Gradients | Toggle OFF = Vanishing Sigmoid Gradients.";

        private static readonly int[] LayerSizes = { 4, 6, 6, 3 };
        private readonly List<GameObject> _spawned = new List<GameObject>();
        private readonly List<List<Renderer>> _layerRenderers = new List<List<Renderer>>();
        private readonly List<List<NodeMetadata>> _layerMetas = new List<List<NodeMetadata>>();
        private readonly List<List<Vector3>> _layerPositions = new List<List<Vector3>>();
        private readonly List<(Transform beam, Renderer rend, int srcLayer, Vector3 a, Vector3 b)> _edges =
            new List<(Transform, Renderer, int, Vector3, Vector3)>();
        private readonly List<(Transform pulse, Renderer rend, int srcLayer, Vector3 a, Vector3 b, int idx)> _pulses =
            new List<(Transform, Renderer, int, Vector3, Vector3, int)>();

        public void Build(Transform root, Material nodeMat, Material beamMat, NeuralNetController controller)
        {
            Cleanup();
            float layerSpacingX = 0.95f;
            float startX = -(LayerSizes.Length - 1) * layerSpacingX * 0.5f;

            for (int l = 0; l < LayerSizes.Length; l++)
            {
                int count = LayerSizes[l];
                var rList = new List<Renderer>();
                var mList = new List<NodeMetadata>();
                var pList = new List<Vector3>();

                float x = startX + l * layerSpacingX;
                for (int i = 0; i < count; i++)
                {
                    float y = 0.65f + (i - (count - 1) * 0.5f) * 0.25f;
                    float z = ((i % 2) == 0 ? -0.12f : 0.12f);
                    Vector3 pos = new Vector3(x, y, z);

                    string role = l == 0 ? "Input Feature" : (l == LayerSizes.Length - 1 ? "Output / Loss" : $"Hidden Layer {l}");
                    var sphere = NeuralNetVisualFactory.CreateNodeSphere(
                        root, $"L{l}_N{i}", pos, 0.075f, nodeMat, role,
                        "Stores forward activation a_i and backward error δ_i = ∂L/∂z_i.");
                    _spawned.Add(sphere);
                    rList.Add(sphere.GetComponent<Renderer>());
                    mList.Add(sphere.GetComponent<NodeMetadata>());
                    pList.Add(pos);
                }

                string label = l == 0 ? "Input" : (l == LayerSizes.Length - 1 ? "Loss (MSE)" : $"Hidden {l}");
                _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, label, new Vector3(x, -0.15f, 0), 0.022f).gameObject);

                _layerRenderers.Add(rList);
                _layerMetas.Add(mList);
                _layerPositions.Add(pList);
            }

            int edgeCounter = 0;
            for (int l = 0; l < LayerSizes.Length - 1; l++)
            {
                for (int i = 0; i < _layerPositions[l].Count; i++)
                {
                    for (int j = 0; j < _layerPositions[l + 1].Count; j++)
                    {
                        Vector3 a = _layerPositions[l][i];
                        Vector3 b = _layerPositions[l + 1][j];
                        var beam = NeuralNetVisualFactory.CreateBeam(root, $"W_{l}_{i}_{j}", a, b, 0.008f, beamMat);
                        _spawned.Add(beam);
                        _edges.Add((beam.transform, beam.GetComponent<Renderer>(), l, a, b));

                        // Create pulse on every 2nd edge to keep mobile VR draw calls lean
                        if ((edgeCounter++ % 2) == 0)
                        {
                            var pulse = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                            pulse.name = $"GradPulse_{edgeCounter}";
                            pulse.transform.SetParent(root, false);
                            pulse.transform.localScale = Vector3.one * 0.03f;
                            var col = pulse.GetComponent<Collider>();
                            if (col != null) Object.Destroy(col);
                            var pr = pulse.GetComponent<Renderer>();
                            pr.sharedMaterial = nodeMat;
                            _spawned.Add(pulse);
                            _pulses.Add((pulse.transform, pr, l, a, b, edgeCounter));
                        }
                    }
                }
            }
        }

        public void StepSimulation(float simTime, float deltaTime, NeuralNetController controller)
        {
            float cycle = (simTime * 0.35f) % 1f;
            bool isForward = cycle < 0.5f;
            float phase = isForward ? (cycle / 0.5f) : ((cycle - 0.5f) / 0.5f);
            bool useAdam = controller.AlgorithmToggle;
            float lrBoost = Mathf.Clamp(controller.Temperature, 0.3f, 2.5f);

            int layerCount = _layerRenderers.Count;
            for (int l = 0; l < layerCount; l++)
            {
                float normLayer = (float)l / (layerCount - 1);
                float dist = isForward ? Mathf.Abs(phase - normLayer) : Mathf.Abs((1f - phase) - normLayer);
                float wave = Mathf.Exp(-dist * dist * 16f);

                // Vanishing gradient factor when Adam is OFF
                float gradStrength = useAdam
                    ? Mathf.Lerp(0.75f, 1.0f, normLayer)
                    : Mathf.Pow(0.32f, (layerCount - 1) - l);

                for (int i = 0; i < _layerRenderers[l].Count; i++)
                {
                    float val = isForward
                        ? 0.3f + 0.7f * wave * (0.6f + 0.4f * Mathf.Sin(simTime * 2f + i))
                        : wave * gradStrength * lrBoost;

                    Color col = isForward
                        ? Color.Lerp(new Color(0.15f, 0.30f, 0.48f), new Color(0.15f, 0.88f, 1.0f), wave)
                        : Color.Lerp(new Color(0.35f, 0.15f, 0.32f), new Color(1.0f, 0.28f, 0.65f), wave * gradStrength);

                    NeuralNetVisualFactory.SetColorAndEmission(_layerRenderers[l][i], col, 0.3f + val * 1.8f);

                    if (isForward) _layerMetas[l][i].Activation = val;
                    else _layerMetas[l][i].Gradient = val;
                }
            }

            foreach (var e in _edges)
            {
                float normLayer = (float)e.srcLayer / (layerCount - 1);
                float gradStrength = useAdam ? 0.9f : Mathf.Pow(0.32f, (layerCount - 2) - e.srcLayer);
                Color col = isForward
                    ? new Color(0.15f, 0.62f, 0.85f)
                    : Color.Lerp(new Color(0.25f, 0.15f, 0.25f), new Color(1.0f, 0.32f, 0.62f), gradStrength);

                float thickness = isForward ? 0.008f : Mathf.Lerp(0.003f, 0.016f, gradStrength);
                Vector3 s = e.beam.localScale;
                e.beam.localScale = new Vector3(thickness, s.y, thickness);
                NeuralNetVisualFactory.SetColorAndEmission(e.rend, col, isForward ? 0.4f : gradStrength * 1.1f);
            }

            foreach (var p in _pulses)
            {
                float t = isForward ? phase : (1f - phase);
                float layerStart = (float)p.srcLayer / (layerCount - 1);
                float layerEnd = (float)(p.srcLayer + 1) / (layerCount - 1);
                float localT = Mathf.InverseLerp(isForward ? layerStart : layerEnd, isForward ? layerEnd : layerStart, t);

                if (localT <= 0.01f || localT >= 0.99f)
                {
                    p.pulse.localScale = Vector3.zero;
                    continue;
                }

                float gradStrength = useAdam ? 1.0f : Mathf.Pow(0.35f, (layerCount - 2) - p.srcLayer);
                Vector3 pos = isForward ? Vector3.Lerp(p.a, p.b, localT) : Vector3.Lerp(p.b, p.a, localT);
                p.pulse.localPosition = pos;
                p.pulse.localScale = Vector3.one * (isForward ? 0.036f : Mathf.Lerp(0.014f, 0.045f, gradStrength));

                Color col = isForward ? new Color(0.25f, 0.95f, 1.0f) : new Color(1.0f, 0.30f, 0.68f);
                NeuralNetVisualFactory.SetColorAndEmission(p.rend, col, 2.2f);
            }
        }

        public void Cleanup()
        {
            foreach (var go in _spawned)
            {
                if (go != null) Object.Destroy(go);
            }
            _spawned.Clear();
            _layerRenderers.Clear();
            _layerMetas.Clear();
            _layerPositions.Clear();
            _edges.Clear();
            _pulses.Clear();
        }
    }
}
