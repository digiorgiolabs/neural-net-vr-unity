using System.Collections.Generic;
using UnityEngine;
using DiGiorgioLabs.NeuralNetVR.Core;

namespace DiGiorgioLabs.NeuralNetVR.Visualizers
{
    /// <summary>
    /// 3D visualization of Sparse Mixture-of-Experts (MoE) Routing:
    /// Input Tokens -> Gating Router (Top-2 Softmax) -> 4 Specialized Experts -> Weighted Output.
    /// </summary>
    public class MixtureOfExpertsVisualizer : INeuralAlgorithmVisualizer
    {
        public string Title => "Mixture of Experts (Sparse Top-2 MoE)";
        public string Subtitle => "Gating Router Dispatches Each Token to Top-2 Specialized Experts";
        public string FormulaSummary =>
            "y = Σᵢ∈TopK(G(x)) G(x)ᵢ · Eᵢ(x)\n" +
            "Only 2 of 4 Experts activate per token, scaling capacity without full compute.";

        private static readonly string[] ExpertNames = { "Syntax", "Math", "Code", "Memory" };
        private static readonly int[,] Top2ByToken =
        {
            { 0, 3 },
            { 1, 2 },
            { 2, 0 },
            { 3, 1 },
        };

        private readonly List<GameObject> _spawned = new List<GameObject>();
        private readonly List<Renderer> _tokenRends = new List<Renderer>();
        private readonly List<Renderer> _expertRends = new List<Renderer>();
        private readonly List<NodeMetadata> _expertMetas = new List<NodeMetadata>();
        private readonly List<(Transform beam, Renderer rend, int expertIdx)> _routerBeams =
            new List<(Transform, Renderer, int)>();
        private readonly List<(Transform beam, Renderer rend, int expertIdx)> _outBeams =
            new List<(Transform, Renderer, int)>();
        private readonly List<(Transform pulse, Renderer rend, Vector3 start, Vector3 end, int expertIdx)> _pulses =
            new List<(Transform, Renderer, Vector3, Vector3, int)>();

        private Renderer _routerRend;
        private Renderer _outputRend;

        public void Build(Transform root, Material nodeMat, Material beamMat, NeuralNetController controller)
        {
            Cleanup();

            for (int i = 0; i < 4; i++)
            {
                Vector3 pos = new Vector3(-1.45f, 0.25f + i * 0.28f, 0f);
                var tok = NeuralNetVisualFactory.CreateNodeSphere(
                    root, $"MoE_Token_{i}", pos, 0.075f, nodeMat, "Input Token", $"Token x_{i} entering Router");
                _spawned.Add(tok);
                _tokenRends.Add(tok.GetComponent<Renderer>());
            }
            _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, "Tokens", new Vector3(-1.45f, -0.05f, 0f), 0.022f).gameObject);

            Vector3 routerPos = new Vector3(-0.65f, 0.67f, 0f);
            var router = NeuralNetVisualFactory.CreateNodeCube(
                root, "GatingRouter", routerPos, Vector3.one * 0.22f, nodeMat,
                "Gating Router G(x)", "Computes softmax(W_g · x) and selects Top-2 experts.");
            _routerRend = router.GetComponent<Renderer>();
            _spawned.Add(router);
            _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, "Router G(x)", new Vector3(-0.65f, -0.05f, 0f), 0.022f).gameObject);

            for (int i = 0; i < 4; i++)
            {
                Vector3 tPos = new Vector3(-1.45f, 0.25f + i * 0.28f, 0f);
                _spawned.Add(NeuralNetVisualFactory.CreateBeam(root, $"TokToRouter_{i}", tPos, routerPos, 0.008f, beamMat));
            }

            Vector3 outPos = new Vector3(1.35f, 0.67f, 0f);
            var outNode = NeuralNetVisualFactory.CreateNodeSphere(
                root, "MoE_Output", outPos, 0.14f, nodeMat,
                "Weighted Combine", "Combines Top-2 expert outputs weighted by router probabilities.");
            _outputRend = outNode.GetComponent<Renderer>();
            _spawned.Add(outNode);
            _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, "Output Σ", new Vector3(1.35f, -0.05f, 0f), 0.022f).gameObject);

            for (int e = 0; e < 4; e++)
            {
                float y = 0.15f + e * 0.36f;
                float z = (e % 2 == 0) ? -0.22f : 0.22f;
                Vector3 ePos = new Vector3(0.35f, y, z);

                var exp = NeuralNetVisualFactory.CreateNodeCube(
                    root, $"Expert_{ExpertNames[e]}", ePos, new Vector3(0.28f, 0.18f, 0.18f), nodeMat,
                    $"Expert {e}: {ExpertNames[e]}", $"Feed-Forward MLP specialized for {ExpertNames[e]} patterns.");
                _spawned.Add(exp);
                _expertRends.Add(exp.GetComponent<Renderer>());
                _expertMetas.Add(exp.GetComponent<NodeMetadata>());

                _spawned.Add(NeuralNetVisualFactory.CreateWorldLabel(root, ExpertNames[e], ePos + new Vector3(0, 0.16f, 0), 0.018f).gameObject);

                var rBeam = NeuralNetVisualFactory.CreateBeam(root, $"RouteToExp_{e}", routerPos, ePos, 0.012f, beamMat);
                _spawned.Add(rBeam);
                _routerBeams.Add((rBeam.transform, rBeam.GetComponent<Renderer>(), e));

                var oBeam = NeuralNetVisualFactory.CreateBeam(root, $"ExpToOut_{e}", ePos, outPos, 0.012f, beamMat);
                _spawned.Add(oBeam);
                _outBeams.Add((oBeam.transform, oBeam.GetComponent<Renderer>(), e));

                var p1 = CreatePulse(root, $"MoEPulseIn_{e}", nodeMat);
                var p2 = CreatePulse(root, $"MoEPulseOut_{e}", nodeMat);
                _pulses.Add((p1.transform, p1.GetComponent<Renderer>(), routerPos, ePos, e));
                _pulses.Add((p2.transform, p2.GetComponent<Renderer>(), ePos, outPos, e));
            }
        }

        private GameObject CreatePulse(Transform root, string name, Material mat)
        {
            var p = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            p.name = name;
            p.transform.SetParent(root, false);
            p.transform.localScale = Vector3.one * 0.045f;
            var col = p.GetComponent<Collider>();
            if (col != null) Object.Destroy(col);
            p.GetComponent<Renderer>().sharedMaterial = mat;
            _spawned.Add(p);
            return p;
        }

        public void StepSimulation(float simTime, float deltaTime, NeuralNetController controller)
        {
            int activeToken = Mathf.FloorToInt(simTime / 2.2f) % 4;
            int expA = Top2ByToken[activeToken, 0];
            int expB = Top2ByToken[activeToken, 1];

            for (int i = 0; i < 4; i++)
            {
                bool active = i == activeToken;
                NeuralNetVisualFactory.SetColorAndEmission(
                    _tokenRends[i],
                    active ? new Color(0.22f, 0.90f, 1.0f) : new Color(0.22f, 0.32f, 0.45f),
                    active ? 2.0f : 0.3f);
            }

            NeuralNetVisualFactory.SetColorAndEmission(_routerRend, new Color(0.95f, 0.68f, 0.18f), 1.6f);
            NeuralNetVisualFactory.SetColorAndEmission(_outputRend, new Color(0.22f, 0.95f, 0.68f), 1.6f);

            for (int e = 0; e < 4; e++)
            {
                bool isSelected = e == expA || e == expB || controller.AlgorithmToggle;
                float weight = e == expA ? 0.68f : (e == expB ? 0.32f : 0.05f);
                Color col = isSelected ? new Color(0.22f, 0.92f, 0.65f) : new Color(0.20f, 0.24f, 0.32f);
                NeuralNetVisualFactory.SetColorAndEmission(_expertRends[e], col, isSelected ? (0.8f + weight * 1.6f) : 0.15f);
                _expertMetas[e].Activation = isSelected ? weight : 0f;
            }

            UpdateBeams(_routerBeams, expA, expB, controller.AlgorithmToggle);
            UpdateBeams(_outBeams, expA, expB, controller.AlgorithmToggle);

            foreach (var p in _pulses)
            {
                bool active = p.expertIdx == expA || p.expertIdx == expB || controller.AlgorithmToggle;
                if (!active)
                {
                    p.pulse.localScale = Vector3.zero;
                    continue;
                }

                float phase = (simTime * 1.1f + p.expertIdx * 0.2f) % 1f;
                p.pulse.localPosition = Vector3.Lerp(p.start, p.end, phase);
                p.pulse.localScale = Vector3.one * 0.045f;
                NeuralNetVisualFactory.SetColorAndEmission(p.rend, new Color(0.30f, 1.0f, 0.72f), 2.4f);
            }
        }

        private static void UpdateBeams(List<(Transform beam, Renderer rend, int expertIdx)> beams, int expA, int expB, bool denseAll)
        {
            foreach (var b in beams)
            {
                bool active = b.expertIdx == expA || b.expertIdx == expB || denseAll;
                float thickness = active ? (b.expertIdx == expA ? 0.024f : 0.014f) : 0.004f;
                Vector3 s = b.beam.localScale;
                b.beam.localScale = new Vector3(thickness, s.y, thickness);
                Color c = active ? new Color(0.25f, 0.95f, 0.68f) : new Color(0.20f, 0.24f, 0.30f);
                NeuralNetVisualFactory.SetColorAndEmission(b.rend, c, active ? 1.5f : 0.1f);
            }
        }

        public void Cleanup()
        {
            foreach (var go in _spawned)
            {
                if (go != null) Object.Destroy(go);
            }
            _spawned.Clear();
            _tokenRends.Clear();
            _expertRends.Clear();
            _expertMetas.Clear();
            _routerBeams.Clear();
            _outBeams.Clear();
            _pulses.Clear();
        }
    }
}
