using System.Collections.Generic;
using UnityEngine;
using DiGiorgioLabs.NeuralNetVR.Core;

namespace DiGiorgioLabs.NeuralNetVR.Visualizers
{
    /// <summary>
    /// 3D visualization of Multi-Head Self-Attention:
    /// Token Embeddings -> Q / K / V projections -> Softmax(Q*K^T / sqrt(d_k)*T) -> Context Output.
    /// </summary>
    public class TransformerAttentionVisualizer : INeuralAlgorithmVisualizer
    {
        public string Title => "Transformer Self-Attention";
        public string Subtitle => "Multi-Head Q/K/V Projections & Softmax(Q·Kᵀ / √dₖ · T)";
        public string FormulaSummary =>
            "Attention(Q,K,V) = softmax((Q·Kᵀ) / (√dₖ · Temperature)) · V\n" +
            "Lower Temperature sharpens focus; Toggle switches Active Query Token.";

        private static readonly string[] Tokens = { "The", "cat", "sat", "on", "mat" };

        // Base semantic affinity logits for Head 1 (Syntactic) and Head 2 (Positional/Object)
        private static readonly float[,] Head1Logits =
        {
            { 1.2f, 2.8f, 0.4f, 0.2f, 0.3f }, // "The" -> "cat"
            { 1.8f, 2.2f, 2.6f, 0.3f, 0.5f }, // "cat" -> "sat", "The"
            { 0.3f, 2.9f, 1.5f, 1.8f, 1.4f }, // "sat" -> "cat", "on"
            { 0.2f, 0.5f, 2.1f, 1.0f, 2.9f }, // "on"  -> "mat", "sat"
            { 0.4f, 0.8f, 1.3f, 2.7f, 1.8f }, // "mat" -> "on", "sat"
        };

        private static readonly float[,] Head2Logits =
        {
            { 2.5f, 1.5f, 0.5f, 0.2f, 0.1f },
            { 1.4f, 2.4f, 1.6f, 0.5f, 0.2f },
            { 0.4f, 1.7f, 2.5f, 1.7f, 0.8f },
            { 0.2f, 0.4f, 1.5f, 2.4f, 1.9f },
            { 0.1f, 0.9f, 1.1f, 1.6f, 2.8f },
        };

        private readonly List<GameObject> _spawned = new List<GameObject>();
        private readonly List<Renderer> _tokenRenderers = new List<Renderer>();
        private readonly List<NodeMetadata> _tokenMetas = new List<NodeMetadata>();
        private readonly List<Renderer> _outputRenderers = new List<Renderer>();
        private readonly List<NodeMetadata> _outputMetas = new List<NodeMetadata>();
        private readonly List<(Transform beam, Renderer rend, int head, int qIdx, int kIdx)> _beams =
            new List<(Transform, Renderer, int, int, int)>();
        private readonly List<(Transform pulse, Renderer rend, Vector3 start, Vector3 end, int head, int qIdx, int kIdx)> _pulses =
            new List<(Transform, Renderer, Vector3, Vector3, int, int, int)>();

        private readonly float[] _weightsBuffer = new float[5];

        public void Build(Transform root, Material nodeMat, Material beamMat, NeuralNetController controller)
        {
            Cleanup();
            int n = Tokens.Length;
            float spacing = 0.72f;
            float xOffset = -(n - 1) * spacing * 0.5f;

            var tokenPositions = new Vector3[n];
            var qPositions = new Vector3[n];
            var vPositions = new Vector3[n];
            var outPositions = new Vector3[n];

            for (int i = 0; i < n; i++)
            {
                float x = xOffset + i * spacing;
                tokenPositions[i] = new Vector3(x, 0.0f, 0.0f);
                qPositions[i] = new Vector3(x, 0.48f, -0.18f);
                vPositions[i] = new Vector3(x, 0.48f, 0.28f);
                outPositions[i] = new Vector3(x, 1.25f, 0.05f);

                // Token Embedding sphere
                var tok = NeuralNetVisualFactory.CreateNodeSphere(
                    root, $"Token_{Tokens[i]}", tokenPositions[i], 0.11f, nodeMat,
                    "Token Embedding", $"Embedding vector x_{i} for \"{Tokens[i]}\"");
                _spawned.Add(tok);
                _tokenRenderers.Add(tok.GetComponent<Renderer>());
                _tokenMetas.Add(tok.GetComponent<NodeMetadata>());

                var lbl = NeuralNetVisualFactory.CreateWorldLabel(root, Tokens[i], tokenPositions[i] + new Vector3(0, -0.2f, 0), 0.024f);
                _spawned.Add(lbl.gameObject);

                // Query & Value projection nodes
                var qNode = NeuralNetVisualFactory.CreateNodeCube(
                    root, $"Q_{Tokens[i]}", qPositions[i], Vector3.one * 0.09f, nodeMat,
                    "Query (W_Q · x)", $"Projects \"{Tokens[i]}\" into Query space to ask what it attends to.");
                NeuralNetVisualFactory.SetColorAndEmission(qNode.GetComponent<Renderer>(), new Color(0.22f, 0.74f, 0.97f), 1.1f);
                _spawned.Add(qNode);

                var vNode = NeuralNetVisualFactory.CreateNodeCube(
                    root, $"V_{Tokens[i]}", vPositions[i], Vector3.one * 0.09f, nodeMat,
                    "Value (W_V · x)", $"Carries semantic content from \"{Tokens[i]}\" to weighted sum.");
                NeuralNetVisualFactory.SetColorAndEmission(vNode.GetComponent<Renderer>(), new Color(0.65f, 0.45f, 0.98f), 1.1f);
                _spawned.Add(vNode);

                // Projection stalks
                _spawned.Add(NeuralNetVisualFactory.CreateBeam(root, $"StalkQ_{i}", tokenPositions[i], qPositions[i], 0.012f, beamMat));
                _spawned.Add(NeuralNetVisualFactory.CreateBeam(root, $"StalkV_{i}", tokenPositions[i], vPositions[i], 0.012f, beamMat));

                // Context Output sphere
                var outNode = NeuralNetVisualFactory.CreateNodeSphere(
                    root, $"Context_{Tokens[i]}", outPositions[i], 0.105f, nodeMat,
                    "Contextualized Output", $"z_{i} = Σ α_ij · V_j across Attention Heads");
                _spawned.Add(outNode);
                _outputRenderers.Add(outNode.GetComponent<Renderer>());
                _outputMetas.Add(outNode.GetComponent<NodeMetadata>());
            }

            // Attention beams (Head 0: Cyan/Blue, Head 1: Amber/Gold)
            for (int head = 0; head < 2; head++)
            {
                float zShift = head == 0 ? -0.05f : 0.35f;
                for (int q = 0; q < n; q++)
                {
                    for (int k = 0; k < n; k++)
                    {
                        Vector3 start = (head == 0 ? qPositions[q] : vPositions[q]) + new Vector3(0, 0.02f, zShift * 0.2f);
                        Vector3 end = outPositions[k];
                        var beam = NeuralNetVisualFactory.CreateBeam(root, $"Attn_H{head}_{q}_{k}", start, end, 0.01f, beamMat);
                        _spawned.Add(beam);
                        _beams.Add((beam.transform, beam.GetComponent<Renderer>(), head, q, k));

                        // Add traveling pulse sphere for top connections
                        var pulse = GameObject.CreatePrimitive(PrimitiveType.Sphere);
                        pulse.name = $"Pulse_H{head}_{q}_{k}";
                        pulse.transform.SetParent(root, false);
                        pulse.transform.localScale = Vector3.one * 0.035f;
                        var col = pulse.GetComponent<Collider>();
                        if (col != null) Object.Destroy(col);
                        var pr = pulse.GetComponent<Renderer>();
                        pr.sharedMaterial = nodeMat;
                        _spawned.Add(pulse);
                        _pulses.Add((pulse.transform, pr, start, end, head, q, k));
                    }
                }
            }

            var header = NeuralNetVisualFactory.CreateWorldLabel(root, "Self-Attention: Q · Kᵀ → Softmax → V", new Vector3(0, 1.56f, 0), 0.024f);
            _spawned.Add(header.gameObject);
        }

        public void StepSimulation(float simTime, float deltaTime, NeuralNetController controller)
        {
            int n = Tokens.Length;
            float temp = Mathf.Max(0.15f, controller.Temperature);

            // Cycle active query token every 3.5s (or lock to Head 1/2 emphasis via toggle)
            int activeQuery = Mathf.FloorToInt(simTime / 3.5f) % n;

            for (int i = 0; i < n; i++)
            {
                bool isFocused = i == activeQuery;
                Color tokCol = isFocused ? new Color(0.22f, 0.85f, 0.98f) : new Color(0.25f, 0.38f, 0.55f);
                NeuralNetVisualFactory.SetColorAndEmission(_tokenRenderers[i], tokCol, isFocused ? 1.8f : 0.4f);
                _tokenMetas[i].Activation = isFocused ? 1.0f : 0.35f;

                Color outCol = isFocused ? new Color(0.18f, 0.92f, 0.68f) : new Color(0.32f, 0.48f, 0.68f);
                NeuralNetVisualFactory.SetColorAndEmission(_outputRenderers[i], outCol, isFocused ? 1.6f : 0.5f);
            }

            // Compute softmax weights for activeQuery across both heads
            float[] head0Weights = ComputeSoftmax(Head1Logits, activeQuery, temp);
            float[] head1Weights = ComputeSoftmax(Head2Logits, activeQuery, temp);

            for (int i = 0; i < n; i++)
            {
                _outputMetas[i].Activation = (head0Weights[i] + head1Weights[i]) * 0.5f;
            }

            foreach (var b in _beams)
            {
                bool active = b.qIdx == activeQuery && (!controller.AlgorithmToggle || b.head == 0);
                float w = b.head == 0 ? head0Weights[b.kIdx] : head1Weights[b.kIdx];
                float thickness = active ? Mathf.Lerp(0.004f, 0.032f, w) : 0.002f;

                Vector3 s = b.beam.localScale;
                b.beam.localScale = new Vector3(thickness, s.y, thickness);

                Color c = b.head == 0 ? new Color(0.20f, 0.78f, 1.0f) : new Color(0.98f, 0.68f, 0.18f);
                float intensity = active ? Mathf.Lerp(0.2f, 2.4f, w) : 0.05f;
                NeuralNetVisualFactory.SetColorAndEmission(b.rend, c * (active ? 1f : 0.25f), intensity);
            }

            foreach (var p in _pulses)
            {
                bool active = p.qIdx == activeQuery && (!controller.AlgorithmToggle || p.head == 0);
                float w = p.head == 0 ? head0Weights[p.kIdx] : head1Weights[p.kIdx];

                if (!active || w < 0.08f)
                {
                    p.pulse.localScale = Vector3.zero;
                    continue;
                }

                float phase = (simTime * 0.9f + p.kIdx * 0.17f + p.head * 0.33f) % 1f;
                p.pulse.localPosition = Vector3.Lerp(p.start, p.end, phase);
                p.pulse.localScale = Vector3.one * Mathf.Lerp(0.025f, 0.075f, w);

                Color c = p.head == 0 ? new Color(0.35f, 0.90f, 1.0f) : new Color(1.0f, 0.78f, 0.25f);
                NeuralNetVisualFactory.SetColorAndEmission(p.rend, c, 2.5f);
            }
        }

        private float[] ComputeSoftmax(float[,] logits, int row, float temp)
        {
            int n = Tokens.Length;
            float maxVal = float.NegativeInfinity;
            for (int j = 0; j < n; j++)
            {
                float v = logits[row, j] / temp;
                if (v > maxVal) maxVal = v;
                _weightsBuffer[j] = v;
            }

            float sum = 0f;
            var result = new float[n];
            for (int j = 0; j < n; j++)
            {
                result[j] = Mathf.Exp(_weightsBuffer[j] - maxVal);
                sum += result[j];
            }
            for (int j = 0; j < n; j++) result[j] /= Mathf.Max(1e-6f, sum);
            return result;
        }

        public void Cleanup()
        {
            foreach (var go in _spawned)
            {
                if (go != null) Object.Destroy(go);
            }
            _spawned.Clear();
            _tokenRenderers.Clear();
            _tokenMetas.Clear();
            _outputRenderers.Clear();
            _outputMetas.Clear();
            _beams.Clear();
            _pulses.Clear();
        }
    }
}
