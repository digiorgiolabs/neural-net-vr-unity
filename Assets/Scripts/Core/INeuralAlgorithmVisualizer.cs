using UnityEngine;

namespace DiGiorgioLabs.NeuralNetVR.Core
{
    public enum AlgorithmMode
    {
        TransformerAttention = 0,
        BackpropAndAdam = 1,
        ResNetSkipAndConv = 2,
        MixtureOfExperts = 3
    }

    public interface INeuralAlgorithmVisualizer
    {
        string Title { get; }
        string Subtitle { get; }
        string FormulaSummary { get; }
        void Build(Transform root, Material nodeMaterial, Material beamMaterial, NeuralNetController controller);
        void StepSimulation(float simTime, float deltaTime, NeuralNetController controller);
        void Cleanup();
    }

    /// <summary>
    /// Attached to interactive 3D nodes (neurons, tokens, experts, kernels) for VR/desktop inspection.
    /// </summary>
    public class NodeMetadata : MonoBehaviour
    {
        public string NodeId;
        public string Role;
        public float Activation;
        public float Gradient;
        public string Notes;

        public string FormatDetails()
        {
            return $"<b>{NodeId}</b> ({Role})\n" +
                   $"Activation: <color=#38BDF8>{Activation:F3}</color> | " +
                   $"Gradient: <color=#F472B6>{Gradient:F4}</color>\n" +
                   $"{Notes}";
        }
    }
}
