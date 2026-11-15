using System.Collections.Generic;
using UnityEngine;
using DiGiorgioLabs.NeuralNetVR.Visualizers;

namespace DiGiorgioLabs.NeuralNetVR.Core
{
    /// <summary>
    /// Orchestrates the active 3D Neural Network visualizer, simulation clock, and parameters.
    /// </summary>
    public class NeuralNetController : MonoBehaviour
    {
        [Header("Simulation Controls")]
        public AlgorithmMode CurrentMode = AlgorithmMode.TransformerAttention;
        public bool IsPlaying = true;
        [Range(0.1f, 3.0f)] public float SimulationSpeed = 1.0f;
        [Range(0.15f, 2.5f)] public float Temperature = 0.8f;
        public bool AlgorithmToggle = true;

        private readonly Dictionary<AlgorithmMode, INeuralAlgorithmVisualizer> _visualizers =
            new Dictionary<AlgorithmMode, INeuralAlgorithmVisualizer>();

        private Transform _stageRoot;
        private Material _nodeMat;
        private Material _beamMat;
        private float _simTime;

        public INeuralAlgorithmVisualizer ActiveVisualizer =>
            _visualizers.TryGetValue(CurrentMode, out var vis) ? vis : null;

        private void Awake()
        {
            _stageRoot = new GameObject("NeuralStageRoot").transform;
            _stageRoot.SetParent(transform, false);
            _stageRoot.localPosition = new Vector3(0f, 0.85f, 2.2f);

            _nodeMat = NeuralNetVisualFactory.CreateDefaultEmissiveMaterial("NeuralNodeMat");
            _beamMat = NeuralNetVisualFactory.CreateDefaultEmissiveMaterial("NeuralBeamMat");

            _visualizers[AlgorithmMode.TransformerAttention] = new TransformerAttentionVisualizer();
            _visualizers[AlgorithmMode.BackpropAndAdam] = new BackpropAdamVisualizer();
            _visualizers[AlgorithmMode.ResNetSkipAndConv] = new ResNetConvVisualizer();
            _visualizers[AlgorithmMode.MixtureOfExperts] = new MixtureOfExpertsVisualizer();
        }

        private void Start()
        {
            SetMode(CurrentMode);
        }

        private void Update()
        {
            if (IsPlaying)
            {
                _simTime += Time.deltaTime * SimulationSpeed;
            }

            ActiveVisualizer?.StepSimulation(_simTime, Time.deltaTime, this);
        }

        public void SetMode(AlgorithmMode mode)
        {
            ActiveVisualizer?.Cleanup();
            CurrentMode = mode;
            _simTime = 0f;
            ActiveVisualizer?.Build(_stageRoot, _nodeMat, _beamMat, this);
            ActiveVisualizer?.StepSimulation(0f, 0.016f, this);
        }

        public void StepOnce(float stepSeconds = 0.25f)
        {
            IsPlaying = false;
            _simTime += stepSeconds;
            ActiveVisualizer?.StepSimulation(_simTime, stepSeconds, this);
        }

        public void TogglePlayPause() => IsPlaying = !IsPlaying;
        public void ToggleAlgorithmFeature() => AlgorithmToggle = !AlgorithmToggle;
    }
}
