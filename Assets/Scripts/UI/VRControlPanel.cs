using UnityEngine;
using UnityEngine.UI;
using DiGiorgioLabs.NeuralNetVR.Core;
using DiGiorgioLabs.NeuralNetVR.Interaction;

namespace DiGiorgioLabs.NeuralNetVR.UI
{
    /// <summary>
    /// World-Space VR Control Panel for switching neural network algorithms,
    /// adjusting Temperature/Speed, and reading live node metrics.
    /// </summary>
    public class VRControlPanel : MonoBehaviour
    {
        public NeuralNetController Controller;
        public XRNodeInspector Inspector;

        private Text _titleText;
        private Text _subtitleText;
        private Text _formulaText;
        private Text _statusText;
        private Text _inspectorText;
        private Text _toggleBtnText;
        private Text _playBtnText;

        private void Start()
        {
            BuildWorldSpaceUI();
            RefreshLabels();
        }

        private void Update()
        {
            RefreshLabels();
        }

        private void BuildWorldSpaceUI()
        {
            var canvasGo = new GameObject("VR_ControlPanel_Canvas");
            canvasGo.transform.SetParent(transform, false);
            canvasGo.transform.localPosition = new Vector3(-1.35f, 1.35f, 1.55f);
            canvasGo.transform.localRotation = Quaternion.Euler(0f, 28f, 0f);
            canvasGo.transform.localScale = Vector3.one * 0.0014f;

            var canvas = canvasGo.AddComponent<Canvas>();
            canvas.renderMode = RenderMode.WorldSpace;
            var rect = canvasGo.GetComponent<RectTransform>();
            rect.sizeDelta = new Vector2(820f, 680f);

            canvasGo.AddComponent<CanvasScaler>();
            canvasGo.AddComponent<GraphicRaycaster>();

            // Background panel
            var bg = CreatePanel(canvasGo.transform, "Background", Vector2.zero, new Vector2(820f, 680f), new Color(0.04f, 0.07f, 0.14f, 0.92f));

            _titleText = CreateText(bg.transform, "Title", new Vector2(0, 295), new Vector2(760, 46), 28, FontStyle.Bold, new Color(0.22f, 0.80f, 1f));
            _subtitleText = CreateText(bg.transform, "Subtitle", new Vector2(0, 255), new Vector2(760, 34), 18, FontStyle.Normal, new Color(0.75f, 0.84f, 0.95f));
            _formulaText = CreateText(bg.transform, "Formula", new Vector2(0, 190), new Vector2(760, 76), 17, FontStyle.Normal, new Color(0.65f, 0.95f, 0.80f));

            // Mode buttons row
            CreateButton(bg.transform, "BtnTransformer", "1. Transformer\nAttention", new Vector2(-285, 95), new Vector2(175, 68),
                () => Controller.SetMode(AlgorithmMode.TransformerAttention));
            CreateButton(bg.transform, "BtnBackprop", "2. Backprop\n& Adam", new Vector2(-95, 95), new Vector2(175, 68),
                () => Controller.SetMode(AlgorithmMode.BackpropAndAdam));
            CreateButton(bg.transform, "BtnResNet", "3. ResNet\n& 3D Conv", new Vector2(95, 95), new Vector2(175, 68),
                () => Controller.SetMode(AlgorithmMode.ResNetSkipAndConv));
            CreateButton(bg.transform, "BtnMoE", "4. Mixture\nof Experts", new Vector2(285, 95), new Vector2(175, 68),
                () => Controller.SetMode(AlgorithmMode.MixtureOfExperts));

            // Transport & Feature buttons
            var playBtn = CreateButton(bg.transform, "BtnPlayPause", "Pause", new Vector2(-250, 5), new Vector2(160, 52),
                () => Controller.TogglePlayPause());
            _playBtnText = playBtn.GetComponentInChildren<Text>();

            CreateButton(bg.transform, "BtnStep", "Step +0.25s", new Vector2(-75, 5), new Vector2(160, 52),
                () => Controller.StepOnce());

            var togBtn = CreateButton(bg.transform, "BtnToggle", "Feature: ON", new Vector2(140, 5), new Vector2(230, 52),
                () => Controller.ToggleAlgorithmFeature());
            _toggleBtnText = togBtn.GetComponentInChildren<Text>();

            // Sliders
            CreateText(bg.transform, "SpeedLbl", new Vector2(-270, -70), new Vector2(200, 30), 18, FontStyle.Bold, Color.white).text = "Simulation Speed";
            var speedSlider = CreateSlider(bg.transform, "SpeedSlider", new Vector2(90, -70), new Vector2(440, 32), 0.1f, 3.0f, Controller.SimulationSpeed);
            speedSlider.onValueChanged.AddListener(v => Controller.SimulationSpeed = v);

            CreateText(bg.transform, "TempLbl", new Vector2(-270, -125), new Vector2(200, 30), 18, FontStyle.Bold, Color.white).text = "Temp / LR Scale";
            var tempSlider = CreateSlider(bg.transform, "TempSlider", new Vector2(90, -125), new Vector2(440, 32), 0.15f, 2.5f, Controller.Temperature);
            tempSlider.onValueChanged.AddListener(v => Controller.Temperature = v);

            _statusText = CreateText(bg.transform, "Status", new Vector2(0, -180), new Vector2(760, 34), 17, FontStyle.Normal, new Color(0.95f, 0.80f, 0.35f));

            // Inspector box
            var inspPanel = CreatePanel(bg.transform, "InspectorBox", new Vector2(0, -265), new Vector2(760, 115), new Color(0.08f, 0.12f, 0.22f, 0.95f));
            _inspectorText = CreateText(inspPanel.transform, "InspectorText", Vector2.zero, new Vector2(730, 95), 17, FontStyle.Normal, Color.white);
            _inspectorText.alignment = TextAnchor.UpperLeft;
        }

        private void RefreshLabels()
        {
            if (Controller == null || _titleText == null) return;

            var vis = Controller.ActiveVisualizer;
            if (vis != null)
            {
                _titleText.text = vis.Title;
                _subtitleText.text = vis.Subtitle;
                _formulaText.text = vis.FormulaSummary;
            }

            _playBtnText.text = Controller.IsPlaying ? "Pause" : "Play";
            _toggleBtnText.text = $"Mode Toggle: {(Controller.AlgorithmToggle ? "ON" : "OFF")}";
            _statusText.text = $"Speed: {Controller.SimulationSpeed:F2}x  |  Temp/LR: {Controller.Temperature:F2}  |  VR/Mouse Click Any 3D Node";

            if (Inspector != null && Inspector.SelectedNode != null)
            {
                _inspectorText.text = Inspector.SelectedNode.FormatDetails();
            }
            else
            {
                _inspectorText.text = "Node Inspector: Point & trigger (VR) or click (Desktop) on any 3D sphere/cube to inspect live activations and gradients.";
            }
        }

        private static GameObject CreatePanel(Transform parent, string name, Vector2 pos, Vector2 size, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var img = go.AddComponent<Image>();
            img.color = color;
            var rt = go.GetComponent<RectTransform>();
            rt.anchoredPosition = pos;
            rt.sizeDelta = size;
            return go;
        }

        private static Text CreateText(Transform parent, string name, Vector2 pos, Vector2 size, int fontSize, FontStyle style, Color color)
        {
            var go = new GameObject(name);
            go.transform.SetParent(parent, false);
            var t = go.AddComponent<Text>();
            t.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf") ?? Resources.GetBuiltinResource<Font>("Arial.ttf");
            t.fontSize = fontSize;
            t.fontStyle = style;
            t.color = color;
            t.alignment = TextAnchor.MiddleCenter;
            t.supportRichText = true;
            var rt = go.GetComponent<RectTransform>();
            rt.anchoredPosition = pos;
            rt.sizeDelta = size;
            return t;
        }

        private static Button CreateButton(Transform parent, string name, string label, Vector2 pos, Vector2 size, UnityEngine.Events.UnityAction onClick)
        {
            var go = CreatePanel(parent, name, pos, size, new Color(0.14f, 0.25f, 0.45f, 1f));
            var btn = go.AddComponent<Button>();
            btn.onClick.AddListener(onClick);
            var txt = CreateText(go.transform, "Label", Vector2.zero, size - new Vector2(10, 8), 18, FontStyle.Bold, Color.white);
            txt.text = label;
            return btn;
        }

        private static Slider CreateSlider(Transform parent, string name, Vector2 pos, Vector2 size, float min, float max, float val)
        {
            var go = CreatePanel(parent, name, pos, size, new Color(0.12f, 0.18f, 0.30f, 1f));
            var slider = go.AddComponent<Slider>();
            slider.minValue = min;
            slider.maxValue = max;
            slider.value = val;

            var fillArea = new GameObject("Fill Area");
            fillArea.transform.SetParent(go.transform, false);
            var fillRt = fillArea.AddComponent<RectTransform>();
            fillRt.anchorMin = Vector2.zero;
            fillRt.anchorMax = Vector2.one;
            fillRt.offsetMin = new Vector2(8, 6);
            fillRt.offsetMax = new Vector2(-8, -6);

            var fill = CreatePanel(fillArea.transform, "Fill", Vector2.zero, Vector2.zero, new Color(0.22f, 0.78f, 1f, 1f));
            slider.fillRect = fill.GetComponent<RectTransform>();

            var handle = CreatePanel(go.transform, "Handle", Vector2.zero, new Vector2(24, 36), Color.white);
            slider.handleRect = handle.GetComponent<RectTransform>();
            slider.targetGraphic = handle.GetComponent<Image>();
            return slider;
        }
    }
}
