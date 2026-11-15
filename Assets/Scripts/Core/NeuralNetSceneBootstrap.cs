using UnityEngine;
using UnityEngine.EventSystems;
using DiGiorgioLabs.NeuralNetVR.Interaction;
using DiGiorgioLabs.NeuralNetVR.UI;

namespace DiGiorgioLabs.NeuralNetVR.Core
{
    /// <summary>
    /// Self-bootstrapping VR/Desktop lab environment for NeuralNetVR.
    /// Automatically creates lighting, lab floor, camera/XR rig, EventSystem, and UI if not already present.
    /// </summary>
    public class NeuralNetSceneBootstrap : MonoBehaviour
    {
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void AutoInit()
        {
            if (FindObjectOfType<NeuralNetController>() != null) return;
            var root = new GameObject("NeuralNetVR_App");
            root.AddComponent<NeuralNetSceneBootstrap>();
        }

        private Camera _cam;
        private float _yaw;
        private float _pitch = 8f;

        private void Awake()
        {
            SetupEnvironment();
            SetupCameraAndEventSystem();

            var controller = gameObject.AddComponent<NeuralNetController>();
            var inspector = gameObject.AddComponent<XRNodeInspector>();
            inspector.TargetCamera = _cam;

            var panel = gameObject.AddComponent<VRControlPanel>();
            panel.Controller = controller;
            panel.Inspector = inspector;
        }

        private void SetupEnvironment()
        {
            RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
            RenderSettings.ambientLight = new Color(0.14f, 0.18f, 0.26f);

            if (FindObjectOfType<Light>() == null)
            {
                var lightGo = new GameObject("KeyLight");
                var light = lightGo.AddComponent<Light>();
                light.type = LightType.Directional;
                light.intensity = 1.15f;
                light.color = new Color(0.92f, 0.96f, 1.0f);
                lightGo.transform.rotation = Quaternion.Euler(45f, -30f, 0f);
            }

            // Lab Floor
            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.name = "VR_Lab_Floor";
            floor.transform.position = Vector3.zero;
            floor.transform.localScale = new Vector3(3f, 1f, 3f);
            var floorMat = NeuralNetVisualFactory.CreateDefaultEmissiveMaterial("LabFloorMat");
            floor.GetComponent<Renderer>().sharedMaterial = floorMat;
            NeuralNetVisualFactory.SetColorAndEmission(floor.GetComponent<Renderer>(), new Color(0.05f, 0.08f, 0.13f), 0.05f);
        }

        private void SetupCameraAndEventSystem()
        {
            _cam = Camera.main;
            if (_cam == null)
            {
                var camGo = new GameObject("Main Camera");
                camGo.tag = "MainCamera";
                _cam = camGo.AddComponent<Camera>();
                camGo.AddComponent<AudioListener>();
                camGo.transform.position = new Vector3(0f, 1.55f, -0.8f);
            }
            _cam.clearFlags = CameraClearFlags.SolidColor;
            _cam.backgroundColor = new Color(0.02f, 0.04f, 0.09f);

            if (FindObjectOfType<EventSystem>() == null)
            {
                var es = new GameObject("EventSystem");
                es.AddComponent<EventSystem>();
                es.AddComponent<StandaloneInputModule>();
            }
        }

        private void Update()
        {
            // Desktop WASD + Right-Mouse-Drag navigation fallback when testing outside Quest headset
            if (_cam == null) return;

            if (Input.GetMouseButton(1))
            {
                _yaw += Input.GetAxis("Mouse X") * 2.5f;
                _pitch = Mathf.Clamp(_pitch - Input.GetAxis("Mouse Y") * 2.5f, -65f, 65f);
                _cam.transform.rotation = Quaternion.Euler(_pitch, _yaw, 0f);
            }

            float speed = 2.2f * Time.deltaTime;
            Vector3 move = Vector3.zero;
            if (Input.GetKey(KeyCode.W)) move += _cam.transform.forward;
            if (Input.GetKey(KeyCode.S)) move -= _cam.transform.forward;
            if (Input.GetKey(KeyCode.D)) move += _cam.transform.right;
            if (Input.GetKey(KeyCode.A)) move -= _cam.transform.right;
            if (Input.GetKey(KeyCode.E)) move += Vector3.up;
            if (Input.GetKey(KeyCode.Q)) move -= Vector3.up;
            _cam.transform.position += move * speed;
        }
    }
}
