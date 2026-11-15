#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using DiGiorgioLabs.NeuralNetVR.Core;

namespace DiGiorgioLabs.NeuralNetVR.Editor
{
    public static class DemoVideoCapture
    {
        public static void CaptureDemo()
        {
            string outDir = "/tmp/frames_glassmind";
            if (Directory.Exists(outDir)) Directory.Delete(outDir, true);
            Directory.CreateDirectory(outDir);

            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            var lightGo = new GameObject("KeyLight");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.25f;
            lightGo.transform.rotation = Quaternion.Euler(45f, -30f, 0f);

            var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
            floor.transform.localScale = new Vector3(3f, 1f, 3f);
            var floorMat = NeuralNetVisualFactory.CreateDefaultEmissiveMaterial("FloorMat");
            floor.GetComponent<Renderer>().sharedMaterial = floorMat;
            NeuralNetVisualFactory.SetColorAndEmission(floor.GetComponent<Renderer>(), new Color(0.05f, 0.08f, 0.13f), 0.05f);

            var camGo = new GameObject("CaptureCam");
            var cam = camGo.AddComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.02f, 0.04f, 0.09f);

            var stageRoot = new GameObject("NeuralStageRoot").transform;
            stageRoot.position = new Vector3(0f, 0.85f, 2.2f);
            var ctrlGo = new GameObject("Ctrl");
            var ctrl = ctrlGo.AddComponent<NeuralNetController>();
            var nodeMat = NeuralNetVisualFactory.CreateDefaultEmissiveMaterial("NodeMat");
            var beamMat = NeuralNetVisualFactory.CreateDefaultEmissiveMaterial("BeamMat");

            var visualizers = new INeuralAlgorithmVisualizer[]
            {
                new DiGiorgioLabs.NeuralNetVR.Visualizers.TransformerAttentionVisualizer(),
                new DiGiorgioLabs.NeuralNetVR.Visualizers.BackpropAdamVisualizer(),
                new DiGiorgioLabs.NeuralNetVR.Visualizers.ResNetConvVisualizer(),
                new DiGiorgioLabs.NeuralNetVR.Visualizers.MixtureOfExpertsVisualizer()
            };

            int width = 1280, height = 720;
            var rt = new RenderTexture(width, height, 24);
            var tex = new Texture2D(width, height, TextureFormat.RGB24, false);
            cam.targetTexture = rt;

            int framesPerMode = 48; // 2s @ 24fps per algorithm = 8s total
            int frameIdx = 0;
            Vector3 target = stageRoot.position + new Vector3(0f, 0.65f, 0f);

            for (int m = 0; m < visualizers.Length; m++)
            {
                var vis = visualizers[m];
                vis.Build(stageRoot, nodeMat, beamMat, ctrl);
                for (int f = 0; f < framesPerMode; f++)
                {
                    float t = f / 24f;
                    vis.StepSimulation(t, 1f / 24f, ctrl);

                    float angle = Mathf.Lerp(-0.45f, 0.45f, (float)f / framesPerMode);
                    cam.transform.position = target + new Vector3(Mathf.Sin(angle) * 3.1f, 0.55f, -Mathf.Cos(angle) * 3.1f);
                    cam.transform.LookAt(target);

                    RenderFrame(cam, rt, tex, outDir, frameIdx++);
                }
                vis.Cleanup();
            }

            cam.targetTexture = null;
            Object.DestroyImmediate(rt);
            Object.DestroyImmediate(tex);
            Debug.Log($"[DemoVideoCapture] Captured {frameIdx} frames to {outDir}");
        }

        private static void RenderFrame(Camera cam, RenderTexture rt, Texture2D tex, string outDir, int idx)
        {
            RenderTexture.active = rt;
            cam.Render();
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            File.WriteAllBytes(Path.Combine(outDir, $"frame_{idx:D4}.png"), tex.EncodeToPNG());
            RenderTexture.active = null;
        }
    }
}
#endif
