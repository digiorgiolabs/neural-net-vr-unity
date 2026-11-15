#if UNITY_EDITOR
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;
using DiGiorgioLabs.NeuralNetVR.Core;

namespace DiGiorgioLabs.NeuralNetVR.Editor
{
    public static class NeuralNetVRSetup
    {
        [MenuItem("Tools/DiGiorgioLabs/1. Build NeuralNet VR Scene")]
        public static void BuildScene()
        {
            Scene scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            var appGo = new GameObject("NeuralNetVR_App");
            appGo.AddComponent<NeuralNetSceneBootstrap>();

            System.IO.Directory.CreateDirectory("Assets/Scenes");
            string path = "Assets/Scenes/NeuralNetVR.unity";
            EditorSceneManager.SaveScene(scene, path);
            EditorBuildSettings.scenes = new[] { new EditorBuildSettingsScene(path, true) };
            Debug.Log($"[DiGiorgioLabs] Saved NeuralNetVR scene to {path}");
        }

        [MenuItem("Tools/DiGiorgioLabs/2. Configure Meta Quest Android Settings")]
        public static void ConfigureQuestAndroid()
        {
            PlayerSettings.SetApplicationIdentifier(BuildTargetGroup.Android, "com.digiorgiolabs.neuralnetvr");
            PlayerSettings.Android.minSdkVersion = AndroidSdkVersions.AndroidApiLevel29;
            PlayerSettings.Android.targetSdkVersion = AndroidSdkVersions.AndroidApiLevel32;
            PlayerSettings.SetScriptingBackend(BuildTargetGroup.Android, ScriptingImplementation.IL2CPP);
            PlayerSettings.Android.targetArchitectures = AndroidArchitecture.ARM64;
            Debug.Log("[DiGiorgioLabs] Configured Android IL2CPP ARM64 for Meta Quest. Enable OpenXR + Meta Quest Support in Project Settings > XR Plug-in Management.");
        }
    }
}
#endif
