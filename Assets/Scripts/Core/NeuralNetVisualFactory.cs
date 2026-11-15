using UnityEngine;

namespace DiGiorgioLabs.NeuralNetVR.Core
{
    /// <summary>
    /// Creates lightweight 3D primitives and updates colors via MaterialPropertyBlock
    /// to avoid per-frame material instancing on Meta Quest mobile GPUs.
    /// </summary>
    public static class NeuralNetVisualFactory
    {
        private static readonly int BaseColorId = Shader.PropertyToID("_BaseColor");
        private static readonly int ColorId = Shader.PropertyToID("_Color");
        private static readonly int EmissionColorId = Shader.PropertyToID("_EmissionColor");

        private static readonly MaterialPropertyBlock SharedBlock = new MaterialPropertyBlock();

        public static Material CreateDefaultEmissiveMaterial(string name)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit")
                            ?? Shader.Find("Standard")
                            ?? Shader.Find("Unlit/Color");

            var mat = new Material(shader) { name = name };
            mat.EnableKeyword("_EMISSION");
            mat.SetFloat("_Smoothness", 0.65f);
            mat.SetFloat("_Metallic", 0.15f);
            return mat;
        }

        public static GameObject CreateNodeSphere(
            Transform parent,
            string name,
            Vector3 localPos,
            float radius,
            Material mat,
            string role,
            string notes)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = Vector3.one * (radius * 2f);

            var renderer = go.GetComponent<Renderer>();
            renderer.sharedMaterial = mat;

            var meta = go.AddComponent<NodeMetadata>();
            meta.NodeId = name;
            meta.Role = role;
            meta.Notes = notes;

            return go;
        }

        public static GameObject CreateNodeCube(
            Transform parent,
            string name,
            Vector3 localPos,
            Vector3 scale,
            Material mat,
            string role,
            string notes)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
            go.name = name;
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;
            go.transform.localScale = scale;

            var renderer = go.GetComponent<Renderer>();
            renderer.sharedMaterial = mat;

            var meta = go.AddComponent<NodeMetadata>();
            meta.NodeId = name;
            meta.Role = role;
            meta.Notes = notes;

            return go;
        }

        public static GameObject CreateBeam(
            Transform parent,
            string name,
            Vector3 startLocal,
            Vector3 endLocal,
            float thickness,
            Material mat)
        {
            var go = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            go.name = name;
            go.transform.SetParent(parent, false);

            var col = go.GetComponent<Collider>();
            if (col != null) Object.Destroy(col);

            UpdateBeamTransform(go.transform, startLocal, endLocal, thickness);

            var renderer = go.GetComponent<Renderer>();
            renderer.sharedMaterial = mat;
            return go;
        }

        public static void UpdateBeamTransform(Transform beam, Vector3 startLocal, Vector3 endLocal, float thickness)
        {
            Vector3 delta = endLocal - startLocal;
            float dist = delta.magnitude;
            if (dist < 0.0001f) dist = 0.0001f;

            beam.localPosition = (startLocal + endLocal) * 0.5f;
            beam.localRotation = Quaternion.FromToRotation(Vector3.up, delta.normalized);
            beam.localScale = new Vector3(thickness, dist * 0.5f, thickness);
        }

        public static TextMesh CreateWorldLabel(
            Transform parent,
            string text,
            Vector3 localPos,
            float characterSize = 0.025f,
            int fontSize = 36,
            Color? color = null)
        {
            var go = new GameObject($"Label_{text}");
            go.transform.SetParent(parent, false);
            go.transform.localPosition = localPos;

            var tm = go.AddComponent<TextMesh>();
            tm.text = text;
            tm.characterSize = characterSize;
            tm.fontSize = fontSize;
            tm.anchor = TextAnchor.MiddleCenter;
            tm.alignment = TextAlignment.Center;
            tm.color = color ?? new Color(0.88f, 0.94f, 1f, 1f);
            return tm;
        }

        public static void SetColorAndEmission(Renderer renderer, Color baseColor, float emissionIntensity)
        {
            if (renderer == null) return;
            renderer.GetPropertyBlock(SharedBlock);
            SharedBlock.SetColor(BaseColorId, baseColor);
            SharedBlock.SetColor(ColorId, baseColor);
            SharedBlock.SetColor(EmissionColorId, baseColor * emissionIntensity);
            renderer.SetPropertyBlock(SharedBlock);
        }
    }
}
