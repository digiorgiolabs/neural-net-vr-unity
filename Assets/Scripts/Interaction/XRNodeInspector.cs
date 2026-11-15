using UnityEngine;
using UnityEngine.EventSystems;
using DiGiorgioLabs.NeuralNetVR.Core;

namespace DiGiorgioLabs.NeuralNetVR.Interaction
{
    /// <summary>
    /// Allows inspecting any 3D neuron, token, kernel, or expert via desktop mouse or VR raycast.
    /// </summary>
    public class XRNodeInspector : MonoBehaviour
    {
        public NodeMetadata SelectedNode { get; private set; }
        public Camera TargetCamera;

        private void Update()
        {
            if (TargetCamera == null) TargetCamera = Camera.main;
            if (TargetCamera == null) return;

            if (Input.GetMouseButtonDown(0))
            {
                if (EventSystem.current != null && EventSystem.current.IsPointerOverGameObject())
                    return;

                Ray ray = TargetCamera.ScreenPointToRay(Input.mousePosition);
                if (Physics.Raycast(ray, out RaycastHit hit, 25f))
                {
                    var meta = hit.collider.GetComponent<NodeMetadata>();
                    if (meta != null)
                    {
                        SelectedNode = meta;
                    }
                }
            }
        }

        public void SelectNode(NodeMetadata node)
        {
            SelectedNode = node;
        }
    }
}
