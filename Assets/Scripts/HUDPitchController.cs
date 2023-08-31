using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HUDPitchController : MonoBehaviour
{
    public GameObject Plane;

    private Transform planeTransform;

    // Start is called before the first frame update
    void Start()
    {
        planeTransform = Plane.GetComponent<Transform>();
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
