using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CameraController : MonoBehaviour
{
    public float fixedAngle;
    // Start is called before the first frame update
    void Start()
    {
        
    }

    // Update is called once per frame
    void Update()
    {
        Vector3 eulerAngles = transform.eulerAngles;
        transform.eulerAngles = new Vector3(fixedAngle, eulerAngles.y, fixedAngle);
    }
}

