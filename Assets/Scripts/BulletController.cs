using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class BulletController : MonoBehaviour
{
    public float Speed; // KPH
    public float DistanceLimit; // m

    private float scale;

    public bool hit;
    public int planeHit;

    // Start is called before the first frame update
    void Start() 
    {
    }

    public void init(float scale, Vector3 position, Vector3 direction)
    {
        this.scale = scale;
        
        transform.rotation *= Quaternion.FromToRotation(transform.right, direction);
        transform.position = position;
        transform.Translate(new Vector3(6, 0, 0));
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        transform.Translate(new Vector3(KPHtoMPS(Speed) * scale * Time.fixedDeltaTime, 0, 0));
        if (Vector3.Magnitude(transform.position) > DistanceLimit)
        {
            Destroy(gameObject);
        }
    }

    float KPHtoMPS(float speed) { return speed / 3.6f; }
}
