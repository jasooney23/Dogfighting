using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class CubeScript : MonoBehaviour
{
    public GameObject Obj;
    public int Cubes;
    // Start is called before the first frame update
    void Start()
    {
        for (int j = 0; j < Cubes; j++)
        {
            for (int i = 0; i < Cubes; i++)
            {
                Instantiate(Obj, new Vector3(200 * i, 200 * j, 0), Quaternion.identity);
            }
        }
    }

    // Update is called once per frame
    void Update()
    {
        
    }
}
