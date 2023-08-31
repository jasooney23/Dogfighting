using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class GameSpeedController : MonoBehaviour
{
    public float GameSpeed;
    // Start is called before the first frame update
    void Start()
    {
        Time.fixedDeltaTime = 0.02f / GameSpeed;
        Time.timeScale = GameSpeed;
    }

    // Update is called once per frame
    void Update()
    {
        Time.timeScale = GameSpeed;
        //Debug.Log(Time.deltaTime);
    }
}
