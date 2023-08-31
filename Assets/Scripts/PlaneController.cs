using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System;
using TMPro;

public class PlaneController : MonoBehaviour
{
    public GameObject EnemyPlane;
    public Camera Camera;

    public float Scale;

    public float InitialSpeed; // KPH
    public float MaxSpeed; // KPH
    public float EngineThrust; // Newtons, determines how fast plane accelerates

    public float Mass; // Kg
    public float Gravity; // m/s^2
    public float StallSpeed; // KPH, where the plane insta-dies

    public float MaxPitchAngularVelocity;
    public float MaxRollAngularVelocity;
    public float FullThrottleTime; // Time in SECONDS to go from 0% to 100% throttle

    public int DistanceLimit;

    public bool ChangeHUD;

    // ============================================================ //

    public GameObject SpeedText;
    public GameObject AltitudeText;
    public GameObject StallText;

    // ============================================================ //

    private Vector3 velocity;
    private float lastAltitude;
    private float throttle;
    private float dragCoefficient { get => EngineThrust / (float)Math.Pow(KPHtoMPS(MaxSpeed), 2); }
    private int dead;

    [HideInInspector]
    public RenderTexture RenderTexture;
    private Camera cam;

    public float Speed { get => Vector3.Magnitude(velocity); }
    public float Altitude { get => transform.localPosition.y / Scale; }
    public float Throttle { get => throttle / EngineThrust; }
    public float[] Attitude 
    { 
        get
        {
            float[] at = { transform.rotation.eulerAngles.x,
                transform.rotation.eulerAngles.y,
                transform.rotation.eulerAngles.z };
            return at;
        }
    }
    public int Dead { get => dead; }

    // Start is called before the first frame update
    void Start()
    {
        velocity = new Vector3(InitialSpeed / 3.6f, 0, 0);

        lastAltitude = Altitude;
        throttle = EngineThrust;

        dead = 0;

        cam = transform.GetChild(0).gameObject.GetComponent<Camera>();
        RenderTexture = new RenderTexture(144, 81, 8, RenderTextureFormat.R8);
        cam.targetTexture = RenderTexture;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        // Calculate altitude change
        float altitudeDelta = Altitude - lastAltitude;
        // Calculate KE change
        float kineticEnergyDelta = Mass * -Gravity * altitudeDelta;
        // Calculate new KE
        float kineticEnergy = Mass * (float)Math.Pow(Speed, 2) / 2 + kineticEnergyDelta;
        lastAltitude = Altitude;

        if (kineticEnergy < 0) { kineticEnergy = 0.000001F; }
        float newSpeed = (float)Math.Pow(Math.Abs(2 * kineticEnergy / Mass), 0.5);
        velocity = transform.right * newSpeed;

        // Calculate net horizontal forces
        Vector3 thrust = Vector3.right * throttle;
        Vector3 drag = Vector3.right * -dragCoefficient 
            * (float)Math.Pow(Speed, 2);

        velocity = transform.right * Speed;
        Vector3 acceleration = (thrust + drag) / Mass;
        // Check accel direction
        if (acceleration.x > 0)
        {
            acceleration = Quaternion.FromToRotation(acceleration, transform.right) * acceleration;
        } 
        else
        {
            acceleration = Quaternion.FromToRotation(acceleration, -transform.right) * acceleration;
        }
        velocity += acceleration * Time.fixedDeltaTime;

        transform.Translate(velocity * Time.fixedDeltaTime * Scale, Space.World);

        // Calculate rotations
        ChangeAttitude(Input.GetAxis("Pitch"), Input.GetAxis("Roll"));
        // Adjust throttle
        ChangeThrottle(Input.GetAxis("Throttle"));

        if (ChangeHUD)
        {
            // Display speed in Knots
            SpeedText.GetComponent<TextMeshProUGUI>().SetText(Math.Round(Speed * 1.94384, 0).ToString());
            // Display altitude in Feet
            AltitudeText.GetComponent<TextMeshProUGUI>().SetText(Math.Round(Altitude * 3.28084, 0).ToString());
            // Show stall text
            TextMeshProUGUI StallTextComponent = StallText.GetComponent<TextMeshProUGUI>();
            if (Speed < KPHtoMPS(StallSpeed))
            {
                StallTextComponent.text = "STALL";
                StallTextComponent.enabled = true;
                dead = 1;
            }
            else if (Speed < KPHtoMPS(StallSpeed) * 1.5)
            {
                StallTextComponent.text = "STALL WARNING";
                StallTextComponent.enabled = true;
            }
            else
            {
                StallTextComponent.enabled = false;
            }

            Camera.Render();

        }
        if (OOB()) { 
            dead = 1; }
    }

    private void OnTriggerEnter(Collider collider)
    {
        // Death if either collided with a bullet
        if (collider.gameObject.GetComponent<BulletController>() != null) 
        { 
            dead = 2;
            Debug.Log($"{gameObject.name} shot");
        }
    }

    float KPHtoMPS(float speed) { return speed / 3.6f; }

    public void ChangeAttitude(float pitch, float roll)
    {
        float pitchSpeed = MaxPitchAngularVelocity * pitch * Time.deltaTime / Speed;
        float rollSpeed = MaxRollAngularVelocity * roll * Time.deltaTime;
        transform.Rotate(new Vector3(-rollSpeed, 0, -pitchSpeed));
    }

    public void ChangeThrottle(float throttleInput)
    {
        throttle += EngineThrust / FullThrottleTime * throttleInput * Time.deltaTime;
        if (throttle > EngineThrust) { throttle = EngineThrust; }
        else if (throttle < 0) { throttle = 0; }
    }

    public bool OOB()
    {
        if (Math.Sqrt(Math.Pow(transform.localPosition.x, 2) + Math.Pow(transform.localPosition.z, 2)) > DistanceLimit) { return true; }
        else if (Altitude <= 0) { return true; }
        else { return false; }
    }

    public float AngleToEnemy()
    {
        //if (ChangeHUD)
        //{
        //    Debug.Log(Vector3.Angle(EnemyPlane.transform.position - transform.position, transform.right).ToString());
        //}
        return Vector3.Angle(EnemyPlane.transform.position - transform.position, transform.right);
    }
}
