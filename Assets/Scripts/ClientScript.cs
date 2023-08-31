using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.SceneManagement;
using Unity.Collections;
using Unity.Collections.LowLevel.Unsafe;

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;
using System.Linq;
using TMPro;

public class ClientScript : MonoBehaviour
{
    public int Port;
    private IPEndPoint localEndPoint;
    private Socket client;

    public int FramesBetweenAgentUpdate;
    private int framesSinceUpdate;
    private float[] lastPrediction1;
    private float[] lastPrediction2;

    public GameObject Plane1;
    public GameObject Plane2;
    private PlaneController PC1;
    private PlaneController PC2;

    public GameObject ValueText;

    private bool terminal;

    // Start is called before the first frame update
    void Start()
    {
        IPHostEntry ipHost = Dns.GetHostEntry(Dns.GetHostName());
        IPAddress ipAddr = ipHost.AddressList[0];
        localEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), Port);
        client = new Socket(ipAddr.AddressFamily,
                     SocketType.Stream, ProtocolType.Tcp);

        client.Connect(localEndPoint);

        PC1 = Plane1.GetComponent<PlaneController>();
        PC2 = Plane2.GetComponent<PlaneController>();

        framesSinceUpdate = 0;

        lastPrediction1 = new float[4] { 0, 0, 0, 0 };
        lastPrediction2 = new float[4] { 0, 0, 0, 0 };

        terminal = false;
        //Debug.Log("================================================");
    }


    // Update is called once per frame
    void FixedUpdate()
    {
        if (!terminal)
        {
            framesSinceUpdate++;
            if (framesSinceUpdate != FramesBetweenAgentUpdate)
            {
                input(PC1, lastPrediction1);
                input(PC2, lastPrediction1);
                return;
            }
            else
            {
                framesSinceUpdate = 0;
            }


            /* B
             * Send update to agent. ALSO SENDS THE FIRST STATE */

            // Send screen data
            NativeArray<byte> cam1 = getCameraData(PC1);
            NativeArray<byte> cam2 = getCameraData(PC2);
            sendBytes(cam1.ToArray());
            //Debug.Log("Sent CAM1");
            sendBytes(cam2.ToArray());
            //Debug.Log("Sent CAM2");

            // Send plane instrument data
            float[] plane1 = getPlaneData(PC1);
            float[] plane2 = getPlaneData(PC2);
            send(plane1.Length.ToString());
            //Debug.Log($"Sent DATA1 len ({plane1.Length.ToString()}), sending DATA1");
            for (int i = 0; i < plane1.Length; i++)
            {
                send(plane1[i].ToString());
                Debug.Log(plane1[i].ToString());
            }
            send(plane2.Length.ToString());
            //Debug.Log($"Sent DATA2 len ({plane2.Length.ToString()}), sending DATA2");
            for (int i = 0; i < plane2.Length; i++)
            {
                send(plane2[i].ToString());
                Debug.Log(plane2[i].ToString());
            }



            // Restart if dead
            if (PC1.Dead > 0 || PC2.Dead > 0)
            {
                restart();
                return;
            }

            /* A
                * Receive agent action */

            // Receive predictions of plane 1
            string data = recv();
            //Debug.Log($"Received ACTION1 len ({data}), receiving ACTION1:");
            int numMessages = int.Parse(data);
            float[] prediction1 = new float[numMessages];
            for (int i = 0; i < numMessages; i++)
            {
                prediction1[i] = float.Parse(recv());
                Debug.Log(prediction1[i].ToString());
            }

            TextMeshProUGUI valueTextComponent = ValueText.GetComponent<TextMeshProUGUI>();
            valueTextComponent.text = prediction1[numMessages - 1].ToString();

            // Receive predictions of plane 2
            data = recv();
            //Debug.Log($"Received ACTION2 len ({data}), receiving ACTION2:");
            numMessages = int.Parse(data);
            float[] prediction2 = new float[numMessages];
            for (int i = 0; i < numMessages; i++)
            {
                prediction2[i] = float.Parse(recv());
                Debug.Log(prediction2[i].ToString());
            }

            lastPrediction1 = prediction1;
            lastPrediction2 = prediction2;
            /* Input actions from agents to plane controller and update the game */
            input(PC1, lastPrediction1);
            input(PC2, lastPrediction2);
        }   
    }

    void sendBytes(byte[] bytes)
    {
        client.Send(bytes);
        client.Receive(new byte[1024]);
    }

    void send(string msg)
    {
        client.Send(stringToBytes(msg));
        //Debug.Log("msg sent");
        client.Receive(new byte[1024]);
        //Debug.Log("msg acknowledged");
    }

    string recv()
    {
        byte[] data = new byte[1024];
        client.Receive(data);
        //Debug.Log("msg received");
        client.Send(stringToBytes("acknowledge"));
        //Debug.Log("msg acknowledged");
        return bytesToString(data);
    }

    byte[] stringToBytes(string data)
    {
        return Encoding.ASCII.GetBytes(data);
    }
    string bytesToString(byte[] data)
    {
        return Encoding.ASCII.GetString(data);
    }

    NativeArray<byte> getCameraData(PlaneController PC)
    {
        RenderTexture ren = PC.RenderTexture;
        Texture2D tex = new Texture2D(ren.width, ren.height, TextureFormat.R8, false);
        RenderTexture.active = ren;
        tex.ReadPixels(new Rect(0, 0, ren.width, ren.height), 0, 0);
        tex.Apply();

        PC.transform.GetChild(0).gameObject.GetComponent<Camera>().Render();

        return tex.GetPixelData<byte>(0);
    }

    float[] getPlaneData(PlaneController PC)
    {
        float[] planeData = new float[7];

        //planeData[0] = PC.transform.position.x / 1000;
        //planeData[1] = PC.transform.position.y / 1000;
        //planeData[2] = PC.transform.position.z / 1000;

        planeData[0] = PC.Attitude[0] / 360;
        planeData[1] = PC.Attitude[1] / 360;
        planeData[2] = PC.Attitude[2] / 360;

        planeData[3] = PC.Speed / PC.MaxSpeed;
        planeData[4] = PC.Throttle / PC.EngineThrust;
        planeData[5] = PC.AngleToEnemy() / 360;

        planeData[6] = PC.Dead;


        return planeData;
    }

    void input(PlaneController PC, float[] prediction)
    {
        PC.ChangeAttitude(prediction[0], prediction[1]);
        PC.ChangeThrottle(prediction[2]);
        if (prediction[3] > 0)
        {
            PC.gameObject.GetComponent<FireController>().Fire();
        }
    }

    void restart()
    {
        // Restart game
        recv();

        //BulletController[] bullets = FindObjectsByType<BulletController>(FindObjectsSortMode.None);
        //for (int i = 0; i < bullets.Length; i++)
        //{
        //    Destroy(bullets[i].gameObject);
        //}
        //PlaneController[] planes = FindObjectsByType<PlaneController>(FindObjectsSortMode.None);
        //for (int i = 0; i < planes.Length; i++)
        //{
        //    Destroy(planes[i].gameObject);
        //}

        SceneManager.LoadScene(SceneManager.GetActiveScene().name);
        terminal = true;
    }
}