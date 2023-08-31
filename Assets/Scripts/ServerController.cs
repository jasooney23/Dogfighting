using System.Collections;
using System.Collections.Generic;
using UnityEngine;

using System;
using System.Net;
using System.Net.Sockets;
using System.Text;

public class ServerController : MonoBehaviour
{
    public int port;
    private IPEndPoint localEndPoint;
    private Socket listener;

    // Start is called before the first frame update
    void Start()
    {
        // Establish the local endpoint
        // for the socket. Dns.GetHostName
        // returns the name of the host
        // running the application.
        IPHostEntry ipHost = Dns.GetHostEntry(Dns.GetHostName());
        IPAddress ipAddr = ipHost.AddressList[0];
        localEndPoint = new IPEndPoint(IPAddress.Parse("127.0.0.1"), port);

        // Creation TCP/IP Socket using
        // Socket Class Constructor
        listener = new Socket(ipAddr.AddressFamily,
                     SocketType.Stream, ProtocolType.Tcp);

        listener.Bind(localEndPoint);
        listener.Listen(10);

        ListenForConnections(listener);
    }

    async void ListenForConnections(Socket listener)
    {
        while (true)
        {
            Socket clientSocket = await listener.AcceptAsync();

            // Data buffer
            byte[] bytes = new Byte[1024];
            string data = null;

            int numByte = clientSocket.Receive(bytes);
            data += Encoding.ASCII.GetString(bytes,
                                       0, numByte);

            Debug.Log($"Text received -> {data} ");
            byte[] message = Encoding.ASCII.GetBytes("Test Server");

            // Send a message to Client
            // using Send() method
            clientSocket.Send(message);

            // Close client Socket using the
            // Close() method. After closing,
            // we can use the closed Socket
            // for a new Client Connection
            clientSocket.Shutdown(SocketShutdown.Both);
            clientSocket.Close();

        }
    }
}
