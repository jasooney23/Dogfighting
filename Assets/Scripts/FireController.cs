using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using TMPro;

public class FireController : MonoBehaviour
{
    PlaneController PC;

    public GameObject Bullet;
    public GameObject AmmoText;

    public int RPM; // Converted to time between shots
    public int MaxAmmo; // Max # of rounds held at any time
    public float ReloadTime; // Time to fully reload gun from empty
    public float ReloadDelay; // Time between last shot, and starting gun reload

    private int ammo;
    private float timeSinceFire;
    private float timeBetweenShots;
    private float timeSinceReload;
    private float timeBetweenReloads;

    public int Ammo { get => ammo; }

    // Start is called before the first frame update
    void Start()
    {
        PC = GetComponent<PlaneController>();

        timeSinceFire = 0;
        timeBetweenShots = 1f / (RPM / 60);

        ammo = MaxAmmo;
        timeSinceReload = 0;
        timeBetweenReloads = ReloadTime / MaxAmmo;
    }

    // Update is called once per frame
    void FixedUpdate()
    {
        timeSinceFire += Time.fixedDeltaTime;
        timeSinceReload += Time.fixedDeltaTime;

        if (Input.GetAxis("Fire") > 0)
        {
            Fire();
        }
        else if (timeSinceFire >= ReloadDelay && timeSinceReload >= timeBetweenReloads && ammo < MaxAmmo)
        {
            timeSinceReload = 0;
            ammo++;
        }

        AmmoText.GetComponent<TextMeshProUGUI>().text = $"{ammo}/{MaxAmmo}";
    }

    public void Fire()
    {
        if (timeSinceFire >= timeBetweenShots && ammo > 0)
        {
            timeSinceFire = 0;
            ammo--;

            GameObject newBullet = Instantiate(Bullet);
            newBullet.SetActive(true);
            newBullet.GetComponent<BulletController>().init(PC.Scale, transform.position, transform.right);
        }
    }
}
