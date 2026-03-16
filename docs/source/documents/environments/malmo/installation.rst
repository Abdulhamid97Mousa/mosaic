Installation Guide
==================

This guide documents the complete installation of **Microsoft Malmo** (MalmoEnv) as used in
MOSAIC. Follow these steps once; after that, starting Minecraft takes only seconds.

Prerequisites
-------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Dependency
     - Notes
   * - **Java 8 (JDK)**
     - **Exactly Java 8** — Gradle 2.14 (bundled) does not support Java 9+.
       On Ubuntu: ``sudo apt install openjdk-8-jdk``
   * - **MOSAIC virtual environment**
     - All Python steps assume ``.venv`` is activated:
       ``source .venv/bin/activate``
   * - **Git LFS** (optional)
     - Only needed if cloning the full MOSAIC repo with binary assets.

.. warning::

   The Gradle wrapper bundled with Malmo (``gradlew``) requires **Java 8 exactly**.
   Running with Java 17 or any other version prints:
   ``Could not determine java version from '17.0.x'`` and aborts.

   Always set ``JAVA_HOME`` before any Malmo/Minecraft command::

       export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64

Step 1: Install the MalmoEnv Python package
--------------------------------------------

The MalmoEnv Python package lives in the MOSAIC repository under
``3rd_party/environments/malmo/MalmoEnv/``.  Install it in editable mode:

.. code-block:: bash

   source .venv/bin/activate
   pip install --no-build-isolation -e 3rd_party/environments/malmo/MalmoEnv/

Verify the installation:

.. code-block:: bash

   python -c "import malmoenv; print('malmoenv OK')"

Step 2: Set ``MALMO_XSD_PATH``
--------------------------------

MalmoEnv requires an environment variable pointing to the Malmo XML schema directory:

.. code-block:: bash

   export MALMO_XSD_PATH=/home/hamid/Desktop/software/mosaic/3rd_party/environments/malmo/Schemas

To make this permanent, add it to ``~/.bashrc``:

.. code-block:: bash

   echo 'export MALMO_XSD_PATH=/home/hamid/Desktop/software/mosaic/3rd_party/environments/malmo/Schemas' >> ~/.bashrc
   source ~/.bashrc

Step 3: Create ``version.properties``
---------------------------------------

The Minecraft Forge mod requires a version file to identify itself.  This file is not tracked
in the repository and must be created manually:

.. code-block:: bash

   echo "malmomod.version=0.37.0" > \
     3rd_party/environments/malmo/Minecraft/src/main/resources/version.properties

Step 4: Download Minecraft Assets (One-time, critical)
---------------------------------------------------------

.. warning::

   **Mojang dropped HTTP support** for their CDN.  The ForgeGradle 2.14 wrapper bundled
   with Malmo 0.37.0 downloads all 1 196 Minecraft 1.11.2 assets over **HTTP**, which now
   returns HTTP 400 errors for every single asset.

   You must download the assets **manually via HTTPS** before Minecraft will launch
   successfully.

The assets index for Minecraft 1.11.2 is at::

    https://launchermeta.mojang.com/v1/packages/...

Run the following Python script to download all missing assets (safe to re-run — skips
already-downloaded files):

.. code-block:: python

   #!/usr/bin/env python3
   """Download Minecraft 1.11.2 assets via HTTPS to bypass ForgeGradle's broken HTTP downloader."""

   import hashlib
   import json
   import os
   import time
   import urllib.request
   from pathlib import Path

   ASSETS_DIR = Path.home() / ".gradle/caches/minecraft/assets/objects"
   INDEX_URL = (
       "https://launchermeta.mojang.com/v1/packages/"
       "d5a285bdf7b0c8f1dd6e57f36564da0ea17e3c87/1.11.json"
   )
   CDN = "https://resources.download.minecraft.net"

   print("Fetching asset index …")
   with urllib.request.urlopen(INDEX_URL) as r:
       index = json.load(r)

   objects = index["objects"]
   total = len(objects)
   downloaded = skipped = 0

   for i, (name, info) in enumerate(objects.items(), 1):
       h = info["hash"]
       prefix = h[:2]
       dest = ASSETS_DIR / prefix / h
       dest.parent.mkdir(parents=True, exist_ok=True)
       if dest.exists() and dest.stat().st_size == info["size"]:
           skipped += 1
           continue
       url = f"{CDN}/{prefix}/{h}"
       for attempt in range(1, 6):
           try:
               urllib.request.urlretrieve(url, dest)
               downloaded += 1
               print(f"[{i}/{total}] {name}")
               break
           except Exception as exc:
               print(f"  attempt {attempt} failed: {exc}")
               time.sleep(2 ** attempt)

   print(f"\nDone. Downloaded: {downloaded}  Skipped (already present): {skipped}  Total: {total}")

Save the script as ``download_assets.py`` and run it once:

.. code-block:: bash

   python download_assets.py

Expected output ends with something like::

   Done. Downloaded: 1196  Skipped (already present): 0  Total: 1196

.. note::

   One asset (``minecraft/sounds/music/game/nether/nether2.ogg``) consistently fails to
   download because it no longer exists on Mojang's CDN.  This is benign — Minecraft logs
   a warning but launches and runs fine without it.

Step 5: Build and Launch Minecraft
-------------------------------------

Before the first run, Gradle needs to compile the Malmo Forge mod:

.. code-block:: bash

   export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64
   cd 3rd_party/environments/malmo/Minecraft
   bash launchClient.sh -port 9000 -env

The first run takes several minutes as Gradle downloads dependencies and compiles the mod.
Subsequent launches take only a few seconds.

When Minecraft is ready you will see in the terminal::

   [Client thread/INFO] [STDOUT]: ***** Start MalmoEnvServer on port 9000
   [Client thread/INFO] [STDOUT]: CLIENT enter state: DORMANT

Minecraft is now waiting for MalmoEnv connections on port 9000.

.. important::

   **Always use the ``-env`` flag** — without it, Minecraft starts in interactive mode and
   ignores Python connections.

Step 6: Eclipse / IDE Launch Configuration
--------------------------------------------

The file ``3rd_party/environments/malmo/Minecraft/Minecraft_Client.launch`` is a
pre-configured Eclipse launch configuration that passes the correct JVM arguments
(``-Xmx2G``) and sets ``JAVA_HOME`` automatically.  You can use it with Eclipse or
IntelliJ (via the Eclipse Compatibility plugin) to launch Minecraft without the terminal.

Step 7: Verify the Integration
---------------------------------

With Minecraft running, test the MOSAIC integration:

.. code-block:: python

   import malmoenv

   env = malmoenv.Env(reshape=True)

   # Read the mission XML
   from pathlib import Path
   xml = (Path("3rd_party/environments/malmo/MalmoEnv/missions") / "mobchase_single_agent.xml").read_text()

   env.init(xml, 9000, server="localhost")
   obs = env.reset()
   print("Observation shape:", obs.shape)   # (84, 84, 3) RGB

   obs, reward, done, info = env.step(0)    # action 0 = move forward
   print("Reward:", reward, "Done:", done)
   env.close()

Then launch MOSAIC, click **Add Operator**, select game family **MalmoEnv**, choose any
mission (e.g. ``MalmoEnv-MobChase-v0``), and the RGB frame from Minecraft will appear in
the render panel.

Troubleshooting
---------------

.. list-table::
   :header-rows: 1
   :widths: 45 55

   * - Error
     - Fix
   * - ``Could not determine java version from '17.0.x'``
     - Set ``export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64`` before running ``launchClient.sh``
   * - ``HTTP 400`` errors for all 1 196 assets during first run
     - Run the asset download script from Step 4 above
   * - ``malmoenv package is not installed``
     - Activate the venv and run ``pip install -e 3rd_party/environments/malmo/MalmoEnv/``
   * - ``MALMO_XSD_PATH not set``
     - Export the variable as shown in Step 2
   * - ``Mission XML not found``
     - Verify that ``3rd_party/environments/malmo/MalmoEnv/missions/`` contains ``.xml`` files
   * - Minecraft launches but MOSAIC can't connect
     - Confirm Minecraft was started with ``-env`` flag and port 9000 is free
   * - ``Connection refused`` on env.reset()
     - Wait until the terminal shows ``CLIENT enter state: DORMANT`` before connecting
   * - ``nether2.ogg does not exist`` warning
     - Benign — Mojang removed this file; Minecraft runs fine without it
