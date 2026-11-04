# Putting MAAIT online (one link for everyone)

The simulator runs as one web service. Participants open its link in a browser
(Chrome recommended); nothing is installed on their computers. Every browser tab
that opens the link gets **its own mission**, so several people can run sessions
at the same time. The administrator's scenario (settings page) applies to all of
them.

The steps below use **Render** (render.com), which can run the included
`Dockerfile` with no server setup. Any host that runs Docker containers and
allows WebSockets works the same way (Railway, Fly.io, a university VM).

## 1. Put the code on GitHub

1. Create a free account at github.com, then **New repository** (e.g. `maait`).
   Make it **Private** if the study materials should not be public.
2. Upload this whole folder, including `src/reschu/web/static/videos` (the camera
   footage, about 100 MB). The easiest way is **GitHub Desktop**: *Add local
   repository* → choose this folder → *Publish repository*. (The web uploader also
   works but takes at most 100 files per upload, so do it in two batches.)

`results/` and `recordings/` are left out on purpose (see `.gitignore`).

## 2. Create the service on Render

1. Create a free account at render.com and connect your GitHub account.
2. **New → Blueprint**, pick the `maait` repository. Render reads `render.yaml`.
3. It asks for **MAAIT_ADMIN_PASSCODE**: choose the administrator passcode
   (participants must not know it). Without it the default passcode
   `maait-admin` would apply, which anyone could guess.
4. Click **Apply**. The first build takes a few minutes. When it is live, Render
   shows the address, e.g. `https://maait-simulator.onrender.com`.

That address is the link you give participants. The administrator uses the same
link and the 🔒 ADMIN button (settings, pause, speed, recording, results).

## 3. Running sessions

- **Before a session**, open the link yourself a minute early. On Render's free
  plan the service sleeps after 15 minutes without use and takes about a minute
  to wake up.
- **Set the scenario** in ADMIN → SETTINGS. It is saved on the server.
- **After each session**, ADMIN → **⬇ DOWNLOAD RESULTS** downloads every results
  CSV as one zip. Do this every time: on the free plan, files on the server are
  wiped whenever the service restarts, redeploys or goes to sleep (the saved
  scenario too).

### Keeping files on the server (optional, paid)

To keep results, recordings and the scenario across restarts, switch the service
to a paid instance type in Render and add a **Disk** mounted at `/data`. The
container already writes everything to `/data` (`MAAIT_DATA_DIR`).

## Updating the simulator

Push the new code to the same GitHub repository (GitHub Desktop: *Commit* →
*Push*). Render rebuilds and redeploys automatically. Download results first if
you are on the free plan.

## Running the same container elsewhere

```bash
docker build -t maait .
docker run -p 8000:8000 -e MAAIT_ADMIN_PASSCODE=yourcode -v "$PWD/data:/data" maait
# open http://localhost:8000
```
