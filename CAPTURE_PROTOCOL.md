# Capture Protocol

**You need:** an iPhone 15 or newer. For the LiDAR tier, an iPhone **Pro** (15 Pro / 16 Pro / 17 Pro).
**Before you start:** turn on all the lights and open the blinds. Open every interior door fully. Lens clean. Battery > 30 %.

## A. Photos (any iPhone 15+) — about 1 min per room
1. Use the built-in **Camera** app, **Photo** mode, **1×** lens (not 0.5×, not 2×). Hold the phone **upright (portrait)** at chest height. No flash. Leave Live Photo and HDR at their defaults.
2. In each room, stand in a corner and take **one photo looking at the opposite corner**. Do this from **every corner** (4 photos for a rectangular room). Each photo must show the floor-to-wall line and the ceiling line.
3. Take **one extra photo of each doorway**, straight on, from 2–3 m away, with the full door frame visible top to bottom. *Rooms are joined through these photos — do not skip them.*
4. 2–8 photos per room. Do not take panoramas, zoom, or crop.
5. **Hand-off:** on a Mac, select one room's photos in Photos → File → Export → *Export Unmodified Original*, into a folder named after the room (`kitchen/`, `hall/`, …). Put all the room folders inside one capture folder. Keep the originals (do not screenshot or resend through messaging apps: that strips the lens data).

## B. Video (any iPhone 15+) — about 1 min per room
1. **Camera** app, **Video** mode, **1×** lens, 4K 30 fps (or 1080p 30 fps), held **landscape** at chest height.
2. Start recording in the first room. Walk slowly (half normal speed) **along the walls**, pointing the camera at the opposite wall so the floor line and ceiling line stay in frame. Go through each doorway **slowly, facing forward**, and repeat in the next room.
3. **Finish back where you started**, pointing at the same view as the first second of the video.
4. Do not pause, zoom, or switch lens. If you lose your way, stop and start a new recording.
5. **Hand-off:** AirDrop the video to the Mac (Options → *All Photos Data* on) or export as an unmodified original. Put it alone in a capture folder.

## C. LiDAR (iPhone Pro only) — about 2 min per room
1. Install **Stray Scanner** (free, App Store). Open it and allow camera access.
2. Tap record. Walk the same path as for the video, in **one recording through all rooms**: walls first, keep about 1–3 m from the wall, sweep the phone up to the ceiling and down to the floor once per wall. Go through doorways slowly.
3. **Finish back where you started**, pointing at the same view you began with. Stop recording.
4. **Hand-off:** connect the iPhone to the Mac → Finder → iPhone → **Files** tab → *Stray Scanner* → drag the recording folder (contains `rgb.mp4`, `depth/`, `confidence/`, `odometry.csv`, `camera_matrix.csv`) to the Mac.
5. **Note the phone model** (Settings → General → About → Model Name). Stray does not record it, and the LiDAR depth correction is per device. Without it, every interval is widened by a 1% depth-scale prior.
6. **Optional, once per phone:** measure one ceiling height with a tape or laser meter in a room you scanned, then run
   `roomscope calibrate-depth <capture-folder> --device "iPhone 15 Pro" --measured R1:ceiling=2.95`.
   That stores the phone's depth scale; later runs with `--device "iPhone 15 Pro"` use it and get the narrower intervals.

## Run (all tiers)
```
roomscope run <capture-folder-or-video> [--device "iPhone 15 Pro"]
```
Output: `out/<capture>/result.json` and `out/<capture>/plan.png`.

## Avoid
- **Mirrors and glass:** do not stand still pointing at a mirror or glass door. Sweep past.
- **Dark rooms:** if the room is dim, turn the lights on. Damage detection skips frames that are too dark and says so; the room geometry has no low-light check, so dim captures can come out wrong without a warning.
- **Moving things:** keep people and pets out of frame. Do not move furniture partway through a capture.
- **Fast motion:** blur ruins every tier. Count "one-thousand-one" per step.
