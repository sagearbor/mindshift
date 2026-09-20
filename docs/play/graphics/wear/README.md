# Wear OS Play Store screenshots

Six 1080×1080 PNGs, generated from a headless Wear OS emulator (no physical
watch involved) on 2026-09-20. Each shows the round watch face centred on a
solid `#0b0f17` background, per Play's Wear OS graphic spec — no device
frame, no marketing captions.

| File | Screen | How it was produced |
|---|---|---|
| `wear-01-off.png` | Main (glance) screen, sentinel **Off** | Fresh launch, before arming. Dial center-display. Red status ring (off), default green meter arc (no reading yet). |
| `wear-02-calm.png` | Main (glance) screen, sentinel **On**, calm | Armed with the **Volume** signal selected. Emulator has no audio input (`-no-audio`), so the mic reads silence (~-76 dB) — a real, honestly-quiet reading, well under the arm/threshold line. Green ring. |
| `wear-03-elevated.png` | Main (glance) screen, sentinel **On**, elevated meter | Armed with the **Heart Rate** signal selected, then the emulator's virtual heart-rate sensor was driven via `adb emu sensor set heart-rate 130` so Health Services delivers a real over-threshold bpm reading. Red ring (meter over threshold). **Note:** this is the closest reachable stand-in for a "nudge level" shot — see "What we could not reach" below. |
| `wear-04-signal-picker.png` | Signal picker (Volume / Heart Rate / Movement) | Opened via the "Signal: …" chip on the glance screen. |
| `wear-05-settings.png` | Settings ("Feel the buzzes" demo section) | Opened via the "⚙ Settings" chip. |
| `wear-06-mode-picker.png` | Mode picker (Standard / Battery Saver / Session) | Opened via the mode chip (bottom of the glance screen list). |

## What we could not reach

A full **nudge/episode** state (the orange/red "Episode" ring + sparkline +
📈 vector icon, `SentinelState.STREAMING`) is triggered purely by the mic
audio path: `SentinelController.processWindow()` runs the loudness detector
on real `AudioRecord` windows, independent of whichever signal (`Volume` /
`Heart Rate` / `Movement`) the wearer has selected for display. The emulator
was started `-no-audio` (per this task's RAM/headless constraints), so
`AudioRecord` only ever delivers digital silence — there is no console/`adb`
command to inject a synthetic microphone signal (the Android Emulator's
"Insert audio" feature lives only in the Extended Controls GUI, wired through
an authenticated gRPC endpoint, not the telnet console or `adb emu`).

The **Heart Rate over-threshold** shot (`wear-03-elevated.png`) is the
closest real, honestly-obtained substitute: the emulator does expose a
virtual heart-rate sensor for Wear Health Services testing
(`adb emu sensor set heart-rate <bpm>`), which the app consumes through the
same `HrSource` (Health Services) path a real Pixel Watch would use. That
gets a genuine `meter.over == true` / red-ring reading, but it is an
elevated **meter**, not a fired **nudge episode** — no 📈 vector icon or
haptic pattern accompanies it, since those are downstream of the mic-only
detector, which never saw a loud window.

There is also no dedicated "baseline calibration" screen in this app —
baseline tracking is automatic/backgrounded (see
`apps/watch/wearApp/src/main/kotlin/app/gauge/wear/control/SentinelController.kt`),
so it was swapped for the signal/mode pickers above to round out the shot
count.

## Reusable recipe (for the next app / next screenshot refresh)

```bash
export PATH="/opt/homebrew/bin:$PATH"
export JAVA_HOME=/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home
SDK=~/Library/Android/sdk
export ANDROID_SDK_ROOT=$SDK ANDROID_HOME=$SDK

# 1. One-time setup — install the Wear system image, create the AVD.
yes | "$SDK/cmdline-tools/latest/bin/sdkmanager" --licenses
"$SDK/cmdline-tools/latest/bin/sdkmanager" \
  "platform-tools" "emulator" "system-images;android-34;android-wear;arm64-v8a"
echo no | "$SDK/cmdline-tools/latest/bin/avdmanager" create avd \
  -n wear_shots -k "system-images;android-34;android-wear;arm64-v8a" \
  -d wearos_small_round --force
sed -i '' 's/^hw.ramSize=512$/hw.ramSize=1536/' ~/.android/avd/wear_shots.avd/config.ini

# 2. Boot headless (foreground shell; backgrounds the emulator process itself,
#    then polls boot state in the same call — respects a "no background jobs"
#    hook).
nohup "$SDK/emulator/emulator" -avd wear_shots -no-window -no-audio \
  -gpu swiftshader_indirect -no-snapshot -netdelay none -netspeed full \
  > tmp/play-shots/emulator-boot.log 2>&1 &
disown
adb wait-for-device
until [ "$(adb shell getprop sys.boot_completed 2>/dev/null | tr -d '\r')" = "1" ]; do sleep 5; done

# 3. Build + install the debug APK, grant the dangerous permissions
#    (no interactive dialogs on a fresh emulator otherwise).
cd apps/watch && ./gradlew :wearApp:assembleDebug
adb install -r wearApp/build/outputs/apk/debug/wearApp-debug.apk
PKG=com.sagearbor.gauge.wear.debug
adb shell pm grant $PKG android.permission.RECORD_AUDIO
adb shell pm grant $PKG android.permission.BODY_SENSORS
adb shell pm grant $PKG android.permission.POST_NOTIFICATIONS

# 4. Launch, navigate with `adb shell input tap/swipe` (use
#    `adb shell uiautomator dump /sdcard/window_dump.xml` + `adb pull` to read
#    back exact chip bounds instead of guessing coordinates), and capture:
adb shell am start -n $PKG/app.gauge.wear.ui.MainActivity
adb exec-out screencap -p > tmp/play-shots/raw-NN-<screen>.png

# Useful sensor injections for a real (non-fabricated) "elevated" reading
# without needing loud audio:
adb -e emu sensor set heart-rate 72   # settle a baseline first
adb -e emu sensor set heart-rate 130  # then spike it

# 5. Post-process every raw capture into the Play spec (round face on
#    #0b0f17, 1080x1080) — see process.py in this same directory for the
#    exact script; it just needs Pillow:
#    tmp/venv/bin/python tmp/play-shots/process.py tmp/play-shots

# 6. Tear down — free the RAM/disk, keep the system image for next time.
adb emu kill
"$SDK/cmdline-tools/latest/bin/avdmanager" delete avd -n wear_shots
```

The `process.py` script used to generate these PNGs is kept at
`tmp/play-shots/process.py` in this repo checkout (gitignored, since `tmp/`
is not tracked) — copy it out if you want it preserved past this session;
it's ~60 lines of Pillow (scale the raw square capture to a ~980px circle,
antialias the mask by supersampling 4x, paste centered on a 1080×1080
`#0b0f17` canvas).
