// Expo config plugin: two native-build fixes the on-device fast loop's
// packages need under Expo SDK 57 / RN 0.86 (Gradle 9.3.1). Found by
// running `expo prebuild` + `./gradlew assembleDebug` on 2026-08-24.
//
// 1. onnxruntime-react-native 1.24.3's android/build.gradle (line ~250)
//    calls `VersionNumber.parse(REACT_NATIVE_VERSION)` unqualified. Gradle 8
//    exposed `org.gradle.util.VersionNumber` as a default script import;
//    Gradle 9 removed it, so project evaluation fails with "Could not get
//    unknown property 'VersionNumber'" before a single task runs. Exposing
//    the still-present internal class as an `ext` property on every project
//    satisfies the lookup without patching node_modules. Drop once ORT-RN
//    ships a Gradle-9-clean build.gradle.
//
// 2. expo-ai-kit pulls com.google.mlkit:genai-prompt (Gemini Nano), whose
//    manifest declares minSdk 26; Expo's default is 24, and the manifest
//    merger refuses the build. Raise the app's minSdk to 26 (Android 8.0 —
//    the on-device STT this feature depends on needs 13+ anyway) through the
//    `android.minSdkVersion` gradle property Expo's autolinking settings
//    plugin maps onto rootProject.ext.minSdkVersion.
//
// 3. (iOS) onnxruntime-react-native's own config plugin writes
//    `pod 'onnxruntime-react-native', :path => '../node_modules/onnxruntime-react-native'`
//    into the Podfile — a path relative to apps/mobile. This repo is an npm
//    WORKSPACE, so the package is hoisted to the ROOT node_modules and that
//    path doesn't exist ("No podspec found for `onnxruntime-react-native`",
//    2026-08-24, first iOS prebuild on the new Mac). Rewrite the `:path` to
//    wherever Node actually resolves the package, relative to ios/.
//
// 4. 16 KB memory page support. Play rejected the vc 36 production release
//    with "Your app does not support 16 KB memory page sizes". Dumping the
//    ELF program headers of every arm64 .so in the AAB found exactly ONE
//    offender out of 25: `libonnxruntimejsi.so`, at p_align=4096. ORT's own
//    prebuilt `libonnxruntime.so` is already fine — it is the small JSI shim
//    that onnxruntime-react-native compiles from source that misses out,
//    because its CMake invocation never opts into the NDK r27 flag that the
//    React Native and Expo modules set for themselves. Pass it explicitly
//    (plus the raw linker flag, so this holds on an NDK that doesn't know
//    the option). Verify after a build with scripts/check_16kb_alignment.py.
//
// 5. onnxruntime-react-native is COMPILED but never REGISTERED on Android, so
//    `NativeModules.Onnxruntime` is null at runtime and the package's own
//    lib/binding.ts calls `Module.install()` on it unconditionally at import:
//
//      TypeError: Cannot read property 'install' of null
//        tryRequire -> ortNative -> buildVad -> probeFastLoopCapabilities
//
//    Live Coach probes fast-loop capabilities on mount, so opening it killed
//    the app instantly (v1.18.0 vc 37 on a Pixel 10, 2026-08-26 — found in
//    `dumpsys dropbox`, six recorded crashes).
//
//    Why it compiles but never registers: ORT's own app.plugin.js adds ONLY
//    the Gradle dependency (`implementation project(':onnxruntime-react-native')`),
//    which is why the classes and .so files are in the APK. For registration
//    it relies on autolinking, and neither path picks it up — it ships a
//    LEGACY `unimodule.json` (modern Expo uses expo-module.config.json) and no
//    `react-native.config.js`, so it is absent from
//    `expo-modules-autolinking react-native-config --platform android`.
//    Registering it by hand in MainApplication's documented escape hatch is
//    exact and does not depend on discovery. `OnnxruntimePackage` is a plain
//    ReactPackage; RN 0.76 bridgeless reaches it through the legacy interop.
const path = require("path");
const {
  withGradleProperties,
  withMainApplication,
  withPodfile,
  withProjectBuildGradle,
} = require("expo/config-plugins");
const generateCode = require("@expo/config-plugins/build/utils/generateCode");

const MIN_SDK = "26";

const ORT_POD_LINE = /pod 'onnxruntime-react-native',\s*:path\s*=>\s*'[^']*'/;

function ortPodPathRelativeToIos(projectRoot) {
  const pkgJson = require.resolve("onnxruntime-react-native/package.json", {
    paths: [projectRoot],
  });
  const rel = path.relative(path.join(projectRoot, "ios"), path.dirname(pkgJson));
  return rel.startsWith(".") ? rel : `./${rel}`;
}

const SNIPPET = `allprojects {
  // onnxruntime-react-native 1.24.3 references the Gradle-8 auto-import
  // \`VersionNumber\`; Gradle 9 dropped it. See plugins/withOrtGradle9.js.
  ext.VersionNumber = org.gradle.util.internal.VersionNumber
}`;

// Appended to onnxruntime-react-native's CMake arguments so the JSI shim it
// builds from source gets 16 KB-aligned LOAD segments. `+=` on the existing
// list, never a replacement: ORT's build.gradle already passes ANDROID_STL,
// NODE_MODULES_DIR, USE_NNAPI and friends there.
const PAGE_SIZE_SNIPPET = `subprojects { sub ->
  // See plugins/withOrtGradle9.js (4): libonnxruntimejsi.so was the single
  // 4 KB-aligned library in the AAB and Play blocks the release for it.
  if (sub.name == "onnxruntime-react-native") {
    sub.afterEvaluate {
      if (sub.extensions.findByName("android") != null) {
        sub.android.defaultConfig.externalNativeBuild.cmake.arguments +=
          ["-DANDROID_SUPPORT_FLEXIBLE_PAGE_SIZES=ON",
           "-DCMAKE_SHARED_LINKER_FLAGS_INIT=-Wl,-z,max-page-size=16384"]
      }
    }
  }
}`;

const withOrtGradle9 = (config) => {
  config = withProjectBuildGradle(config, (mod) => {
    if (mod.modResults.language !== "groovy") {
      throw new Error("withOrtGradle9: android/build.gradle is not Groovy");
    }
    mod.modResults.contents = generateCode.mergeContents({
      src: mod.modResults.contents,
      newSrc: SNIPPET,
      tag: "mindshift-ort-gradle9",
      anchor: /^allprojects\s*\{/m,
      offset: 0,
      comment: "//",
    }).contents;
    mod.modResults.contents = generateCode.mergeContents({
      src: mod.modResults.contents,
      newSrc: PAGE_SIZE_SNIPPET,
      tag: "mindshift-ort-16kb",
      anchor: /^allprojects\s*\{/m,
      offset: 0,
      comment: "//",
    }).contents;
    return mod;
  });

  config = withGradleProperties(config, (mod) => {
    mod.modResults = mod.modResults.filter(
      (item) => !(item.type === "property" && item.key === "android.minSdkVersion"),
    );
    mod.modResults.push({
      type: "comment",
      value: "ML Kit genai-prompt (expo-ai-kit / Gemini Nano) declares minSdk 26 — see plugins/withOrtGradle9.js",
    });
    mod.modResults.push({ type: "property", key: "android.minSdkVersion", value: MIN_SDK });
    return mod;
  });

  config = withMainApplication(config, (mod) => {
    if (mod.modResults.language !== "kt") {
      throw new Error("withOrtGradle9: MainApplication is not Kotlin");
    }
    const src = mod.modResults.contents;
    if (src.includes("OnnxruntimePackage()")) return mod;
    // The generated hook is exactly:
    //   PackageList(this).packages.apply {
    //     // Packages that cannot be autolinked yet can be added manually here...
    //   }
    const anchor = "PackageList(this).packages.apply {";
    if (!src.includes(anchor)) {
      throw new Error(
        "withOrtGradle9: MainApplication has no PackageList(...).apply hook to register ORT into",
      );
    }
    mod.modResults.contents = src.replace(
      anchor,
      `${anchor}\n          // onnxruntime-react-native ships no working autolinking on Android —\n          // see plugins/withOrtGradle9.js (5). Without this the fast loop's\n          // ORT import throws and Live Coach crashes on mount.\n          add(ai.onnxruntime.reactnative.OnnxruntimePackage())`,
    );
    return mod;
  });

  config = withPodfile(config, (mod) => {
    const rel = ortPodPathRelativeToIos(mod.modRequest.projectRoot);
    mod.modResults.contents = mod.modResults.contents.replace(
      ORT_POD_LINE,
      `pod 'onnxruntime-react-native', :path => '${rel}' # workspace-hoisted; see plugins/withOrtGradle9.js`,
    );
    return mod;
  });

  return config;
};

module.exports = withOrtGradle9;
