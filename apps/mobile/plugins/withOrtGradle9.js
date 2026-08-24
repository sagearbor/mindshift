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
const path = require("path");
const {
  withGradleProperties,
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
