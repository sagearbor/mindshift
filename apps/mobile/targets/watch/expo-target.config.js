/**
 * watchOS companion target for MindShift.
 *
 * Generated into ios/ by @bacons/apple-targets on every `expo prebuild`; the
 * Swift in this directory is the source of truth and ios/ stays gitignored.
 *
 * `name` is the Xcode *target/product* name, and it is load-bearing in three
 * places that must agree or the cloud build fails at signing:
 *   - the key in apps/mobile/credentials.json (multi-target form)
 *   - extra.eas.build.experimental.ios.appExtensions[].targetName (written by
 *     the plugin, read by eas-cli to enumerate targets for a CNG project)
 *   - the target name in the generated pbxproj
 * scripts/ios_credentials_bootstrap.py --watch-target takes the same string.
 *
 * `bundleIdentifier` starting with "." is appended to the app's bundle id, so
 * this is com.sagearbor.mindshift.app.watchkitapp — the suffix Apple expects
 * for a watch companion.
 *
 * @type {import('@bacons/apple-targets/app.plugin').Config}
 */
module.exports = {
  type: "watch",
  name: "MindShiftWatch",
  displayName: "MindShift",
  bundleIdentifier: ".watchkitapp",
  // watchOS 10 is the floor for the SwiftUI APIs the wrist UI will want
  // (NavigationSplitView on watch, .containerBackground). Raise, don't lower.
  deploymentTarget: "10.0",
  // The watch app's own icon. Apple rejected the first delivery that carried
  // this target (build 5, 2026-09-25) with ITMS-90391 "No icons found for
  // watch application" and ITMS-90713 "CFBundleIconName missing": the plugin
  // only writes an AppIcon asset catalog and that Info.plist key when an icon
  // is given. Same artwork as the phone (1024×1024, no alpha), which is what
  // watchOS expects; the system masks it to a circle.
  icon: "../../assets/icon.png",
  frameworks: ["SwiftUI", "WatchConnectivity"],
};
