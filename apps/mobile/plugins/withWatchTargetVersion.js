// Expo config plugin: make the watchOS target carry the SAME version and build
// number as the phone app.
//
// WHY
//   @bacons/apple-targets@5.0.0 hardcodes `MARKETING_VERSION: "1.0"` for every
//   generated target — createWatchAppConfigurationList() in
//   build/configuration-list.js. Combined with GENERATE_INFOPLIST_FILE = YES
//   (there is no real Info.plist for a `watch` target; getTargetInfoPlistForType
//   returns {}), the watch app's CFBundleShortVersionString is built from that
//   literal. MindShift is at 1.18.0, so App Store Connect rejects the upload:
//
//     ITMS-90473: CFBundleShortVersionString Mismatch. The CFBundleShortVersion-
//     String value '1.0' of the WatchKit app does not match the value '1.18.0'
//     of the companion iOS app.
//
//   Upstream issue: EvanBacon/expo-apple-targets#147.
//
//   CFBundleVersion has the sibling failure (ITMS-90379). The plugin *does*
//   seed CURRENT_PROJECT_VERSION from `config.ios.buildNumber` (with-widget.js),
//   so that one is usually right — but only if app.json's buildNumber is what
//   ships. eas.json uses `autoIncrement: "buildNumber"`, which rewrites it
//   before prebuild, so it normally is. We re-assert it anyway: one place that
//   says "the watch matches the phone" beats two places that might.
//
// ORDERING — this is the part that is easy to get wrong
//   This plugin MUST be listed BEFORE "@bacons/apple-targets" in app.json.
//   expo/config-plugins runs mods in REVERSE registration order (withExtendedMod
//   runs its own action, then calls nextMod — the previously registered one), so
//   the plugin listed EARLIER runs LATER, i.e. after the watch target exists.
//   Listing it after @bacons/apple-targets is worse than useless: that package
//   applies its base mod (`withXcodeProjectBetaBaseMod`) at the end of its own
//   plugin, and a custom mod registered after the base mods are generated is
//   never given a provider and silently never runs.
//
//   It also has to ride @bacons' Xcode mod (`xcodeProjectBeta2`), not Expo's
//   `withXcodeProject`: the two use different parsers (@bacons/xcode vs
//   xcode), and the target the plugin creates only exists in the former's
//   object graph during prebuild.
//
// VERIFY IT WORKED (cheap, local, before spending a 20-minute cloud build)
//   npx expo prebuild -p ios --clean
//   grep -c 'MARKETING_VERSION = 1.18.0' ios/MindShift.xcodeproj/project.pbxproj
const { withXcodeProjectBeta } = require("@bacons/apple-targets/build/with-bacons-xcode");

const withWatchTargetVersion = (config) => {
  return withXcodeProjectBeta(config, (config) => {
    const version = config.version;
    const buildNumber = config.ios?.buildNumber;
    if (!version) {
      throw new Error("withWatchTargetVersion: expo.version is not set in app.json");
    }

    const project = config.modResults;
    const targets = project.rootObject.props.targets ?? [];
    // A watchOS target is an *application* product (not an app-extension) whose
    // build settings carry WATCHOS_DEPLOYMENT_TARGET — the same test
    // @bacons/apple-targets uses in isNativeTargetOfType("watch"). Matching on
    // the setting rather than the target name means renaming the target in
    // targets/watch/expo-target.config.js cannot silently disable this.
    const watchTargets = targets.filter((target) => {
      const buildConfigs =
        target.props?.buildConfigurationList?.props?.buildConfigurations ?? [];
      return buildConfigs.some(
        (bc) => bc.props?.buildSettings?.WATCHOS_DEPLOYMENT_TARGET != null,
      );
    });

    if (watchTargets.length === 0) {
      throw new Error(
        "withWatchTargetVersion: no watchOS target found in the Xcode project. " +
          "Either targets/watch/expo-target.config.js is missing, or this plugin " +
          "ran before @bacons/apple-targets created the target — check the plugin " +
          "order in app.json (this one goes FIRST).",
      );
    }

    for (const target of watchTargets) {
      for (const bc of target.props.buildConfigurationList.props.buildConfigurations) {
        bc.props.buildSettings.MARKETING_VERSION = version;
        if (buildNumber) {
          bc.props.buildSettings.CURRENT_PROJECT_VERSION = String(buildNumber);
        }
      }
      console.log(
        `[withWatchTargetVersion] ${target.props.productName ?? target.getDisplayName()}: ` +
          `MARKETING_VERSION=${version}` +
          (buildNumber ? ` CURRENT_PROJECT_VERSION=${buildNumber}` : ""),
      );
    }

    return config;
  });
};

module.exports = withWatchTargetVersion;
