// App entry point.
//
// `@expo/metro-runtime` MUST be imported first: on web it installs the runtime
// that boots the app (AppRegistry.runApplication). Without it the Metro web
// bundle loads but never mounts — a silent blank screen. No-op on native.
//
// `registerRootComponent(App)` registers the root component. It calls
// `AppRegistry.registerComponent('main', () => App)` and, on web, mounts into
// the `#root` element from index.html. This is Expo's canonical entry; pointing
// package.json "main" at a bare component export (App.tsx) skipped it, which is
// why the web build rendered nothing.
import "@expo/metro-runtime";
import { registerRootComponent } from "expo";

import App from "./App";

registerRootComponent(App);
