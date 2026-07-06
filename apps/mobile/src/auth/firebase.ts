import { Platform } from "react-native";
import { initializeApp, getApps, getApp } from "firebase/app";
import {
  initializeAuth,
  getAuth,
  browserLocalPersistence,
  type Auth,
  type Persistence,
} from "firebase/auth";
import * as firebaseAuth from "firebase/auth";
import { firebaseConfig } from "./firebaseConfig";
import { secureStorePersistence } from "./secureStorePersistence";

/**
 * getReactNativePersistence lives only in firebase/auth's React Native build,
 * which Metro resolves at runtime via the "react-native" export condition. It
 * is intentionally absent from the package's top-level TypeScript types (and
 * from the web/node builds), so we reach it through a locally-typed indirection
 * and only ever call it on native (see the Platform.OS guard below).
 */
const getReactNativePersistence = (
  firebaseAuth as unknown as {
    getReactNativePersistence: (storage: unknown) => Persistence;
  }
).getReactNativePersistence;

const app = getApps().length ? getApp() : initializeApp(firebaseConfig);

/**
 * A single Auth instance for the whole app. Persistence is platform-specific:
 *  - native (iOS/Android): expo-secure-store, so the session survives restarts
 *  - web: browserLocalPersistence (localStorage)
 *
 * initializeAuth must run exactly once; a Fast-Refresh reload re-imports this
 * module against the already-initialized app, so fall back to getAuth().
 */
function createAuth(): Auth {
  try {
    return initializeAuth(app, {
      persistence:
        Platform.OS === "web"
          ? browserLocalPersistence
          : getReactNativePersistence(secureStorePersistence),
    });
  } catch {
    // Already initialized (Fast Refresh / double import).
    return getAuth(app);
  }
}

export const auth = createAuth();
export { app };
