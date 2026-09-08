/**
 * Which number the Replay chart plots on its y-axis (heat / loudness / pitch
 * / speech rate) — remembered PER DEVICE, so the owner who switches to
 * "Loudness" sees loudness on the next recording too. Not per account: it is
 * a viewing preference, not data.
 *
 * Same cross-platform persistence as live/scoreboardPrefs.ts (expo-secure-
 * store on native, localStorage on web), fail-open to the default.
 */
import { Platform } from "react-native";
import * as SecureStore from "expo-secure-store";
import { DEFAULT_CHART_METRIC, isChartMetric, type ChartMetric } from "./chartMetrics";

export const CHART_METRIC_KEY = "mindshift.replayChartMetric.v1";

function webStorage(): Storage | null {
  try {
    const g = globalThis as { localStorage?: Storage };
    return g.localStorage ?? null;
  } catch {
    return null;
  }
}

export async function loadChartMetric(): Promise<ChartMetric> {
  try {
    const raw =
      Platform.OS === "web"
        ? webStorage()?.getItem(CHART_METRIC_KEY) ?? null
        : await SecureStore.getItemAsync(CHART_METRIC_KEY);
    return isChartMetric(raw) ? raw : DEFAULT_CHART_METRIC;
  } catch {
    return DEFAULT_CHART_METRIC;
  }
}

export async function saveChartMetric(metric: ChartMetric): Promise<void> {
  try {
    if (Platform.OS === "web") {
      webStorage()?.setItem(CHART_METRIC_KEY, metric);
    } else {
      await SecureStore.setItemAsync(CHART_METRIC_KEY, metric);
    }
  } catch {
    // Fail-open: the choice still applies to this screen.
  }
}
