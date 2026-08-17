import { useEffect, useState } from "react";

import { getGrowth, type GrowthResult } from "../api/client";

export interface GrowthPreviewState {
  /** True until the first fetch settles (success or failure). */
  loading: boolean;
  /** The fetched growth data, or null while loading / on any failure. Never
   *  throws — a failed fetch just means "nothing to show", same fail-open
   *  contract every self-fetching glanceable surface in this app uses. */
  result: GrowthResult | null;
}

/**
 * Self-fetching "Your growth" data, shared by every glanceable preview that
 * wants it (the home strip; Task N4's growth home-box). One `getGrowth()`
 * call per mounted consumer — deliberately not a shared cache/store: growth
 * data is cheap, per-user, and each consumer already unmounts/remounts
 * independently, so a cache would be premature plumbing for a same-screen
 * read that happens at most a couple of times per Home visit.
 */
export function useGrowthPreview(): GrowthPreviewState {
  const [result, setResult] = useState<GrowthResult | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    getGrowth()
      .then((r) => {
        if (!cancelled) setResult(r);
      })
      .catch(() => {
        if (!cancelled) setResult(null);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  return { loading, result };
}
