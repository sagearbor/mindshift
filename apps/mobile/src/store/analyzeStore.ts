import { create } from "zustand";
import type { RelationshipId } from "../components/RelationshipPicker";

/**
 * UI state for the Analyze flow that should survive screen re-mounts: the
 * relationship picker's "smart default" is simply the last relationship the
 * user chose this session (first-run default: partners). Kept in a store —
 * not component state — because the screen-union navigation unmounts
 * AnalyzeScreen whenever a sub-screen (record / recordings / dynamics) is
 * pushed on top of it.
 */
interface AnalyzeUiState {
  relationship: RelationshipId;
  setRelationship: (relationship: RelationshipId) => void;
}

export const useAnalyzeStore = create<AnalyzeUiState>((set) => ({
  relationship: "partners",
  setRelationship: (relationship) => set({ relationship }),
}));
