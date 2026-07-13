import { create } from "zustand";
import type { RelationshipId } from "../components/RelationshipPicker";

/**
 * UI state for the Analyze flow that should survive screen re-mounts.
 *
 * The relationship picker is OPTIONAL: it starts with nothing selected
 * (`null`), in which case NO relationship context is sent and the analysis
 * infers the relationship from the conversation itself. Once the user taps a
 * pill, that choice becomes the "smart default" — remembered here and
 * pre-selected on later opens. Tapping the selected pill deselects it (back
 * to null / infer mode). Kept in a store — not component state — because the
 * screen-union navigation unmounts AnalyzeScreen whenever a sub-screen
 * (record / recordings / dynamics) is pushed on top of it.
 */
interface AnalyzeUiState {
  relationship: RelationshipId | null;
  setRelationship: (relationship: RelationshipId | null) => void;
}

export const useAnalyzeStore = create<AnalyzeUiState>((set) => ({
  relationship: null,
  setRelationship: (relationship) => set({ relationship }),
}));
