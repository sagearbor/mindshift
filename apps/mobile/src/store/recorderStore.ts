import { create } from "zustand";

/**
 * A video the user just recorded in-app, handed off to the Session screen's
 * existing upload flow. Native-only (in-app recording is mobile-only), so unlike
 * SessionScreen's PickedRecording there's no web `File` — just the local URI and
 * the metadata the upload path needs (`size` lets SessionScreen pick the
 * chunked-vs-direct route honestly).
 */
export interface RecordedFile {
  uri: string;
  name: string;
  mimeType: string;
  size?: number;
}

/**
 * A one-shot hand-off channel between RecordScreen and SessionScreen. The
 * screen-union navigation in App unmounts SessionScreen while RecordScreen is
 * pushed, so a prop can't survive the round trip — this tiny store carries the
 * freshly-recorded file across, and SessionScreen consumes (and clears) it on
 * mount. Kept separate from sessionStore so the transcript store stays focused.
 */
interface RecorderState {
  /** The recorded file waiting to be picked up by the Session screen, or null. */
  pendingFile: RecordedFile | null;
  /** Stash a recorded file for hand-off (RecordScreen → Session). Pass null to
   *  clear after consumption. */
  setPendingFile: (file: RecordedFile | null) => void;
}

export const useRecorderStore = create<RecorderState>((set) => ({
  pendingFile: null,
  setPendingFile: (file) => set({ pendingFile: file }),
}));
