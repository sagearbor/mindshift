import * as SecureStore from "expo-secure-store";
import { useUploadPrefsStore } from "../src/store/uploadPrefsStore";

const mockGet = SecureStore.getItemAsync as jest.Mock;
const mockSet = SecureStore.setItemAsync as jest.Mock;

const STORAGE_KEY = "mindshift.upload.sendOriginalQuality";

beforeEach(() => {
  mockGet.mockReset();
  mockSet.mockReset();
  mockGet.mockResolvedValue(null);
  mockSet.mockResolvedValue(undefined);
  useUploadPrefsStore.setState({ sendOriginalQuality: false, hydrated: false });
});

describe("uploadPrefsStore", () => {
  it("defaults to compress (Send original quality OFF), not yet hydrated", () => {
    const s = useUploadPrefsStore.getState();
    expect(s.sendOriginalQuality).toBe(false);
    expect(s.hydrated).toBe(false);
  });

  it("setSendOriginalQuality updates in memory and persists to secure storage", () => {
    useUploadPrefsStore.getState().setSendOriginalQuality(true);
    expect(useUploadPrefsStore.getState().sendOriginalQuality).toBe(true);
    expect(mockSet).toHaveBeenCalledWith(STORAGE_KEY, "1");

    useUploadPrefsStore.getState().setSendOriginalQuality(false);
    expect(useUploadPrefsStore.getState().sendOriginalQuality).toBe(false);
    expect(mockSet).toHaveBeenCalledWith(STORAGE_KEY, "0");
  });

  it("hydrate loads a persisted ON value and marks hydrated", async () => {
    mockGet.mockResolvedValueOnce("1");
    await useUploadPrefsStore.getState().hydrate();
    const s = useUploadPrefsStore.getState();
    expect(mockGet).toHaveBeenCalledWith(STORAGE_KEY);
    expect(s.sendOriginalQuality).toBe(true);
    expect(s.hydrated).toBe(true);
  });

  it("hydrate keeps the OFF default when nothing is stored", async () => {
    mockGet.mockResolvedValueOnce(null);
    await useUploadPrefsStore.getState().hydrate();
    const s = useUploadPrefsStore.getState();
    expect(s.sendOriginalQuality).toBe(false);
    expect(s.hydrated).toBe(true);
  });

  it("hydrate survives a storage read failure (falls back to OFF, still hydrated)", async () => {
    mockGet.mockRejectedValueOnce(new Error("keystore unavailable"));
    await useUploadPrefsStore.getState().hydrate();
    const s = useUploadPrefsStore.getState();
    expect(s.sendOriginalQuality).toBe(false);
    expect(s.hydrated).toBe(true);
  });
});
