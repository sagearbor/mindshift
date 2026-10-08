import React from "react";
import { Alert } from "react-native";
import { render, fireEvent, waitFor, act } from "@testing-library/react-native";
import * as DocumentPicker from "expo-document-picker";
import LibraryScreen from "../src/screens/LibraryScreen";
import LibraryPicker from "../src/components/LibraryPicker";
import {
  createLibraryNote,
  deleteLibraryItem,
  getLibraryItem,
  listLibrary,
  updateLibraryItem,
  uploadLibraryFile,
  type LibraryItem,
} from "../src/api/library";
import { createFsLibrarySelectionStore, normalizeSelection } from "../src/live/librarySelection";
import { MemoryFs } from "../src/recorder/memoryFs";

jest.mock("../src/api/library", () => ({
  ...jest.requireActual("../src/api/library"),
  listLibrary: jest.fn(),
  getLibraryItem: jest.fn(),
  createLibraryNote: jest.fn(),
  uploadLibraryFile: jest.fn(),
  updateLibraryItem: jest.fn(),
  deleteLibraryItem: jest.fn(),
}));

const NOTE: LibraryItem = {
  id: "11111111-1111-4111-8111-111111111111",
  title: "Pricing",
  kind: "note",
  chars: 40,
  status: "ready",
  error: null,
  created_at: "2026-10-07T00:00:00Z",
  updated_at: "2026-10-07T00:00:00Z",
  indexed: false,
};
const DOC: LibraryItem = {
  ...NOTE,
  id: "22222222-2222-4222-8222-222222222222",
  title: "Brief.pdf",
  kind: "document",
  chars: 52000,
  status: "processing",
};
const LIMITS = { max_file_bytes: 10485760, max_text_chars: 2097152, max_items: 50 };

const mList = listLibrary as jest.Mock;

function listOf(items: LibraryItem[], retrieval_available = true) {
  return { items, retrieval_available, limits: LIMITS };
}

beforeEach(() => {
  jest.clearAllMocks();
  mList.mockResolvedValue(listOf([NOTE]));
});

describe("LibraryScreen", () => {
  it("lists items with their status and says honestly when search is unavailable", async () => {
    mList.mockResolvedValue(listOf([NOTE, DOC], false));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId(`library-item-${NOTE.id}`)).toBeTruthy());
    expect(s.getByTestId(`library-status-${NOTE.id}`).props.children).toMatch(/Note · 40 chars · ready/);
    expect(s.getByTestId(`library-status-${DOC.id}`).props.children).toMatch(/Document · 52k chars · processing/);
    expect(s.getByTestId("library-retrieval-unavailable").props.children).toMatch(
      /Search unavailable.*large documents can't be used yet/,
    );
    await s.unmount();
  });

  it("hides the search notice when retrieval is available, and shows an empty state", async () => {
    mList.mockResolvedValue(listOf([]));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-empty")).toBeTruthy());
    expect(s.queryByTestId("library-retrieval-unavailable")).toBeNull();
    await s.unmount();
  });

  it("adds a note (title + text) and refreshes", async () => {
    (createLibraryNote as jest.Mock).mockResolvedValue(NOTE);
    mList.mockResolvedValueOnce(listOf([])).mockResolvedValue(listOf([NOTE]));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-empty")).toBeTruthy());
    await fireEvent.press(s.getByTestId("library-add-note"));
    await fireEvent.changeText(s.getByTestId("library-note-title"), " Pricing ");
    await fireEvent.changeText(s.getByTestId("library-note-text"), "Starter is 49 a month.");
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-note-save"));
    });
    expect(createLibraryNote).toHaveBeenCalledWith("Pricing", "Starter is 49 a month.");
    await waitFor(() => expect(s.getByTestId(`library-item-${NOTE.id}`)).toBeTruthy());
    expect(s.queryByTestId("library-note-form")).toBeNull();
    await s.unmount();
  });

  it("uploads a picked file and shows the server's refusal when it is too big", async () => {
    (DocumentPicker.getDocumentAsync as jest.Mock).mockResolvedValueOnce({
      canceled: false,
      assets: [{ uri: "file:///big.pdf", name: "big.pdf", mimeType: "application/pdf", size: 11e6 }],
    });
    const err = Object.assign(new Error("file is larger than the 10,485,760-byte (10 MB) limit"), { status: 413 });
    (uploadLibraryFile as jest.Mock).mockRejectedValueOnce(err);
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId(`library-item-${NOTE.id}`)).toBeTruthy());
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-upload"));
    });
    expect(DocumentPicker.getDocumentAsync).toHaveBeenCalledWith(
      expect.objectContaining({ multiple: false, type: expect.arrayContaining(["application/pdf"]) }),
    );
    expect(uploadLibraryFile).toHaveBeenCalledWith(
      expect.objectContaining({ uri: "file:///big.pdf", name: "big.pdf", mimeType: "application/pdf" }),
    );
    await waitFor(() => expect(s.getByTestId("library-action-error").props.children).toMatch(/10 MB/));
    await s.unmount();
  });

  it("a cancelled picker uploads nothing", async () => {
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId(`library-item-${NOTE.id}`)).toBeTruthy());
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-upload"));
    });
    expect(uploadLibraryFile).not.toHaveBeenCalled();
    await s.unmount();
  });

  it("renames an item", async () => {
    (updateLibraryItem as jest.Mock).mockResolvedValue({ ...NOTE, title: "Prices" });
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId(`library-rename-${NOTE.id}`)).toBeTruthy());
    await fireEvent.press(s.getByTestId(`library-rename-${NOTE.id}`));
    await fireEvent.changeText(s.getByTestId(`library-rename-input-${NOTE.id}`), "Prices");
    await act(async () => {
      await fireEvent.press(s.getByTestId(`library-rename-save-${NOTE.id}`));
    });
    expect(updateLibraryItem).toHaveBeenCalledWith(NOTE.id, { title: "Prices" });
    await s.unmount();
  });

  it("edits a note's text from its full preview, and refuses to edit a truncated one", async () => {
    (getLibraryItem as jest.Mock).mockResolvedValueOnce({
      ...NOTE, preview: "Starter is 49.", preview_truncated: false, chunk_count: 0,
    });
    (updateLibraryItem as jest.Mock).mockResolvedValue(NOTE);
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId(`library-edit-${NOTE.id}`)).toBeTruthy());
    await act(async () => {
      await fireEvent.press(s.getByTestId(`library-edit-${NOTE.id}`));
    });
    const input = s.getByTestId(`library-edit-input-${NOTE.id}`);
    expect(input.props.value).toBe("Starter is 49.");
    await fireEvent.changeText(input, "Starter is 59.");
    await act(async () => {
      await fireEvent.press(s.getByTestId(`library-edit-save-${NOTE.id}`));
    });
    expect(updateLibraryItem).toHaveBeenCalledWith(NOTE.id, { text: "Starter is 59." });

    (getLibraryItem as jest.Mock).mockResolvedValueOnce({
      ...NOTE, preview: "x".repeat(2000), preview_truncated: true, chunk_count: 0,
    });
    await waitFor(() => expect(s.getByTestId(`library-edit-${NOTE.id}`)).toBeTruthy());
    await act(async () => {
      await fireEvent.press(s.getByTestId(`library-edit-${NOTE.id}`));
    });
    expect(s.getByTestId(`library-edit-blocked-${NOTE.id}`)).toBeTruthy();
    expect(s.queryByTestId(`library-edit-input-${NOTE.id}`)).toBeNull();
    await s.unmount();
  });

  it("documents have no Edit text action", async () => {
    mList.mockResolvedValue(listOf([{ ...DOC, status: "ready" }]));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId(`library-rename-${DOC.id}`)).toBeTruthy());
    expect(s.queryByTestId(`library-edit-${DOC.id}`)).toBeNull();
    await s.unmount();
  });

  it("deletes only after confirming", async () => {
    const alertSpy = jest.spyOn(Alert, "alert").mockImplementation(() => {});
    (deleteLibraryItem as jest.Mock).mockResolvedValue({ deleted: true, id: NOTE.id });
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId(`library-delete-${NOTE.id}`)).toBeTruthy());
    await fireEvent.press(s.getByTestId(`library-delete-${NOTE.id}`));
    expect(deleteLibraryItem).not.toHaveBeenCalled();
    const buttons = alertSpy.mock.calls[0][2]!;
    await act(async () => {
      buttons.find((b) => b.style === "destructive")!.onPress!();
    });
    expect(deleteLibraryItem).toHaveBeenCalledWith(NOTE.id);
    alertSpy.mockRestore();
    await s.unmount();
  });

  it("shows the load error with a retry", async () => {
    mList.mockRejectedValueOnce(new Error("Library request failed (503)")).mockResolvedValue(listOf([NOTE]));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-load-error")).toBeTruthy());
    await act(async () => {
      await fireEvent.press(s.getByText("Try again"));
    });
    await waitFor(() => expect(s.getByTestId(`library-item-${NOTE.id}`)).toBeTruthy());
    await s.unmount();
  });
});

describe("LibraryPicker (Live Coach)", () => {
  it("toggles items, ignores unready ones, prunes deleted ids, and opens the manager", async () => {
    mList.mockResolvedValue(listOf([NOTE, DOC], false));
    const onChange = jest.fn();
    const onManage = jest.fn();
    const GONE = "33333333-3333-4333-8333-333333333333";
    const s = await render(<LibraryPicker selectedIds={[GONE]} onChange={onChange} onManage={onManage} />);
    expect(s.getByTestId("library-picker-toggle")).toBeTruthy();
    expect(listLibrary).not.toHaveBeenCalled(); // collapsed: no fetch
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-picker-toggle"));
    });
    await waitFor(() => expect(s.getByTestId(`library-pick-${NOTE.id}`)).toBeTruthy());
    expect(onChange).toHaveBeenCalledWith([]); // the deleted id is dropped
    expect(s.getByTestId("library-picker-retrieval-unavailable")).toBeTruthy();
    await s.rerender(<LibraryPicker selectedIds={[]} onChange={onChange} onManage={onManage} />);
    await fireEvent.press(s.getByTestId(`library-pick-${NOTE.id}`));
    expect(onChange).toHaveBeenLastCalledWith([NOTE.id]);
    expect(s.getByTestId(`library-pick-${DOC.id}`).props.accessibilityState.disabled).toBe(true);
    await fireEvent.press(s.getByTestId("library-picker-manage"));
    expect(onManage).toHaveBeenCalled();
    await s.unmount();
  });

  it("says when the server ignored selected items", async () => {
    const s = await render(<LibraryPicker selectedIds={["a", "b"]} onChange={jest.fn()} ignoredCount={2} />);
    expect(s.getByTestId("library-picker-ignored")).toBeTruthy();
    await s.unmount();
  });
});

describe("library selection store", () => {
  it("remembers ids on the device, deduped and capped at 20", () => {
    const fs = new MemoryFs();
    const uri = `${fs.documentDirUri()}/live-library-selection.json`;
    const store = createFsLibrarySelectionStore(fs, uri);
    expect(store.load()).toEqual([]);
    store.save(["a", "b", "a"]);
    expect(createFsLibrarySelectionStore(fs, uri).load()).toEqual(["a", "b"]);
    expect(normalizeSelection(Array.from({ length: 30 }, (_, i) => `i${i}`))).toHaveLength(20);
    store.save([]);
    expect(fs.exists(uri)).toBe(false);
  });
});
