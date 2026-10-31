import React from "react";
import { render, fireEvent, waitFor, act } from "@testing-library/react-native";
import * as DocumentPicker from "expo-document-picker";
import LibraryScreen from "../src/screens/LibraryScreen";
import LibraryPicker from "../src/components/LibraryPicker";
import {
  createLibraryProject,
  deleteLibraryProject,
  listLibrary,
  updateLibraryItem,
  updateLibraryProject,
  uploadLibraryFile,
  type LibraryItem,
  type LibraryProject,
} from "../src/api/library";
import { MAX_LIBRARY_SELECTION } from "../src/live/librarySelection";
import { showAlert } from "../src/utils/showAlert";

jest.mock("../src/api/library", () => ({
  ...jest.requireActual("../src/api/library"),
  listLibrary: jest.fn(),
  createLibraryProject: jest.fn(),
  updateLibraryProject: jest.fn(),
  deleteLibraryProject: jest.fn(),
  updateLibraryItem: jest.fn(),
  uploadLibraryFile: jest.fn(),
}));
jest.mock("../src/utils/showAlert", () => ({ showAlert: jest.fn() }));

const T = "2026-10-31T00:00:00Z";
const LIMITS = { max_file_bytes: 1, max_text_chars: 1, max_items: 50 };
const P = (id: string, name: string, parent_id: string | null = null): LibraryProject => ({
  id, name, parent_id, created_at: T, updated_at: T,
});
const I = (id: string, project_id: string | null = null, title = id): LibraryItem => ({
  id, title, kind: "note", chars: 5, status: "ready", error: null,
  created_at: T, updated_at: T, indexed: false, project_id,
});
const mList = listLibrary as jest.Mock;
const list = (items: LibraryItem[], projects: LibraryProject[] = []) => ({
  items, projects, retrieval_available: true, limits: LIMITS,
});

beforeEach(() => {
  jest.clearAllMocks();
});

describe("LibraryScreen projects", () => {
  const projects = [P("pa", "Clients"), P("pb", "Acme", "pa")];
  const items = [I("root1", null, "Loose"), I("in-a", "pa", "Pricing"), I("in-b", "pb", "Contract")];

  it("shows a collapsible folder tree; root items stay at top level", async () => {
    mList.mockResolvedValue(list(items, projects));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-project-pa")).toBeTruthy());
    expect(s.getByTestId("library-item-root1")).toBeTruthy();
    // Folders start expanded: nested content visible.
    expect(s.getByTestId("library-item-in-a")).toBeTruthy();
    expect(s.getByTestId("library-project-pb")).toBeTruthy();
    expect(s.getByTestId("library-item-in-b")).toBeTruthy();
    await fireEvent.press(s.getByTestId("library-project-toggle-pa"));
    expect(s.queryByTestId("library-item-in-a")).toBeNull();
    expect(s.queryByTestId("library-project-pb")).toBeNull();
    expect(s.getByTestId("library-item-root1")).toBeTruthy();
    await s.unmount();
  });

  it("creates a project (and a sub-project) by name", async () => {
    mList.mockResolvedValue(list([], projects));
    (createLibraryProject as jest.Mock).mockResolvedValue(P("new", "Ops"));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-new-project")).toBeTruthy());
    await fireEvent.press(s.getByTestId("library-new-project"));
    await fireEvent.changeText(s.getByTestId("library-project-name-input"), "  Ops ");
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-project-name-save"));
    });
    expect(createLibraryProject).toHaveBeenCalledWith("Ops", null);
    await fireEvent.press(s.getByTestId("library-project-add-pa"));
    await fireEvent.changeText(s.getByTestId("library-project-name-input"), "Sub");
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-project-name-save"));
    });
    expect(createLibraryProject).toHaveBeenLastCalledWith("Sub", "pa");
    await s.unmount();
  });

  it("moves an item via 'Move to…' (including back to top level) and a project into another", async () => {
    mList.mockResolvedValue(list(items, projects));
    (updateLibraryItem as jest.Mock).mockResolvedValue(items[0]);
    (updateLibraryProject as jest.Mock).mockResolvedValue(projects[0]);
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-move-root1")).toBeTruthy());
    await fireEvent.press(s.getByTestId("library-move-root1"));
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-move-target-pb"));
    });
    expect(updateLibraryItem).toHaveBeenCalledWith("root1", { project_id: "pb" });
    await fireEvent.press(s.getByTestId("library-move-in-a"));
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-move-target-root"));
    });
    expect(updateLibraryItem).toHaveBeenLastCalledWith("in-a", { project_id: null });
    // A project cannot be offered itself or its own sub-project as a target.
    await fireEvent.press(s.getByTestId("library-project-move-pa"));
    expect(s.queryByTestId("library-move-target-pa")).toBeNull();
    expect(s.queryByTestId("library-move-target-pb")).toBeNull();
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-move-target-root"));
    });
    expect(updateLibraryProject).toHaveBeenCalledWith("pa", { parent_id: null });
    await s.unmount();
  });

  it("deleting a non-empty project offers an explicit 'move contents up' choice; empty deletes plainly", async () => {
    mList.mockResolvedValue(list(items, [...projects, P("pe", "Empty")]));
    (deleteLibraryProject as jest.Mock).mockResolvedValue({ deleted: true, id: "x" });
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-project-delete-pa")).toBeTruthy());
    await fireEvent.press(s.getByTestId("library-project-delete-pa"));
    const [, , buttons] = (showAlert as jest.Mock).mock.calls[0];
    expect(buttons.map((b: { text: string }) => b.text)).toEqual(["Cancel", "Move contents up & delete"]);
    expect(deleteLibraryProject).not.toHaveBeenCalled();
    await act(async () => {
      await buttons[1].onPress();
    });
    expect(deleteLibraryProject).toHaveBeenCalledWith("pa", "move_to_parent");

    (showAlert as jest.Mock).mockClear();
    await fireEvent.press(s.getByTestId("library-project-delete-pe"));
    const [, , b2] = (showAlert as jest.Mock).mock.calls[0];
    await act(async () => {
      await b2[b2.length - 1].onPress();
    });
    expect(deleteLibraryProject).toHaveBeenLastCalledWith("pe", undefined);
    await s.unmount();
  });

  it("an uploaded file with no title is sent with its picked name (title fix is in the client)", async () => {
    mList.mockResolvedValue(list([], []));
    (DocumentPicker.getDocumentAsync as jest.Mock).mockResolvedValue({
      canceled: false,
      assets: [{ uri: "file:///cache/uuid.pdf", name: "Brief.pdf", mimeType: "application/pdf" }],
    });
    (uploadLibraryFile as jest.Mock).mockResolvedValue(I("n"));
    const s = await render(<LibraryScreen onBack={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-upload")).toBeTruthy());
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-upload"));
    });
    expect((uploadLibraryFile as jest.Mock).mock.calls[0][0]).toMatchObject({ name: "Brief.pdf" });
    await s.unmount();
  });
});

describe("LibraryPicker: zero-selected warning and project ticking", () => {
  it("warns '0 selected' + plain text when the library has files and none are selected, even collapsed", async () => {
    mList.mockResolvedValue(list([I("a"), I("b")]));
    const s = await render(<LibraryPicker selectedIds={[]} onChange={jest.fn()} />);
    await waitFor(() => expect(s.getByTestId("library-picker-none-selected")).toBeTruthy());
    expect(s.getByTestId("library-picker-none-selected").props.children).toBe(
      "No files in use — tick the ones you want",
    );
    expect(s.getByTestId("library-picker-toggle")).toHaveTextContent(/0 selected/);
    expect(s.queryByTestId("library-pick-a")).toBeNull(); // still collapsed
    await s.unmount();
  });

  it("stays quiet with an empty library, and clears the warning once something is selected", async () => {
    mList.mockResolvedValue(list([]));
    const s = await render(<LibraryPicker selectedIds={[]} onChange={jest.fn()} />);
    await waitFor(() => expect(listLibrary).toHaveBeenCalled());
    expect(s.queryByTestId("library-picker-none-selected")).toBeNull();
    expect(s.getByTestId("library-picker-toggle")).not.toHaveTextContent(/selected/);
    await s.unmount();

    mList.mockResolvedValue(list([I("a")]));
    const s2 = await render(<LibraryPicker selectedIds={["a"]} onChange={jest.fn()} />);
    await waitFor(() => expect(s2.getByTestId("library-picker-toggle")).toHaveTextContent(/1 selected/));
    expect(s2.queryByTestId("library-picker-none-selected")).toBeNull();
    await s2.unmount();
  });

  it("ticking a project selects every file in it and its sub-projects", async () => {
    mList.mockResolvedValue(
      list(
        [I("x", "pa"), I("y", "pb"), I("z", null)],
        [P("pa", "Clients"), P("pb", "Acme", "pa")],
      ),
    );
    const onChange = jest.fn();
    const s = await render(<LibraryPicker selectedIds={["z"]} onChange={onChange} />);
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-picker-toggle"));
    });
    await waitFor(() => expect(s.getByTestId("library-pick-project-pa")).toBeTruthy());
    await fireEvent.press(s.getByTestId("library-pick-project-pa"));
    expect(onChange).toHaveBeenLastCalledWith(["z", "x", "y"]);
    await s.rerender(<LibraryPicker selectedIds={["z", "x", "y"]} onChange={onChange} />);
    expect(s.getByTestId("library-pick-project-pa").props.accessibilityState.checked).toBe(true);
    await fireEvent.press(s.getByTestId("library-pick-project-pa"));
    expect(onChange).toHaveBeenLastCalledWith(["z"]);
    await s.unmount();
  });

  it("shows a clear message and changes nothing when a project exceeds the session cap", async () => {
    const many = Array.from({ length: MAX_LIBRARY_SELECTION + 1 }, (_, n) => I(`m${n}`, "big"));
    mList.mockResolvedValue(list(many, [P("big", "Everything")]));
    const onChange = jest.fn();
    const s = await render(<LibraryPicker selectedIds={[]} onChange={onChange} />);
    await act(async () => {
      await fireEvent.press(s.getByTestId("library-picker-toggle"));
    });
    await waitFor(() => expect(s.getByTestId("library-pick-project-big")).toBeTruthy());
    onChange.mockClear();
    await fireEvent.press(s.getByTestId("library-pick-project-big"));
    expect(onChange).not.toHaveBeenCalled();
    expect(s.getByTestId("library-picker-limit").props.children).toMatch(
      new RegExp(`"Everything".*${MAX_LIBRARY_SELECTION + 1}.*at most ${MAX_LIBRARY_SELECTION}`),
    );
    await s.unmount();
  });
});
