import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Job } from "../../api";
import type { SceneDoc } from "../types";

vi.mock("../../api", () => ({ getJobStatus: vi.fn() }));
vi.mock("../composeApi", () => ({
  assetFileUrl: (id: string) => `/assets/${id}`,
  backendUrl: (path: string) => `http://backend${path}`,
  createScene: vi.fn(),
  exportScene: vi.fn(),
  getScene: vi.fn(),
  listAssets: vi.fn(),
  listScenes: vi.fn(),
  saveScene: vi.fn(),
  uploadAsset: vi.fn(),
}));
vi.mock("../ComposeViewport", () => ({ ComposeViewport: () => <div data-testid="viewport" /> }));

import { getJobStatus } from "../../api";
import * as composeApi from "../composeApi";
import { ComposePage } from "../ComposePage";

const api = vi.mocked(composeApi);
const jobStatus = vi.mocked(getJobStatus);

const DOC: SceneDoc = {
  version: 1,
  id: "s1",
  name: "Bahçe",
  viewUp: "y",
  objects: [
    {
      id: "base",
      kind: "splat",
      asset: "a1",
      name: "garden",
      role: "base",
      visible: true,
      transform: { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 },
    },
  ],
};

function job(patch: Partial<Job>): Job {
  return {
    id: "j1",
    scene: "compose-s1",
    status: "running",
    smoke_test: false,
    phase: { name: "compose", progress: 0.5, message: "birleştiriliyor", details: {} },
    overall_progress: 0.5,
    created_at: 0,
    started_at: 0,
    finished_at: null,
    error: null,
    ply_dir: null,
    download_url: null,
    ...patch,
  };
}

/** Let pending promises (and zero-delay timers) settle inside act. */
async function flush() {
  await act(async () => {
    for (let i = 0; i < 5; i++) await Promise.resolve();
  });
}

async function openScene() {
  render(<ComposePage active />);
  fireEvent.click(await screen.findByRole("button", { name: "Aç" }));
  await screen.findByTestId("viewport");
}

function makeDirty() {
  fireEvent.change(screen.getByDisplayValue("+Y"), { target: { value: "-y" } });
  expect(screen.getByText(/Bahçe •/)).toBeInTheDocument();
}

beforeEach(() => {
  vi.clearAllMocks();
  api.listScenes.mockResolvedValue([{ id: "s1", name: "Bahçe", updated_ts: 0, object_count: 1 }]);
  api.listAssets.mockResolvedValue([]);
  api.getScene.mockResolvedValue(DOC);
});

afterEach(() => {
  vi.useRealTimers();
  vi.restoreAllMocks();
});

describe("ComposePage", () => {
  it("saves once on Ctrl+S even when pressed twice quickly", async () => {
    let resolveSave: (doc: SceneDoc) => void = () => {};
    api.saveScene.mockImplementation(() => new Promise((r) => (resolveSave = r)));
    await openScene();
    makeDirty();

    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    await waitFor(() => expect(api.saveScene).toHaveBeenCalledTimes(1));
    expect(api.saveScene.mock.calls[0][0].viewUp).toBe("-y");

    await act(async () => resolveSave(DOC));
    expect(await screen.findByText("Kaydedildi")).toBeInTheDocument();
    expect(screen.queryByText(/Bahçe •/)).not.toBeInTheDocument();

    // Not dirty any more: Ctrl+S is a no-op.
    fireEvent.keyDown(window, { key: "s", ctrlKey: true });
    await new Promise((r) => setTimeout(r, 20));
    expect(api.saveScene).toHaveBeenCalledTimes(1);
  });

  it("keeps polling the export job after a failed poll and then shows the download link", async () => {
    await openScene();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    api.exportScene.mockResolvedValue({ job_id: "j1" });
    jobStatus
      .mockResolvedValueOnce(job({}))
      .mockRejectedValueOnce(new Error("Failed to fetch"))
      .mockResolvedValueOnce(job({ status: "completed", overall_progress: 1, download_url: "/download/j1" }));

    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    await flush();
    expect(screen.getByText(/50% · birleştiriliyor/)).toBeInTheDocument();

    // First poll fails…
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1000);
    });
    expect(jobStatus).toHaveBeenCalledTimes(2);
    // …then it retries (with backoff) instead of stopping.
    await act(async () => {
      await vi.advanceTimersByTimeAsync(5000);
    });
    expect(jobStatus).toHaveBeenCalledTimes(3);
    const link = screen.getByRole("link", { name: ".zip indir" });
    expect(link).toHaveAttribute("href", "http://backend/download/j1");
    expect(screen.getByRole("button", { name: "Export" })).not.toBeDisabled();
  });

  it("marks the export failed after repeated poll failures and re-enables Export", async () => {
    await openScene();
    vi.useFakeTimers({ toFake: ["setTimeout", "clearTimeout"] });
    api.exportScene.mockResolvedValue({ job_id: "j1" });
    jobStatus.mockResolvedValueOnce(job({})).mockRejectedValue(new Error("Failed to fetch"));

    fireEvent.click(screen.getByRole("button", { name: "Export" }));
    await flush();
    // Each retry is re-armed after a render, so advance one backoff step per act.
    for (let i = 0; i < 6; i++) {
      await act(async () => {
        await vi.advanceTimersByTimeAsync(20_000);
      });
    }
    expect(jobStatus).toHaveBeenCalledTimes(6);
    expect(screen.getByText(/Export başarısız: .*Failed to fetch/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Export" })).not.toBeDisabled();
  });

  it("asks for confirmation before closing a dirty scene", async () => {
    const confirm = vi.spyOn(window, "confirm").mockReturnValue(false);
    await openScene();
    makeDirty();

    fireEvent.click(screen.getByRole("button", { name: "← Sahneler" }));
    expect(confirm).toHaveBeenCalledTimes(1);
    expect(screen.getByTestId("viewport")).toBeInTheDocument();

    confirm.mockReturnValue(true);
    fireEvent.click(screen.getByRole("button", { name: "← Sahneler" }));
    expect(confirm).toHaveBeenCalledTimes(2);
    expect(await screen.findByRole("heading", { name: "Sahneler" })).toBeInTheDocument();
  });
});
