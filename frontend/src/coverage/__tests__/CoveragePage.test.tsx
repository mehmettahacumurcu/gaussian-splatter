import { act, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { Job } from "../../api";
import type { CoverageDetail, CoverageView } from "../coverageApi";

interface ViewerProps {
  normalUrl: string;
  heatUrl: string;
  view: CoverageView | null;
}
const viewer = vi.hoisted(() => ({ props: null as ViewerProps | null }));

vi.mock("../../api", () => ({ getJobStatus: vi.fn() }));
vi.mock("../coverageApi", () => ({
  coveragePhotoUrl: (id: string, i: number) => `/photo/${id}/${i}`,
  coverageSplatUrl: (id: string, which: string) => `/splat/${id}/${which}`,
  deleteCoverage: vi.fn(),
  getCoverage: vi.fn(),
  listCoverage: vi.fn(),
  startCoverage: vi.fn(),
}));
vi.mock("../DualSplatViewer", () => ({ DualSplatViewer: (props: ViewerProps) => {
  viewer.props = props;
  return <div data-testid="viewer" />;
} }));

import { getJobStatus } from "../../api";
import * as coverageApi from "../coverageApi";
import { CoveragePage } from "../CoveragePage";

const api = vi.mocked(coverageApi);
const jobStatus = vi.mocked(getJobStatus);

const view = (index: number, name: string, test: boolean, weak: number): CoverageView => ({
  index, name, test, weak, score: 1 - weak, c2w: Array.from({ length: 16 }, (_, i) => (i % 5 ? 0 : 1)), fov_y: 40, aspect: 1.5,
});

const DETAIL: CoverageDetail = {
  id: "room2-abc",
  name: "ROOM2",
  created_at: 1,
  ply_path: "E:/room2/splat.ply",
  sparse_dir: "E:/room2/sparse/0",
  images_dir: "E:/room2/images",
  cameras: 4,
  test_views: 2,
  up: [0, 0, 1],
  seconds: 30,
  summary: { splats: 3000, opaque_splats: 1200, unseen_share: 0.062, weak_share: 0.07, count_percentiles: { "50": 168 }, good_count: 40, weak_score: 0.5 },
  views: [
    view(0, "eval_00001.png", true, 0.1),
    view(1, "eval_00033.png", true, 0.4),
    view(2, "train_00001.png", false, 0.9),
    view(3, "train_00002.png", false, 0.05),
  ],
};

const job = (patch: Partial<Job>): Job => ({
  id: "j1", scene: "coverage-room2-abc", status: "running", smoke_test: false,
  phase: { name: "coverage_scan", progress: 0.5, message: "500/1008 kamera tarandı", details: {} },
  overall_progress: 0.4, created_at: 0, started_at: 0, finished_at: null, error: null, ply_dir: null, download_url: null,
  ...patch,
});

beforeEach(() => {
  viewer.props = null;
  api.listCoverage.mockResolvedValue([DETAIL]);
  api.getCoverage.mockResolvedValue(DETAIL);
});

afterEach(() => {
  vi.useRealTimers();
  vi.clearAllMocks();
});

describe("CoveragePage", () => {
  it("opens a result, shows stats and jumps to the weakest test view first", async () => {
    render(<CoveragePage active />);
    fireEvent.click(await screen.findByText("ROOM2", { selector: "strong" }));
    await screen.findByTestId("viewer");
    expect(viewer.props?.normalUrl).toBe("/splat/room2-abc/normal");
    expect(viewer.props?.heatUrl).toBe("/splat/room2-abc/heat");
    await waitFor(() => expect(viewer.props?.view?.name).toBe("eval_00033.png"));
    expect(screen.getByText("%7.0")).toBeTruthy();
    expect(screen.getByRole("img", { name: "eval_00033.png" }).getAttribute("src")).toBe("/photo/room2-abc/1");

    // Only test views are listed until "eğitim" is ticked.
    expect(screen.queryByText("train_00001.png")).toBeNull();
    fireEvent.click(screen.getByRole("checkbox"));
    await waitFor(() => expect(viewer.props?.view?.name).toBe("train_00001.png"));
    fireEvent.click(screen.getByText("▶"));
    expect(viewer.props?.view?.name).toBe("train_00002.png");
  });

  it("keeps the viewer unmounted while the tab is hidden", async () => {
    const { rerender } = render(<CoveragePage active={false} />);
    fireEvent.click(await screen.findByText("ROOM2", { selector: "strong" }));
    await waitFor(() => expect(api.getCoverage).toHaveBeenCalled());
    expect(screen.queryByTestId("viewer")).toBeNull();
    rerender(<CoveragePage active />);
    expect(await screen.findByTestId("viewer")).toBeTruthy();
  });

  it("starts a scan, shows progress and opens the result when the job completes", async () => {
    vi.useFakeTimers({ shouldAdvanceTime: true });
    api.listCoverage.mockResolvedValueOnce([]);
    api.startCoverage.mockResolvedValue({ job_id: "j1", coverage_id: "room2-abc", ply_path: "p", sparse_dir: "s" });
    jobStatus.mockResolvedValueOnce(job({})).mockResolvedValueOnce(job({ status: "completed", overall_progress: 1 }));
    render(<CoveragePage active />);

    const button = screen.getByRole("button", { name: "Kapsamayı hesapla" });
    expect((button as HTMLButtonElement).disabled).toBe(true);
    fireEvent.change(screen.getByPlaceholderText(/ROOM2/), { target: { value: "  G:/runs/ROOM2  " } });
    fireEvent.click(button);
    await waitFor(() => expect(api.startCoverage).toHaveBeenCalledWith({ path: "G:/runs/ROOM2", ply_path: "", sparse_dir: "" }));

    await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
    expect((await screen.findByRole("status")).textContent).toContain("500/1008 kamera tarandı");
    await act(async () => { await vi.advanceTimersByTimeAsync(1100); });
    await screen.findByTestId("viewer");
    expect(api.getCoverage).toHaveBeenCalledWith("room2-abc");
  });

  it("shows the backend's reason when a scan cannot start", async () => {
    api.startCoverage.mockRejectedValue(new Error('HTTP 400 Bad Request: {"detail":"COLMAP modeli bulunamadı (sparse/0)."}'));
    render(<CoveragePage active />);
    fireEvent.change(screen.getByPlaceholderText(/ROOM2/), { target: { value: "G:/x" } });
    fireEvent.click(screen.getByRole("button", { name: "Kapsamayı hesapla" }));
    expect((await screen.findByRole("alert")).textContent).toBe("COLMAP modeli bulunamadı (sparse/0).");
  });

  it("deletes a result after confirmation", async () => {
    vi.spyOn(window, "confirm").mockReturnValue(true);
    api.deleteCoverage.mockResolvedValue({ ok: true });
    render(<CoveragePage active />);
    fireEvent.click(await screen.findByTitle("Sil"));
    await waitFor(() => expect(api.deleteCoverage).toHaveBeenCalledWith("room2-abc"));
    expect(api.listCoverage).toHaveBeenCalledTimes(2);
  });
});
