import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import type { Job } from "../../api";

const api = vi.hoisted(() => ({ listJobs: vi.fn(), cancelJob: vi.fn() }));
const spirula = vi.hoisted(() => ({ getRunLog: vi.fn() }));
vi.mock("../../api", () => api);
vi.mock("../../newsplat/spirulaApi", () => spirula);

import { JobsPage } from "../JobsPage";

function job(over: Partial<Job>): Job {
  return {
    id: "j1", scene: "spirula-20261022-130501-oda2", status: "running", smoke_test: false,
    phase: { name: "spirula_train", progress: 0.5, message: "25.000/50.000 adım", details: {} },
    overall_progress: 0.6, created_at: 1_760_000_000, started_at: 1_760_000_000, finished_at: null,
    error: null, ply_dir: null, download_url: null, ...over,
  };
}

const noop = () => {};

beforeEach(() => {
  api.listJobs.mockReset();
  spirula.getRunLog.mockReset();
  spirula.getRunLog.mockResolvedValue({ label: "train", lines: ["step 25000/50000 splats 2400000"] });
});

describe("JobsPage", () => {
  it("shows training stats, the live Spirula viewer and the run log", async () => {
    api.listJobs.mockResolvedValue({ total: 1, jobs: [job({
      phase: { name: "spirula_train", progress: 0.5, message: "25.000/50.000 adım",
        details: { viewer_url: "http://127.0.0.1:7007", step: 25000, total: 50000, splats: 2400000, psnr: 27.5, eta: "00:21:00" } },
    })] });
    render(<JobsPage selectedId={null} onSelect={noop} onViewJob={noop} onCoverage={noop} onNewSplat={noop} />);
    expect(await screen.findByRole("heading", { name: "oda2" })).toBeInTheDocument();
    expect(screen.getByText("2.400.000")).toBeInTheDocument();
    expect(screen.getByText("27.50 dB")).toBeInTheDocument();
    expect(screen.getByTitle("Spirula canlı eğitim önizlemesi")).toHaveAttribute("src", "http://127.0.0.1:7007");
    expect(screen.getByText("Eğitim")).toHaveClass("jb-step-now");
    expect(screen.getByText("SfM")).toHaveClass("jb-step-done");
    await waitFor(() => expect(spirula.getRunLog).toHaveBeenCalledWith("20261022-130501-oda2", 120));
    expect(await screen.findByText(/splats 2400000/)).toBeInTheDocument();
  });

  it("waits with the preview until training starts", async () => {
    api.listJobs.mockResolvedValue({ total: 1, jobs: [job({ phase: { name: "spirula_sfm", progress: 0.2, message: "", details: {} } })] });
    render(<JobsPage selectedId={null} onSelect={noop} onViewJob={noop} onCoverage={noop} onNewSplat={noop} />);
    expect(await screen.findByText(/eğitim başlayınca burada açılır/)).toBeInTheDocument();
  });

  it("offers viewer and coverage for a finished Spirula run", async () => {
    const onCoverage = vi.fn();
    const onViewJob = vi.fn();
    const done = job({
      status: "completed", finished_at: 1_760_003_600, overall_progress: 1, download_url: "/spirula/runs/r/splat",
      result: { train: { step: 30000, total: 30000, splats: 1000000, psnr: 29.12 }, workspace:"E:\\data\\spirula\\r", splat_path: "E:\\data\\spirula\\r\\result\\splat.ply",
        sfm: { selected_reconstruction: "E:\\data\\spirula\\r\\dataset\\sparse\\0", registered_images: 370, input_images: 380 } },
    });
    api.listJobs.mockResolvedValue({ total: 1, jobs: [done] });
    render(<JobsPage selectedId="j1" onSelect={noop} onViewJob={onViewJob} onCoverage={onCoverage} onNewSplat={noop} />);
    fireEvent.click(await screen.findByRole("button", { name: "Kapsama analizi" }));
    expect(onCoverage).toHaveBeenCalledWith({
      path: "E:\\data\\spirula\\r", plyPath: "E:\\data\\spirula\\r\\result\\splat.ply",
      sparseDir: "E:\\data\\spirula\\r\\dataset\\sparse\\0",
    });
    fireEvent.click(screen.getByRole("button", { name: "Viewer'da aç" }));
    expect(onViewJob).toHaveBeenCalledWith(done);
    expect(screen.getByText("SfM: 370/380 kare kayıtlı")).toBeInTheDocument();
    expect(screen.getByText("29.12 dB")).toBeInTheDocument();
    expect(screen.getByText("1.000.000")).toBeInTheDocument();
  });

  it("shows the error message and folds the traceback", async () => {
    api.listJobs.mockResolvedValue({ total: 1, jobs: [job({
      status: "failed", error: "SfM kapsamı yetersiz: 40% kare kayıtlı\n\nTraceback (most recent call last):\n  ...",
    })] });
    render(<JobsPage selectedId={null} onSelect={noop} onViewJob={noop} onCoverage={noop} onNewSplat={noop} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("SfM kapsamı yetersiz");
    expect(screen.getByText("Teknik ayrıntı")).toBeInTheDocument();
  });

  it("shows a cancelled run as cancelled, not as an error", async () => {
    api.listJobs.mockResolvedValue({ total: 1, jobs: [job({ status: "failed", cancel_requested: true, error: "İptal edildi\n\nTraceback ..." })] });
    render(<JobsPage selectedId={null} onSelect={noop} onViewJob={noop} onCoverage={noop} onNewSplat={noop} />);
    expect(await screen.findByText("İptal edildi.")).toBeInTheDocument();
    expect(screen.queryByRole("alert")).not.toBeInTheDocument();
  });

  it("selects a job from the list", async () => {
    const onSelect = vi.fn();
    api.listJobs.mockResolvedValue({ total: 2, jobs: [job({}), job({ id: "j2", scene: "coverage-room2", status: "completed" })] });
    render(<JobsPage selectedId={null} onSelect={onSelect} onViewJob={noop} onCoverage={noop} onNewSplat={noop} />);
    fireEvent.click(await screen.findByRole("button", { name: /room2/ }));
    expect(onSelect).toHaveBeenCalledWith("j2");
  });
});
