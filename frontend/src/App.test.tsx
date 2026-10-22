import { fireEvent, render, screen } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

vi.mock("./api", () => ({
  getHealth: vi.fn(() => new Promise(() => {})),
  getSplatInfo: vi.fn(),
  listJobs: vi.fn().mockResolvedValue({ jobs: [], total: 0 }),
  listDiskScenes: vi.fn(),
}));
vi.mock("./notebook/NotebookGeneratorPanel", () => ({
  NotebookGeneratorPanel: () => (
    <div data-testid="notebook-panel">
      <button type="button">Generate fake notebook</button>
    </div>
  ),
}));
vi.mock("./components/JobSubmitPanel", () => ({
  JobSubmitPanel: ({ onJobSubmitted }: { onJobSubmitted: (value: object, scene: string) => void }) => (
    <div data-testid="legacy-panel">
      <button type="button" onClick={() => onJobSubmitted({}, "legacy-scene")}>Submit legacy job</button>
    </div>
  ),
}));
vi.mock("./jobs/JobsPage", () => ({
  JobsPage: ({ selectedId }: { selectedId: string | null }) => <div>Jobs content {selectedId ?? "none"}</div>,
}));
vi.mock("./newsplat/NewSplatPage", () => ({
  NewSplatPage: ({ onLocalStarted }: { onLocalStarted: (job: string, run: string) => void }) => (
    <div data-testid="new-splat">
      <button type="button" onClick={() => onLocalStarted("job42", "run42")}>Fake local start</button>
    </div>
  ),
}));
vi.mock("./coverage/CoveragePage", () => ({ CoveragePage: () => <div data-testid="coverage-page" /> }));
vi.mock("./components/SplatViewer", () => ({ SplatViewer: () => <div /> }));
vi.mock("./components/TrainingAnalytics", () => ({ TrainingAnalytics: () => <div /> }));
vi.mock("./components/NvsEvalPanel", () => ({ NvsEvalPanel: () => <div /> }));
vi.mock("./components/ConnectionSettings", () => ({ ConnectionSettings: () => null }));
vi.mock("./interactive/InteractivePage", () => ({ InteractivePage: () => <div /> }));
vi.mock("./compose/ComposePage", () => ({ ComposePage: () => <div data-testid="compose-page" /> }));

import App from "./App";

beforeEach(() => {
  window.localStorage.clear();
});

describe("App navigation", () => {
  it("opens Yeni Splat by default and a local start jumps to that job", () => {
    render(<App />);
    expect(screen.getByTestId("new-splat")).toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Fake local start" }));
    expect(screen.getByText("Jobs content job42")).toBeInTheDocument();
  });

  it("keeps Yeni Splat mounted while visiting other tabs", () => {
    render(<App />);
    const page = screen.getByTestId("new-splat");
    fireEvent.click(screen.getByRole("button", { name: "İşler" }));
    expect(page).not.toBeVisible();
    fireEvent.click(screen.getByRole("button", { name: "Yeni Splat" }));
    expect(screen.getByTestId("new-splat")).toBe(page);
    expect(page).toBeVisible();
  });

  it("reaches the other notebooks through the Diğer menu; generating never navigates", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /Diğer/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: /Diğer notebook'lar/ }));
    expect(screen.getByTestId("notebook-panel")).toBeInTheDocument();
    expect(screen.queryByRole("menu")).not.toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Generate fake notebook" }));
    expect(screen.getByTestId("notebook-panel")).toBeInTheDocument();
  });

  it("preserves the 4D local job form and its navigation to İşler", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: /Diğer/ }));
    fireEvent.click(screen.getByRole("menuitem", { name: /4D local eğitim/ }));
    expect(screen.getByTestId("legacy-panel")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Submit legacy job" }));
    expect(screen.getByText(/Jobs content/)).toBeInTheDocument();
  });

  it("opens the scene composer tab", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Sahne Editörü" }));
    expect(screen.getByTestId("compose-page")).toBeInTheDocument();
  });

  it("keeps the scene composer mounted when switching tabs", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Sahne Editörü" }));
    const page = screen.getByTestId("compose-page");
    fireEvent.click(screen.getByRole("button", { name: "İşler" }));
    expect(screen.getByText(/Jobs content/)).toBeInTheDocument();
    expect(page).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Sahne Editörü" }));
    expect(screen.getByTestId("compose-page")).toBe(page);
  });
});
