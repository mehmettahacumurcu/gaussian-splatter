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
vi.mock("./components/JobsList", () => ({ JobsList: () => <div>Jobs content</div> }));
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

describe("App notebook and legacy tabs", () => {
  it("opens Notebook by default and notebook generation never navigates", () => {
    render(<App />);
    expect(screen.getByTestId("notebook-panel")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Generate fake notebook" }));
    expect(screen.getByTestId("notebook-panel")).toBeInTheDocument();
  });

  it("preserves Local Job and legacy submission navigation", () => {
    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "Local Job" }));
    expect(screen.getByTestId("legacy-panel")).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Submit legacy job" }));
    expect(screen.getByText("Jobs content")).toBeInTheDocument();
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
    fireEvent.click(screen.getByRole("button", { name: "Jobs" }));
    expect(screen.getByText("Jobs content")).toBeInTheDocument();
    expect(page).toBeInTheDocument();
    fireEvent.click(screen.getByRole("button", { name: "Sahne Editörü" }));
    expect(screen.getByTestId("compose-page")).toBe(page);
  });
});
