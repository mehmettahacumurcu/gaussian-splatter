import { fireEvent, render, screen } from "@testing-library/react";
import { describe, expect, it, vi } from "vitest";

import { NotebookDownload } from "../NotebookDownload";

describe("NotebookDownload", () => {
  it("downloads each artifact once and revokes replacement/unmount URLs", () => {
    Object.defineProperty(URL, "createObjectURL", {
      configurable: true,
      value: () => "",
    });
    Object.defineProperty(URL, "revokeObjectURL", {
      configurable: true,
      value: () => undefined,
    });
    const create = vi
      .spyOn(URL, "createObjectURL")
      .mockReturnValueOnce("blob:first")
      .mockReturnValueOnce("blob:second");
    const revoke = vi
      .spyOn(URL, "revokeObjectURL")
      .mockImplementation(() => undefined);
    const click = vi
      .spyOn(HTMLAnchorElement.prototype, "click")
      .mockImplementation(() => {});
    const first = { blob: new Blob(["1"]), filename: "first.ipynb" };
    const second = { blob: new Blob(["2"]), filename: "second.ipynb" };
    const { rerender, unmount } = render(
      <NotebookDownload artifact={first} onRegenerate={vi.fn()} onEdit={vi.fn()} />,
    );
    expect(create).toHaveBeenCalledWith(first.blob);
    expect(click).toHaveBeenCalledTimes(1);
    rerender(
      <NotebookDownload artifact={second} onRegenerate={vi.fn()} onEdit={vi.fn()} />,
    );
    expect(revoke).toHaveBeenCalledWith("blob:first");
    expect(click).toHaveBeenCalledTimes(2);
    fireEvent.click(screen.getByRole("button", { name: "Download again" }));
    expect(click).toHaveBeenCalledTimes(3);
    unmount();
    expect(revoke).toHaveBeenCalledWith("blob:second");
    expect(revoke).toHaveBeenCalledTimes(2);
  });
});
