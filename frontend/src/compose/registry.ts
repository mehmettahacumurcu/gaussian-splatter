import type { Box3, Group, Object3D } from "three";
import type { SplatMesh } from "@sparkjsdev/spark";

/** Live three.js objects behind each scene object (filled by the renderers). */
export interface RegistryEntry {
  group: Group;
  splat?: SplatMesh;
  cropTarget?: Object3D;
  /** Splat bounds in the object's local (raw file) frame, centres only. */
  localBounds?: Box3;
}

export class ObjectRegistry {
  private entries = new Map<string, RegistryEntry>();

  set(id: string, entry: RegistryEntry): void {
    this.entries.set(id, entry);
  }

  patch(id: string, patch: Partial<RegistryEntry>): void {
    const entry = this.entries.get(id);
    if (entry) this.entries.set(id, { ...entry, ...patch });
  }

  get(id: string): RegistryEntry | undefined {
    return this.entries.get(id);
  }

  delete(id: string, group?: Group): void {
    // Only drop the entry the caller owns (a remount may already have replaced it).
    if (!group || this.entries.get(id)?.group === group) this.entries.delete(id);
  }

  all(): [string, RegistryEntry][] {
    return Array.from(this.entries.entries());
  }
}
