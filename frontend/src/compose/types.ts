/** TS mirror of backend/compose/models.py (SceneDoc v1). */
export type Vec3 = [number, number, number];
/** three.js order: x, y, z, w */
export type Quat = [number, number, number, number];

export interface Transform {
  position: Vec3;
  quaternion: Quat;
  scale: number;
}

export interface CropBox {
  center: Vec3;
  halfSize: Vec3;
  quaternion: Quat;
}

export interface ColorAdjust {
  exposure: number;
  tint: Vec3;
  saturation: number;
}

export type ObjectKind = "splat" | "mesh";
export type ViewUp = "y" | "-y";

export interface SceneObject {
  id: string;
  kind: ObjectKind;
  asset: string;
  name: string;
  role: "base" | "object";
  visible: boolean;
  transform: Transform;
  crop?: CropBox | null;
  color?: ColorAdjust | null;
}

export interface SceneDoc {
  version: 1;
  id: string;
  name: string;
  viewUp: ViewUp;
  objects: SceneObject[];
}

export interface Asset {
  id: string;
  kind: ObjectKind;
  name: string;
  size_bytes: number;
  source: "upload" | "pipeline";
  created_ts: number;
}

export interface SceneSummary {
  id: string;
  name: string;
  updated_ts: number;
  object_count: number;
}

export type GizmoMode = "translate" | "rotate" | "scale";

export const IDENTITY_TRANSFORM: Transform = { position: [0, 0, 0], quaternion: [0, 0, 0, 1], scale: 1 };
export const DEFAULT_COLOR: ColorAdjust = { exposure: 0, tint: [1, 1, 1], saturation: 1 };
